"""
Module 6 : Costs & Rebalancing Engine
=======================================
- Calcul des frais de transaction par broker (Fortuneo / Boursorama / IBKR / Swissquote)
- Slippage fonction de la taille de l'ordre
- Tracking PRU (Prix de Revient Unitaire) pour flat tax CTO France
- Décision de rebalancing cost-aware : ne rebalance que si gain net > coûts
- Regroupement des ordres US (batch) pour minimiser les frais fixes
"""

import numpy as np
import pandas as pd
import yaml
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class Position:
    """Représente une position avec tracking PRU pour la fiscalité."""
    ticker: str
    quantity: float = 0.0
    avg_cost: float = 0.0          # PRU (Prix de Revient Unitaire)
    region: str = "EU"             # "EU" | "US"
    total_cost_basis: float = 0.0  # cost_basis total pour FIFO
    realized_pnl: float = 0.0
    tax_paid: float = 0.0

    def update_buy(self, qty: float, price: float, fees: float):
        """Met à jour le PRU après un achat."""
        total_new = qty * price + fees
        self.avg_cost = (self.total_cost_basis + total_new) / (self.quantity + qty)
        self.quantity += qty
        self.total_cost_basis += total_new

    def update_sell(self, qty: float, price: float, fees: float,
                    flat_tax: float = 0.30) -> float:
        """
        Met à jour la position après une vente.
        Calcule et déduit la flat tax française sur la plus-value.
        Retourne le montant net perçu après taxe.
        """
        if qty <= 0 or self.quantity <= 0:
            return 0.0

        qty = min(qty, self.quantity)
        gross_proceeds = qty * price - fees
        cost_basis_sold = self.avg_cost * qty

        pnl = gross_proceeds - cost_basis_sold
        tax = max(pnl * flat_tax, 0.0)   # taxe seulement sur plus-values

        net_proceeds = gross_proceeds - tax
        self.realized_pnl += pnl
        self.tax_paid += tax
        self.quantity -= qty
        self.total_cost_basis -= cost_basis_sold

        if self.quantity <= 0:
            self.quantity = 0.0
            self.total_cost_basis = 0.0
            self.avg_cost = 0.0

        return net_proceeds


class BrokerCosts:
    """Calcule les frais de transaction pour le broker sélectionné."""

    def __init__(self, config: dict):
        self.broker_name = config["broker"]["name"]
        self.broker_cfg = config["broker"]["brokers"][self.broker_name]
        self.slippage_cfg = config["broker"]["slippage"]
        self.flat_tax = config["taxation"]["flat_tax_pct"]

    def transaction_cost(self, amount_eur: float, region: str,
                         price_per_share: Optional[float] = None,
                         is_large_cap: bool = True) -> float:
        """
        Calcule le coût total d'une transaction (frais broker + slippage).

        Paramètres
        ----------
        amount_eur     : montant de la transaction en €
        region         : "EU" ou "US"
        price_per_share: prix unitaire (pour brokers à frais par action)
        is_large_cap   : détermine le slippage

        Retour
        ------
        coût total en €
        """
        b = self.broker_cfg

        if region == "EU":
            fee_fixed = b.get("eu_fixed_fee", 0.0)
            fee_pct = b.get("eu_pct_fee", 0.0) * amount_eur
            fee = max(fee_fixed + fee_pct, b.get("min_fee", 0.0))
        else:  # US
            fee_fixed = b.get("us_fixed_fee", 0.0)
            fee_pct = b.get("us_pct_fee", 0.0) * amount_eur
            fee_min = b.get("us_min_fee", fee_fixed)
            fee = max(fee_fixed + fee_pct, fee_min, b.get("min_fee", 0.0))

        # Slippage
        slip_pct = (self.slippage_cfg["large_cap_pct"] if is_large_cap
                    else self.slippage_cfg["mid_cap_pct"])
        slippage = amount_eur * slip_pct

        return fee + slippage

    def annual_custody_fee(self, portfolio_value: float) -> float:
        """Droits de garde annuels (0 pour Fortuneo/Boursorama, fixe pour Swissquote)."""
        return self.broker_cfg.get("custody_annual_fee", 0.0)

    def rebalance_cost_estimate(self, trades: Dict[str, float],
                                prices: Dict[str, float],
                                regions: Dict[str, str]) -> float:
        """Estime le coût total d'un ensemble de trades avant de les exécuter."""
        total = 0.0
        for ticker, amount in trades.items():
            region = regions.get(ticker, "EU")
            total += self.transaction_cost(abs(amount), region)
        return total

    def breakeven_drift(self, position_value: float, region: str) -> float:
        """
        Dérive minimale nécessaire pour rentabiliser un rebalancing.
        breakeven = coût_transaction / valeur_position
        """
        cost = self.transaction_cost(position_value, region)
        tax_drag = self.flat_tax * 0.05   # hypothèse : 5% de PV latente taxée
        return (cost + tax_drag * position_value) / position_value if position_value > 0 else 1.0


class RebalancingEngine:
    """
    Moteur de rebalancing cost-aware.

    Ne déclenche un trade que si : gain espéré net > coûts totaux
    Regroupe les ordres US pour minimiser les frais fixes Fortuneo (50€).
    """

    def __init__(self, config: dict):
        self.cfg = config["rebalancing"]
        self.port_cfg = config["portfolio"]
        self.broker = BrokerCosts(config)
        self.flat_tax = config["taxation"]["flat_tax_pct"]
        self.positions: Dict[str, Position] = {}
        self.cash = config["portfolio"]["initial_capital"]
        self.last_rebalance: Optional[pd.Timestamp] = None
        self.trade_history: List[dict] = []

    def initialize(self, target_weights: pd.Series, prices: pd.Series,
                   date: pd.Timestamp, region_map: Dict[str, str]):
        """Premier investissement — construction initiale du portefeuille."""
        portfolio_value = self.cash
        logger.info(f"Initialisation portefeuille : {portfolio_value:,.0f}€")

        # Trie : EU d'abord (moins cher), US ensuite (groupés)
        eu_tickers = [t for t in target_weights.index
                      if region_map.get(t, "EU") == "EU" and target_weights[t] > 0.001]
        us_tickers = [t for t in target_weights.index
                      if region_map.get(t, "US") == "US" and target_weights[t] > 0.001]

        total_cost = 0.0
        for ticker in eu_tickers + us_tickers:
            if ticker not in prices.index or pd.isna(prices[ticker]):
                continue
            region = region_map.get(ticker, "EU")
            target_amount = portfolio_value * target_weights.get(ticker, 0)

            if target_amount < self.port_cfg["min_position_size"]:
                continue

            cost = self.broker.transaction_cost(target_amount, region)
            total_cost += cost
            qty = target_amount / prices[ticker]

            pos = Position(ticker=ticker, region=region)
            pos.update_buy(qty, prices[ticker], cost)
            self.positions[ticker] = pos

            self.trade_history.append({
                "date": date, "ticker": ticker, "action": "BUY",
                "amount": target_amount, "fee": cost, "region": region,
            })

        self.cash -= (sum(target_weights.get(t, 0) * portfolio_value
                         for t in eu_tickers + us_tickers) + total_cost)
        self.last_rebalance = date
        logger.info(f"Portefeuille initialisé — {len(self.positions)} positions, "
                    f"frais totaux : {total_cost:.0f}€")

    def portfolio_value(self, prices: pd.Series) -> float:
        """Valeur totale du portefeuille."""
        val = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices.index and pd.notna(prices[ticker]):
                val += pos.quantity * prices[ticker]
        return val

    def current_weights(self, prices: pd.Series) -> pd.Series:
        """Poids actuels du portefeuille."""
        pv = self.portfolio_value(prices)
        if pv <= 0:
            return pd.Series(dtype=float)
        weights = {}
        for ticker, pos in self.positions.items():
            if ticker in prices.index and pd.notna(prices[ticker]):
                weights[ticker] = pos.quantity * prices[ticker] / pv
        return pd.Series(weights)

    def should_rebalance(self, target_weights: pd.Series, prices: pd.Series,
                         date: pd.Timestamp, region_map: Dict[str, str]) -> Tuple[bool, str]:
        """
        Décide si on rebalance. Logique cost-aware :
        1. Jamais plus de max_frequency_days entre rebalancings
        2. Jamais moins de min_frequency_days
        3. Rebalance seulement si dérive > seuil ET gain net > coûts
        """
        if self.last_rebalance is None:
            return True, "initial"

        days_since = (date - self.last_rebalance).days

        # Forcer si trop longtemps sans rebalancer
        if days_since >= self.cfg["min_frequency_days"]:
            return True, "forced_annual"

        # Ne pas rebalancer trop souvent
        if days_since < self.cfg["max_frequency_days"]:
            return False, "too_recent"

        # Calcul de la dérive
        current_w = self.current_weights(prices)
        pv = self.portfolio_value(prices)
        trades = self._compute_trades(target_weights, current_w, pv, prices, region_map)

        if not trades:
            return False, "no_significant_drift"

        # Estimation du coût du rebalancing
        total_cost = self.broker.rebalance_cost_estimate(trades, prices.to_dict(), region_map)
        total_trade_amount = sum(abs(v) for v in trades.values())

        # Gain espéré du rebalancing (approximation conservative)
        drift_total = sum(abs(target_weights.get(t, 0) - current_w.get(t, 0))
                         for t in set(target_weights.index) | set(current_w.index))

        expected_gain = drift_total * pv * 0.02   # hypothèse : 2% alpha par unité de dérive
        min_gain = self.cfg["min_net_gain_eur"]

        if total_cost > expected_gain or total_cost > min_gain * 3:
            return False, f"not_profitable (coût:{total_cost:.0f}€ > gain:{expected_gain:.0f}€)"

        if drift_total < self.cfg["min_drift_pct"]:
            return False, f"drift_too_small ({drift_total:.1%})"

        return True, f"drift:{drift_total:.1%} cost:{total_cost:.0f}€"

    def _compute_trades(self, target_w: pd.Series, current_w: pd.Series,
                        pv: float, prices: pd.Series,
                        region_map: Dict[str, str]) -> Dict[str, float]:
        """Calcule les montants à trader (positif = achat, négatif = vente)."""
        all_tickers = set(target_w.index) | set(current_w.index)
        trades = {}

        for ticker in all_tickers:
            tw = target_w.get(ticker, 0.0)
            cw = current_w.get(ticker, 0.0)
            drift = tw - cw

            if abs(drift) < self.cfg["min_drift_pct"] / 2:
                continue

            amount = drift * pv
            region = region_map.get(ticker, "EU")
            cost = self.broker.transaction_cost(abs(amount), region)

            # Ne trade que si le montant justifie les frais
            breakeven = cost / abs(amount) if amount != 0 else 1.0
            if abs(drift) > breakeven:
                trades[ticker] = amount

        return trades

    def execute_rebalance(self, target_weights: pd.Series, prices: pd.Series,
                          date: pd.Timestamp, region_map: Dict[str, str]) -> Dict:
        """
        Exécute le rebalancing :
        1. Ventes d'abord (libère du cash)
        2. Achats ensuite
        3. Groupe les ordres US si batch activé
        """
        current_w = self.current_weights(prices)
        pv = self.portfolio_value(prices)
        trades = self._compute_trades(target_weights, current_w, pv, prices, region_map)

        if not trades:
            return {"trades": {}, "total_fees": 0.0, "total_tax": 0.0}

        sells = {t: v for t, v in trades.items() if v < 0}
        buys = {t: v for t, v in trades.items() if v > 0}

        total_fees = 0.0
        total_tax = 0.0

        # Ventes
        for ticker, amount in sells.items():
            if ticker not in self.positions or ticker not in prices.index:
                continue
            pos = self.positions[ticker]
            region = region_map.get(ticker, "EU")
            fee = self.broker.transaction_cost(abs(amount), region)
            qty_to_sell = abs(amount) / prices[ticker]

            net = pos.update_sell(qty_to_sell, prices[ticker], fee, self.flat_tax)
            self.cash += net
            total_fees += fee
            total_tax += pos.tax_paid  # approximatif

            self.trade_history.append({
                "date": date, "ticker": ticker, "action": "SELL",
                "amount": abs(amount), "fee": fee, "region": region,
                "pnl": pos.realized_pnl,
            })
            logger.info(f"  VENTE {ticker} : {abs(amount):,.0f}€ | frais: {fee:.0f}€")

        # Achats (groupés US si batch activé)
        if self.cfg.get("batch_us_orders", True):
            us_buys = {t: v for t, v in buys.items() if region_map.get(t, "EU") == "US"}
            eu_buys = {t: v for t, v in buys.items() if region_map.get(t, "EU") == "EU"}
            ordered_buys = list(eu_buys.items()) + list(us_buys.items())
        else:
            ordered_buys = list(buys.items())

        for ticker, amount in ordered_buys:
            if ticker not in prices.index or pd.isna(prices[ticker]):
                continue
            region = region_map.get(ticker, "EU")
            fee = self.broker.transaction_cost(amount, region)

            if self.cash < amount + fee:
                amount = max(self.cash - fee - 10, 0)   # garde 10€ de marge
                if amount < self.port_cfg["min_position_size"]:
                    logger.debug(f"  SKIP {ticker} : cash insuffisant")
                    continue

            qty = amount / prices[ticker]
            if ticker not in self.positions:
                self.positions[ticker] = Position(ticker=ticker, region=region)
            self.positions[ticker].update_buy(qty, prices[ticker], fee)
            self.cash -= (amount + fee)
            total_fees += fee

            self.trade_history.append({
                "date": date, "ticker": ticker, "action": "BUY",
                "amount": amount, "fee": fee, "region": region,
            })
            logger.info(f"  ACHAT {ticker} : {amount:,.0f}€ | frais: {fee:.0f}€")

        self.last_rebalance = date
        logger.info(f"Rebalancing terminé — frais totaux : {total_fees:.0f}€, taxe : {total_tax:.0f}€")

        return {
            "trades": trades,
            "total_fees": total_fees,
            "total_tax": total_tax,
            "portfolio_value_after": self.portfolio_value(prices),
        }

    def get_trade_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_history)

    def summary(self, prices: pd.Series) -> Dict:
        pv = self.portfolio_value(prices)
        total_invested = sum(pos.total_cost_basis for pos in self.positions.values())
        total_pnl = pv - total_invested - self.cash
        total_fees = sum(r.get("fee", 0) for r in self.trade_history)
        total_tax = sum(pos.tax_paid for pos in self.positions.values())

        return {
            "portfolio_value": pv,
            "cash": self.cash,
            "n_positions": len([p for p in self.positions.values() if p.quantity > 0]),
            "total_fees_paid": total_fees,
            "total_tax_paid": total_tax,
            "total_pnl": total_pnl,
            "fee_drag_pct": total_fees / total_invested if total_invested > 0 else 0,
        }


if __name__ == "__main__":
    cfg = load_config()
    broker = BrokerCosts(cfg)
    print(f"Broker : {cfg['broker']['name'].upper()}")
    print(f"Frais achat EU 2000€  : {broker.transaction_cost(2000, 'EU'):.2f}€")
    print(f"Frais achat US 2000€  : {broker.transaction_cost(2000, 'US'):.2f}€")
    print(f"Frais achat US 5000€  : {broker.transaction_cost(5000, 'US'):.2f}€")
    print(f"Breakeven drift EU    : {broker.breakeven_drift(2000, 'EU'):.2%}")
    print(f"Breakeven drift US    : {broker.breakeven_drift(2000, 'US'):.2%}")
