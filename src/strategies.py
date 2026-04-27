"""
Module 9 : Strategy Framework
================================
Plusieurs stratégies pluggables, comparables en backtest :

  1. BL_Quantamental   — Black-Litterman + ML (stratégie principale)
  2. MeanVariance      — Markowitz classique (baseline)
  3. EqualWeight       — Équipondéré (naïf benchmark)
  4. Momentum          — Pure momentum 12-1M
  5. MinVariance       — Minimum variance
  6. RiskParity        — Risk parity (vol égale par titre)
  7. MaxSharpe         — Max Sharpe analytique

Chaque stratégie hérite de BaseStrategy et implémente generate_weights().
"""

import numpy as np
import pandas as pd
import yaml
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
#  Base Strategy
# ══════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """Interface commune à toutes les stratégies."""

    name: str = "BaseStrategy"

    def __init__(self, config: dict):
        self.cfg = config
        self.port = config["portfolio"]
        self.max_w = self.port["max_weight_per_stock"]
        self.min_stocks = self.port["min_stocks"]

    @abstractmethod
    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str],
                         **kwargs) -> pd.Series:
        """Retourne une série de poids normalisés (somme = 1)."""

    def _apply_constraints(self, weights: np.ndarray,
                           tickers: List[str]) -> pd.Series:
        """Applique les contraintes de base : long-only, max poids."""
        w = np.maximum(weights, 0)
        w = np.minimum(w, self.max_w)
        total = w.sum()
        if total <= 0:
            w = np.ones(len(tickers)) / len(tickers)
        else:
            w /= total
        return pd.Series(w, index=tickers)

    def _covariance(self, returns: pd.DataFrame) -> np.ndarray:
        clean = returns.dropna(axis=1, how="any").dropna()
        if len(clean) < 10:
            return np.eye(len(returns.columns)) * 0.04
        lw = LedoitWolf().fit(clean.values)
        n = len(returns.columns)
        full = np.eye(n) * 0.04
        idx = [returns.columns.get_loc(c) for c in clean.columns]
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                full[ii, jj] = lw.covariance_[i, j]
        return full


# ══════════════════════════════════════════════════════════════
#  1. Equal Weight
# ══════════════════════════════════════════════════════════════

class EqualWeightStrategy(BaseStrategy):
    name = "EqualWeight"

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str], **kwargs) -> pd.Series:
        active = [t for t in tickers if t in returns.columns]
        n = len(active)
        return pd.Series(1/n, index=active) if n > 0 else pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════
#  2. Minimum Variance
# ══════════════════════════════════════════════════════════════

class MinVarianceStrategy(BaseStrategy):
    name = "MinVariance"

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str], **kwargs) -> pd.Series:
        active = [t for t in tickers if t in returns.columns]
        ret = returns[active].dropna(how="all")
        cov = self._covariance(ret)
        n = len(active)

        try:
            import cvxpy as cp
            w = cp.Variable(n)
            prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)),
                              [cp.sum(w) == 1, w >= 0, w <= self.max_w])
            prob.solve(solver=cp.ECOS)
            if prob.status == "optimal" and w.value is not None:
                return self._apply_constraints(w.value, active)
        except Exception:
            pass

        # Fallback analytique
        try:
            ones = np.ones(n)
            inv_cov = np.linalg.pinv(cov)
            w_raw = inv_cov @ ones / (ones @ inv_cov @ ones)
            return self._apply_constraints(w_raw, active)
        except Exception:
            return pd.Series(1/n, index=active)


# ══════════════════════════════════════════════════════════════
#  3. Max Sharpe (Mean-Variance)
# ══════════════════════════════════════════════════════════════

class MaxSharpeStrategy(BaseStrategy):
    name = "MaxSharpe"

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str], rf: float = 0.02,
                         **kwargs) -> pd.Series:
        active = [t for t in tickers if t in returns.columns]
        ret = returns[active].dropna(how="all")
        mu = ret.mean().values * 252
        cov = self._covariance(ret)
        n = len(active)

        try:
            import cvxpy as cp
            w = cp.Variable(n)
            port_ret = mu @ w
            port_risk = cp.quad_form(w, cov)
            sharpe_proxy = port_ret - rf - 3.0 / 2 * port_risk
            prob = cp.Problem(cp.Maximize(sharpe_proxy),
                              [cp.sum(w) == 1, w >= 0, w <= self.max_w])
            prob.solve(solver=cp.ECOS)
            if prob.status == "optimal" and w.value is not None:
                return self._apply_constraints(w.value, active)
        except Exception:
            pass

        # Fallback : max Sharpe analytique (sans contraintes)
        try:
            inv_cov = np.linalg.pinv(cov)
            excess = mu - rf
            w_raw = inv_cov @ excess
            return self._apply_constraints(w_raw, active)
        except Exception:
            return pd.Series(1/n, index=active)


# ══════════════════════════════════════════════════════════════
#  4. Momentum
# ══════════════════════════════════════════════════════════════

class MomentumStrategy(BaseStrategy):
    name = "Momentum"

    def __init__(self, config: dict, window_months: int = 12, skip_months: int = 1):
        super().__init__(config)
        self.window = window_months
        self.skip = skip_months

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str], **kwargs) -> pd.Series:
        active = [t for t in tickers if t in returns.columns]
        ret = returns[active]

        end = ret.index[-1] - pd.DateOffset(months=self.skip)
        start = end - pd.DateOffset(months=self.window)

        prices_proxy = (1 + ret).cumprod()
        p_end = prices_proxy.asof(end)
        p_start = prices_proxy.asof(start)
        momentum = (p_end / p_start - 1).dropna()

        # Top N titres par momentum
        n_select = min(self.port["max_stocks"], len(momentum))
        top = momentum.nlargest(n_select).index.tolist()

        # Poids proportionnels au momentum positif
        mom_pos = momentum[top].clip(lower=0)
        if mom_pos.sum() > 0:
            w = mom_pos / mom_pos.sum()
        else:
            w = pd.Series(1/len(top), index=top)

        return self._apply_constraints(w.reindex(active).fillna(0).values, active)


# ══════════════════════════════════════════════════════════════
#  5. Risk Parity
# ══════════════════════════════════════════════════════════════

class RiskParityStrategy(BaseStrategy):
    name = "RiskParity"

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str], **kwargs) -> pd.Series:
        active = [t for t in tickers if t in returns.columns]
        ret = returns[active].dropna(how="all")
        vols = ret.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()

        if vols.empty:
            n = len(active)
            return pd.Series(1/n, index=active)

        # Poids inversement proportionnels à la volatilité
        inv_vol = 1 / vols
        w_raw = inv_vol / inv_vol.sum()
        return self._apply_constraints(
            w_raw.reindex(active).fillna(0).values, active)


# ══════════════════════════════════════════════════════════════
#  6. BL Quantamental (stratégie principale)
# ══════════════════════════════════════════════════════════════

class BLQuantamentalStrategy(BaseStrategy):
    name = "BL_Quantamental"

    def __init__(self, config: dict, ml_engine=None, optimizer=None):
        super().__init__(config)
        self.ml_engine = ml_engine
        self.optimizer = optimizer

    def generate_weights(self, returns: pd.DataFrame,
                         tickers: List[str],
                         macro: Optional[pd.DataFrame] = None,
                         as_of: Optional[pd.Timestamp] = None,
                         **kwargs) -> pd.Series:
        if macro is None:
            macro = pd.DataFrame()
        if as_of is None:
            as_of = returns.index[-1]

        active = [t for t in tickers if t in returns.columns]

        # Signaux ML
        if self.ml_engine is not None and self.ml_engine._fitted:
            try:
                signals = self.ml_engine.generate_signals(
                    returns, macro, active, as_of=as_of)
            except Exception as e:
                logger.warning(f"BL signals échoués : {e}")
                signals = {"scores": pd.Series(0.0, index=active),
                           "regime": "bull",
                           "regime_prob": np.array([1., 0., 0.]),
                           "vol_forecast": None}
        else:
            signals = {"scores": pd.Series(0.0, index=active),
                       "regime": "bull",
                       "regime_prob": np.array([1., 0., 0.]),
                       "vol_forecast": None}

        # Optimisation BL
        if self.optimizer is not None:
            try:
                result = self.optimizer.optimize(
                    returns, signals, active,
                    vol_forecast=signals.get("vol_forecast"))
                return result["weights"].reindex(active).fillna(0)
            except Exception as e:
                logger.warning(f"BL optimize échoué : {e}")

        return pd.Series(1/len(active), index=active)


# ══════════════════════════════════════════════════════════════
#  Strategy Registry
# ══════════════════════════════════════════════════════════════

STRATEGY_REGISTRY = {
    "equal_weight": EqualWeightStrategy,
    "min_variance": MinVarianceStrategy,
    "max_sharpe": MaxSharpeStrategy,
    "momentum": MomentumStrategy,
    "risk_parity": RiskParityStrategy,
    "bl_quantamental": BLQuantamentalStrategy,
}


def get_strategy(name: str, config: dict, **kwargs) -> BaseStrategy:
    """Factory : retourne une stratégie par nom."""
    name_lower = name.lower()
    if name_lower not in STRATEGY_REGISTRY:
        raise ValueError(f"Stratégie inconnue : {name}. "
                         f"Disponibles : {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name_lower](config, **kwargs)


def compare_strategies(strategies: Dict[str, BaseStrategy],
                        returns: pd.DataFrame,
                        tickers: List[str],
                        macro: Optional[pd.DataFrame] = None,
                        train_window: int = 756,
                        ) -> Dict[str, pd.Series]:
    """
    Compare plusieurs stratégies sur les mêmes données.
    Retourne un dict {nom_stratégie: série_rendements}.
    """
    from backtest import compute_metrics

    results = {}
    for name, strategy in strategies.items():
        logger.info(f"Simulation stratégie : {name}")
        port_returns = []

        for i in range(train_window, len(returns) - 1, 21):  # mensuel
            train = returns.iloc[max(0, i-train_window):i]
            try:
                w = strategy.generate_weights(
                    train, tickers,
                    macro=macro.iloc[:i] if macro is not None else None,
                    as_of=returns.index[i])
            except Exception as e:
                logger.debug(f"{name}: {e}")
                w = pd.Series(1/len(tickers), index=tickers)

            # Rendement sur le mois suivant (21j)
            future = returns.iloc[i:i+21]
            w_aligned = w.reindex(future.columns).fillna(0)
            if w_aligned.sum() > 0:
                w_aligned /= w_aligned.sum()
            monthly_ret = (future * w_aligned).sum(axis=1)
            port_returns.extend(monthly_ret.values)

        idx = returns.index[train_window:train_window + len(port_returns)]
        results[name] = pd.Series(port_returns[:len(idx)], index=idx)

    # Résumé
    logger.info("\n── COMPARAISON STRATÉGIES ──")
    for name, rets in results.items():
        m = compute_metrics(rets, name=name)
        logger.info(f"  {name:<20} Sharpe={m.get('sharpe_ratio',0):.2f} "
                    f"Ann={m.get('annualized_return',0):.1%} "
                    f"DD={m.get('max_drawdown',0):.1%}")
    return results


if __name__ == "__main__":
    cfg = load_config()
    print("Stratégies disponibles :")
    for k in STRATEGY_REGISTRY:
        print(f"  • {k}")
