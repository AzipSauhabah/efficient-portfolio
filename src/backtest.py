"""
Module 8 : Backtest Engine — Walk-Forward
==========================================
Backtest walk-forward complet sur 20 ans avec :
- Anti-lookahead bias strict (train → test, jamais l'inverse)
- Métriques complètes : Sharpe, Sortino, Max DD, Calmar, VaR, CVaR
- Comparaison vs benchmarks (Euro Stoxx 50, S&P 500)
- Décomposition des coûts (frais, taxes, purification)
- Export des résultats pour visualisation Colab
"""

import numpy as np
import pandas as pd
import yaml
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
#  Métriques
# ══════════════════════════════════════════════════════════════

def compute_metrics(returns: pd.Series, rf: float = 0.02,
                    name: str = "Portfolio") -> Dict:
    """Calcule toutes les métriques de performance."""
    r = returns.dropna()
    if len(r) < 20:
        return {}

    ann = 252
    total_ret = (1 + r).prod() - 1
    ann_ret = (1 + total_ret) ** (ann / len(r)) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (ann_ret - rf) / vol if vol > 0 else 0

    # Sortino (downside risk seulement)
    downside = r[r < 0].std() * np.sqrt(ann)
    sortino = (ann_ret - rf) / downside if downside > 0 else 0

    # Max Drawdown
    cum = (1 + r).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # VaR & CVaR (95%)
    var_95 = float(np.percentile(r, 5))
    cvar_95 = float(r[r <= var_95].mean())

    # Skewness / Kurtosis
    skew = float(r.skew())
    kurt = float(r.kurt())

    return {
        "name": name,
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "n_days": len(r),
        "positive_days_pct": (r > 0).mean(),
    }


# ══════════════════════════════════════════════════════════════
#  Backtest Engine
# ══════════════════════════════════════════════════════════════

class WalkForwardBacktest:
    """
    Backtest walk-forward : entraîne sur N ans, teste sur K mois, avance.
    Garantit qu'aucune donnée future n'est utilisée pendant l'optimisation.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.bt_cfg = config["backtest"]
        self.train_years = self.bt_cfg["train_window_years"]
        self.test_months = self.bt_cfg["test_window_months"]
        self.capital = config["portfolio"]["initial_capital"]

    def run(self, data: Dict, tickers: List[str],
            ml_engine=None, optimizer=None,
            rebalance_engine=None, rl_agent=None) -> Dict:
        """
        Lance le backtest walk-forward complet.

        Paramètres
        ----------
        data         : dict de DataPipeline.prepare()
        tickers      : liste des tickers Shariah
        ml_engine    : MLEngine (peut être None → signaux nuls)
        optimizer    : BlackLittermanOptimizer
        rebalance_engine : RebalancingEngine
        rl_agent     : RLRebalancingAgent (peut être None)

        Retour
        ------
        dict : portfolio_returns, weights_history, metrics, cost_history, benchmarks
        """
        returns = data["returns"]
        prices = data["prices"]
        macro = data["macro"]

        # Dates de début/fin du backtest
        start = pd.Timestamp(self.cfg["data"]["start_date"])
        end = pd.Timestamp(self.cfg["data"]["end_date"])

        # Fenêtres walk-forward
        windows = self._build_windows(returns, start, end)
        logger.info(f"Walk-forward : {len(windows)} fenêtres | "
                    f"train={self.train_years}Y test={self.test_months}M")

        # Historiques
        port_returns = pd.Series(dtype=float, name="Shariah_Portfolio")
        weights_history = {}
        cost_history = []
        regime_history = {}

        # Région map
        region_map = {}
        for t in tickers:
            region_map[t] = "US" if "." not in t else "EU"

        # Initialisation portefeuille
        if rebalance_engine is None:
            from costs import RebalancingEngine
            rebalance_engine = RebalancingEngine(self.cfg)

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n── Fenêtre {i+1}/{len(windows)} : "
                        f"train [{train_start.date()} → {train_end.date()}] "
                        f"test [{test_start.date()} → {test_end.date()}]")

            # Données d'entraînement
            train_returns = returns.loc[train_start:train_end, :]
            train_returns = train_returns[[t for t in tickers if t in train_returns.columns]]

            if train_returns.empty or len(train_returns) < 100:
                logger.warning(f"  Fenêtre {i+1} : données insuffisantes, skip")
                continue

            # Entraînement ML sur la fenêtre train
            if ml_engine is not None and not ml_engine._fitted:
                try:
                    ml_engine.fit(train_returns, macro.loc[train_start:train_end],
                                  list(train_returns.columns))
                except Exception as e:
                    logger.warning(f"  ML fit échoué : {e}")

            # Test out-of-sample
            test_returns = returns.loc[test_start:test_end, :]
            test_prices = prices.loc[test_start:test_end, :]

            if test_returns.empty:
                continue

            # Génération signaux ML à la date test_start
            try:
                if ml_engine is not None and ml_engine._fitted:
                    signals = ml_engine.generate_signals(
                        returns.loc[:test_start],
                        macro.loc[:test_start],
                        list(train_returns.columns),
                        as_of=test_start,
                    )
                else:
                    signals = {"scores": pd.Series(0.0, index=tickers),
                               "regime": "bull",
                               "regime_prob": np.array([1., 0., 0.]),
                               "vol_forecast": None}
            except Exception as e:
                logger.warning(f"  Signaux ML échoués : {e}")
                signals = {"scores": pd.Series(0.0, index=tickers),
                           "regime": "bull",
                           "regime_prob": np.array([1., 0., 0.]),
                           "vol_forecast": None}

            regime_history[test_start] = signals.get("regime", "unknown")

            # Optimisation BL
            try:
                if optimizer is not None:
                    opt_result = optimizer.optimize(
                        train_returns, signals,
                        list(train_returns.columns),
                        vol_forecast=signals.get("vol_forecast"),
                    )
                    target_weights = opt_result["weights"]
                else:
                    target_weights = pd.Series(
                        1/len(tickers), index=tickers)
            except Exception as e:
                logger.warning(f"  Optimisation BL échouée : {e} — poids égaux")
                target_weights = pd.Series(1/len(tickers), index=tickers)

            weights_history[test_start] = target_weights

            # Simulation de la période de test
            window_returns = self._simulate_period(
                test_returns, test_prices, target_weights, region_map,
                rebalance_engine, rl_agent, macro, signals, tickers
            )

            if window_returns is not None and len(window_returns) > 0:
                port_returns = pd.concat([port_returns, window_returns])

            # Log coûts
            if rebalance_engine is not None:
                summary = rebalance_engine.summary(
                    test_prices.iloc[-1] if not test_prices.empty else pd.Series())
                cost_history.append({**summary, "window": i+1, "date": test_start})

        # Benchmarks
        benchmarks = self._fetch_benchmarks(start, end)

        # Métriques finales
        port_returns = port_returns[~port_returns.index.duplicated(keep="last")]
        port_returns = port_returns.sort_index()

        metrics = compute_metrics(port_returns, name="Shariah Portfolio")

        # Métriques benchmarks
        bench_metrics = {}
        for name, bret in benchmarks.items():
            bench_metrics[name] = compute_metrics(bret, name=name)

        logger.info("\n══ RÉSULTATS BACKTEST ══")
        logger.info(f"Rendement total    : {metrics.get('total_return', 0):.1%}")
        logger.info(f"Rendement annualisé: {metrics.get('annualized_return', 0):.1%}")
        logger.info(f"Sharpe ratio       : {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown       : {metrics.get('max_drawdown', 0):.1%}")

        return {
            "portfolio_returns": port_returns,
            "weights_history": weights_history,
            "metrics": metrics,
            "bench_metrics": bench_metrics,
            "cost_history": pd.DataFrame(cost_history),
            "regime_history": pd.Series(regime_history),
            "benchmarks": benchmarks,
            "trade_history": (rebalance_engine.get_trade_history()
                              if rebalance_engine else pd.DataFrame()),
        }

    def _simulate_period(self, test_returns, test_prices, target_weights,
                         region_map, rebalance_engine, rl_agent,
                         macro, signals, tickers) -> Optional[pd.Series]:
        """Simule la performance sur une fenêtre de test."""
        if test_returns.empty:
            return None

        daily_returns = []
        current_weights = target_weights.copy()

        # Initialisation si première fenêtre
        if not rebalance_engine.positions:
            first_prices = test_prices.iloc[0]
            rebalance_engine.initialize(target_weights, first_prices,
                                        test_returns.index[0], region_map)

        for date in test_returns.index:
            day_ret = test_returns.loc[date]
            prices_today = test_prices.loc[date] if date in test_prices.index else pd.Series()

            # Rendement du portefeuille ce jour
            port_ret = 0.0
            for ticker, w in current_weights.items():
                if ticker in day_ret.index and pd.notna(day_ret[ticker]):
                    port_ret += w * day_ret[ticker]

            daily_returns.append(port_ret)

            # Mise à jour des poids (drift marché)
            for ticker in list(current_weights.index):
                if ticker in day_ret.index and pd.notna(day_ret[ticker]):
                    current_weights[ticker] *= (1 + day_ret[ticker])
            total_w = current_weights.sum()
            if total_w > 0:
                current_weights /= total_w

            # Décision de rebalancing
            if not prices_today.empty and rebalance_engine is not None:
                if rl_agent is not None and rl_agent.model is not None:
                    # Agent RL décide
                    env = PortfolioEnvLite(current_weights, signals, rebalance_engine)
                    obs = env.get_obs()
                    action, label = rl_agent.decide(obs)
                    if action > 0:
                        result = rebalance_engine.execute_rebalance(
                            target_weights, prices_today, date, region_map)
                        current_weights = target_weights.copy()
                else:
                    # Règles cost-aware classiques
                    should, reason = rebalance_engine.should_rebalance(
                        target_weights, prices_today, date, region_map)
                    if should:
                        rebalance_engine.execute_rebalance(
                            target_weights, prices_today, date, region_map)
                        current_weights = target_weights.copy()

        if not daily_returns:
            return None

        return pd.Series(daily_returns, index=test_returns.index, name="portfolio")

    def _build_windows(self, returns: pd.DataFrame,
                       start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple]:
        """Génère les fenêtres train/test walk-forward."""
        windows = []
        train_start = start
        train_end = start + pd.DateOffset(years=self.train_years)

        while train_end < end:
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months)
            test_end = min(test_end, end)

            windows.append((train_start, train_end, test_start, test_end))

            # Avance d'une fenêtre de test
            train_end = test_end
        return windows

    def _fetch_benchmarks(self, start: pd.Timestamp,
                          end: pd.Timestamp) -> Dict[str, pd.Series]:
        """Télécharge les benchmarks pour comparaison."""
        benchmarks = {}
        bench_tickers = self.bt_cfg.get("benchmark_tickers", ["^GSPC"])

        for ticker in bench_tickers:
            try:
                import yfinance as yf
                h = yf.Ticker(ticker).history(
                    start=start, end=end, auto_adjust=True)
                if not h.empty:
                    r = h["Close"].pct_change().dropna()
                    name = ticker.replace("^", "").replace(".", "_")
                    benchmarks[name] = r
            except Exception as e:
                logger.debug(f"Benchmark {ticker}: {e}")

        return benchmarks


class PortfolioEnvLite:
    """Version légère de l'env pour inférence RL en backtest."""
    def __init__(self, weights, signals, rebalance_engine):
        self.weights = weights
        self.signals = signals
        self.engine = rebalance_engine

    def get_obs(self) -> np.ndarray:
        n = len(self.weights)
        target = np.ones(n) / n
        w = self.weights.values if hasattr(self.weights, 'values') else np.array(list(self.weights.values()))
        drift = w - target
        regime_p = self.signals.get("regime_prob", np.array([1., 0., 0.]))
        days = (self.engine.days_since_rebalance
                if hasattr(self.engine, 'days_since_rebalance') else 0)
        obs = np.concatenate([w, drift, regime_p,
                               [days/365.], [0., 0.], [0.]]).astype(np.float32)
        return np.clip(obs, -5., 5.)


if __name__ == "__main__":
    cfg = load_config()
    print("Backtest engine chargé ✅")
    print(f"Train window : {cfg['backtest']['train_window_years']} ans")
    print(f"Test window  : {cfg['backtest']['test_window_months']} mois")
