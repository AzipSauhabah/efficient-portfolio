"""
Module 5 : Optimizer — Black-Litterman + Covariance Ensemble
=============================================================
1. Covariance Ensemble : Ledoit-Wolf + DCC-GARCH + Fama-French 5F
2. Black-Litterman : prior Shariah index + views ML → poids optimaux
3. Contraintes : poids max, sectoriel, nb min titres, long-only
"""

import numpy as np
import pandas as pd
import yaml
import logging
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
#  Covariance Ensemble
# ══════════════════════════════════════════════════════════════

class CovarianceEngine:
    def __init__(self, config: dict):
        self.cfg = config["covariance"]
        self.weights = self.cfg["ensemble_weights"]

    def ledoit_wolf(self, returns: pd.DataFrame) -> np.ndarray:
        clean = returns.dropna(axis=1, how="any").dropna()
        lw = LedoitWolf().fit(clean.values)
        return self._reindex(lw.covariance_, clean.columns, returns.columns)

    def dcc_garch(self, returns: pd.DataFrame) -> np.ndarray:
        """DCC-GARCH(1,1) simplifié."""
        clean = returns.dropna(axis=1, how="any").dropna()
        n, T = clean.shape[1], clean.shape[0]
        alpha_g, beta_g, omega = 0.05, 0.90, 1e-6

        eps = clean.values.copy()
        h = np.full((T, n), np.var(eps, axis=0))
        for t in range(1, T):
            h[t] = omega + alpha_g * eps[t-1]**2 + beta_g * h[t-1]
            h[t] = np.maximum(h[t], 1e-8)

        std_r = eps / np.sqrt(h)
        Q_bar = np.corrcoef(std_r.T)
        Q = Q_bar.copy()
        alpha_d, beta_d = 0.01, 0.97

        for t in range(1, T):
            e = std_r[t].reshape(-1, 1)
            Q = (1 - alpha_d - beta_d) * Q_bar + alpha_d * (e @ e.T) + beta_d * Q

        d = np.diag(1 / np.sqrt(np.maximum(np.diag(Q), 1e-8)))
        R = d @ Q @ d
        D = np.diag(np.sqrt(h[-1]))
        cov = D @ R @ D
        return self._reindex(cov, clean.columns, returns.columns)

    def fama_french(self, returns: pd.DataFrame) -> np.ndarray:
        """Approx FF5 via PCA à 5 composantes (proxy facteurs)."""
        clean = returns.dropna(axis=1, how="any").dropna()
        from sklearn.decomposition import PCA
        n_factors = min(5, clean.shape[1] - 1, clean.shape[0] - 1)
        pca = PCA(n_components=n_factors)
        F = pca.fit_transform(clean.values)           # T x K facteurs
        B = pca.components_.T                          # N x K loadings
        Sigma_F = np.cov(F.T)
        systematic = B @ Sigma_F @ B.T
        residuals = clean.values - F @ B.T
        D = np.diag(np.var(residuals, axis=0))
        cov = systematic + D
        return self._reindex(cov, clean.columns, returns.columns)

    def _reindex(self, cov: np.ndarray, src_cols, tgt_cols) -> np.ndarray:
        """Réindexe une matrice de covariance sur tous les tickers."""
        n = len(tgt_cols)
        full = np.eye(n) * 0.04   # variance par défaut 20%²
        idx = {c: i for i, c in enumerate(tgt_cols)}
        src_idx = [idx[c] for c in src_cols if c in idx]
        for i, ii in enumerate(src_idx):
            for j, jj in enumerate(src_idx):
                full[ii, jj] = cov[i, j]
        return full

    def ensemble(self, returns: pd.DataFrame,
                 vol_forecast: Optional[float] = None) -> np.ndarray:
        """Covariance ensemble pondérée des 3 estimateurs."""
        method = self.cfg["method"]
        if method == "ledoit_wolf":
            cov = self.ledoit_wolf(returns)
        elif method == "dcc_garch":
            cov = self.dcc_garch(returns)
        elif method == "fama_french":
            cov = self.fama_french(returns)
        else:  # ensemble
            w = self.weights
            cov = (
                w["ledoit_wolf"] * self.ledoit_wolf(returns) +
                w["dcc_garch"] * self.dcc_garch(returns) +
                w["fama_french"] * self.fama_french(returns)
            )

        # Ajustement diagonal si LSTM a prévu une vol différente
        if vol_forecast is not None:
            current_vol = np.sqrt(np.trace(cov) / len(cov))
            if current_vol > 0:
                scale = (vol_forecast / current_vol) ** 2
                cov = cov * np.clip(scale, 0.5, 2.0)

        # Regularisation PD
        cov = self._make_pd(cov)
        return cov

    @staticmethod
    def _make_pd(cov: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """Force la matrice à être définie positive."""
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals.min() < epsilon:
            cov += (-eigvals.min() + epsilon) * np.eye(len(cov))
        return cov


# ══════════════════════════════════════════════════════════════
#  Black-Litterman
# ══════════════════════════════════════════════════════════════

class BlackLittermanOptimizer:
    """
    Optimiseur Black-Litterman avec views générées par le ML Engine.

    Prior : rendements implicites d'équilibre basés sur les capitalisations
            de l'univers Shariah (pas du marché global).
    Views : scores ML traduits en rendements relatifs attendus.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.bl = config["black_litterman"]
        self.port = config["portfolio"]
        self.tau = self.bl["tau"]
        self.risk_aversion = self.bl["risk_aversion"]
        self.cov_engine = CovarianceEngine(config)

    def _market_weights(self, tickers: List[str]) -> np.ndarray:
        """Poids de capitalisation boursière de l'univers Shariah (prior)."""
        try:
            import yfinance as yf
            caps = {}
            for t in tickers:
                info = yf.Ticker(t).info
                caps[t] = info.get("marketCap", 1e9) or 1e9
            w = np.array([caps[t] for t in tickers], dtype=float)
            return w / w.sum()
        except Exception:
            return np.ones(len(tickers)) / len(tickers)

    def _implied_returns(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Rendements implicites d'équilibre : π = λ * Σ * w_mkt."""
        return self.risk_aversion * cov @ weights

    def _build_views(self, ml_scores: pd.Series, tickers: List[str],
                     implied_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Traduit les scores ML [-1, +1] en matrices P, Q, Omega BL.

        View : "ticker i surperforme la moyenne de l'univers de X%"
        P    : matrice de vues (nb_views x nb_assets)
        Q    : rendements attendus des vues
        Omega: incertitude des vues (diagonale)
        """
        n = len(tickers)
        views = []

        for i, ticker in enumerate(tickers):
            score = ml_scores.get(ticker, 0.0)
            if abs(score) < 0.1:   # filtre les signaux faibles
                continue

            # View relative : ticker vs univers (1 longue, reste courte)
            p_row = -np.ones(n) / n
            p_row[i] = 1 - 1/n

            # Magnitude du rendement attendu (calibrée sur volatilité historique)
            vol_i = np.sqrt(implied_returns[i]**2 + 0.01)
            q_val = score * vol_i * 0.5   # signal → rendement en %

            views.append((p_row, q_val, abs(score)))

        if not views:
            # Pas de views significatives → prior pur
            P = np.zeros((1, n))
            Q = np.zeros(1)
            Omega = np.eye(1) * 1.0
            return P, Q, Omega

        P = np.vstack([v[0] for v in views])
        Q = np.array([v[1] for v in views])
        # Omega proportionnel à l'incertitude du signal (1/|score|)
        certainty = np.array([v[2] for v in views])
        Omega = np.diag(self.tau / (certainty + 1e-6))

        return P, Q, Omega

    def optimize(self, returns: pd.DataFrame, ml_signals: Dict,
                 tickers: List[str], vol_forecast: Optional[float] = None,
                 sector_map: Optional[Dict] = None) -> Dict:
        """
        Optimisation Black-Litterman complète.

        Retour
        ------
        dict : weights, expected_returns, cov, diagnostics
        """
        n = len(tickers)
        ret_tickers = [t for t in tickers if t in returns.columns]

        if len(ret_tickers) < 3:
            logger.warning("Trop peu de tickers — poids égaux")
            return {"weights": pd.Series(1/n, index=tickers),
                    "expected_returns": pd.Series(0.0, index=tickers),
                    "cov": np.eye(n) * 0.04}

        ret_sub = returns[ret_tickers].dropna(how="all")

        # Covariance ensemble
        logger.info("Calcul covariance ensemble...")
        cov_sub = self.cov_engine.ensemble(ret_sub, vol_forecast)
        tickers_sub = ret_tickers

        # Prior : poids marché Shariah
        w_mkt = self._market_weights(tickers_sub)
        pi = self._implied_returns(w_mkt, cov_sub)

        # Views ML
        ml_scores = ml_signals.get("scores", pd.Series(dtype=float))
        ml_scores_aligned = ml_scores.reindex(tickers_sub).fillna(0.0)
        P, Q, Omega = self._build_views(ml_scores_aligned, tickers_sub, pi)

        # Black-Litterman posterior
        tau_sigma = self.tau * cov_sub
        if P.shape[0] > 0 and Q.sum() != 0:
            # μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
            try:
                A = np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
                b = np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(Omega) @ Q
                mu_bl = np.linalg.solve(A, b)
                sigma_bl = np.linalg.inv(A) + cov_sub
            except np.linalg.LinAlgError:
                mu_bl = pi.copy()
                sigma_bl = cov_sub + tau_sigma
        else:
            mu_bl = pi.copy()
            sigma_bl = cov_sub + tau_sigma

        # Optimisation quadratique avec contraintes
        weights = self._qp_optimize(mu_bl, sigma_bl, tickers_sub, sector_map)

        # Résultat
        w_series = pd.Series(0.0, index=tickers)
        for i, t in enumerate(tickers_sub):
            w_series[t] = weights[i]

        exp_ret = pd.Series(0.0, index=tickers)
        for i, t in enumerate(tickers_sub):
            exp_ret[t] = mu_bl[i] * 252  # annualisé

        cov_full = np.eye(n) * 0.04
        t_idx = {t: i for i, t in enumerate(tickers)}
        for i, ti in enumerate(tickers_sub):
            for j, tj in enumerate(tickers_sub):
                cov_full[t_idx[ti], t_idx[tj]] = sigma_bl[i, j]

        logger.info(f"Optimisation BL terminée — {(weights>0.001).sum()} titres actifs")
        return {
            "weights": w_series,
            "expected_returns": exp_ret,
            "cov": cov_full,
            "tickers": tickers,
            "regime": ml_signals.get("regime", "unknown"),
        }

    def _qp_optimize(self, mu: np.ndarray, sigma: np.ndarray,
                     tickers: List[str],
                     sector_map: Optional[Dict] = None) -> np.ndarray:
        """Optimisation quadratique : max μ'w - λ/2 w'Σw sous contraintes."""
        try:
            import cvxpy as cp

            n = len(tickers)
            w = cp.Variable(n)

            ret = mu @ w
            risk = cp.quad_form(w, sigma)
            obj = cp.Maximize(ret - self.risk_aversion / 2 * risk)

            constraints = [
                cp.sum(w) == 1,
                w >= 0,                                      # long-only
                w <= self.port["max_weight_per_stock"],      # 10% max
            ]

            # Contrainte sectorielle
            if sector_map:
                sectors = {}
                for t, s in sector_map.items():
                    if t in tickers:
                        sectors.setdefault(s, []).append(tickers.index(t))
                for s_name, s_idx in sectors.items():
                    constraints.append(
                        cp.sum(w[s_idx]) <= self.port["max_weight_per_sector"]
                    )

            # Nb minimum de titres actifs (relaxed via soft constraint)
            min_stocks = min(self.port["min_stocks"], n)
            if min_stocks > 1:
                constraints.append(cp.sum(w >= 0.01 / n) >= 0)  # soft

            prob = cp.Problem(obj, constraints)
            prob.solve(solver=cp.ECOS, warm_start=True)

            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                result = np.maximum(w.value, 0)
                return result / result.sum()
            else:
                logger.warning(f"CVXPY status : {prob.status} — fallback poids égaux")
                return np.ones(n) / n

        except ImportError:
            logger.warning("CVXPY non disponible — max Sharpe analytique")
            return self._analytic_weights(mu, sigma, len(tickers))

    @staticmethod
    def _analytic_weights(mu, sigma, n):
        """Solution analytique max Sharpe (sans contraintes) comme fallback."""
        try:
            sigma_inv = np.linalg.pinv(sigma)
            w = sigma_inv @ mu
            w = np.maximum(w, 0)
            return w / w.sum() if w.sum() > 0 else np.ones(n) / n
        except Exception:
            return np.ones(n) / n


if __name__ == "__main__":
    cfg = load_config()
    print("Optimizer module chargé ✅")
    print(f"Méthode covariance : {cfg['covariance']['method']}")
    print(f"Risk aversion λ   : {cfg['black_litterman']['risk_aversion']}")
