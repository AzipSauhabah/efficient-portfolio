"""
Module 3 : ML Engine — XGBoost + HMM + LSTM + FinBERT
=======================================================
4 modèles IA complémentaires qui génèrent des signaux alpha :

  1. HMM          → Détection régime marché (bull/bear/crisis)
  2. XGBoost      → Prédiction rendements 1M (interprétable SHAP)
  3. LSTM         → Prédiction volatilité (séries temporelles)
  4. FinBERT      → Sentiment NLP sur titres financiers (HuggingFace)

Les signaux sont combinés en scores [-1, +1] par ticker,
alimentant ensuite Black-Litterman comme views actives.
"""

import numpy as np
import pandas as pd
import yaml
import logging
import warnings
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════
#  1. HMM — Régime de Marché
# ══════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Hidden Markov Model à 3 états : Bull / Bear / Crisis.
    Features : rendement portfolio, volatilité réalisée, VIX z-score.
    """

    def __init__(self, config: dict):
        self.n = config["ai"]["hmm"]["n_regimes"]
        self.cov_type = config["ai"]["hmm"]["covariance_type"]
        self.n_iter = config["ai"]["hmm"]["n_iter"]
        self.model = None
        self.regime_labels = {}   # {0: "bull", 1: "bear", 2: "crisis"}

    def _build_features(self, returns: pd.DataFrame,
                        macro: pd.DataFrame) -> pd.DataFrame:
        eq = returns.mean(axis=1)
        feats = pd.DataFrame(index=eq.index)
        feats["ret_1d"] = eq
        feats["vol_21d"] = eq.rolling(21).std() * np.sqrt(252)
        feats["vol_63d"] = eq.rolling(63).std() * np.sqrt(252)
        feats["ret_21d"] = eq.rolling(21).sum()
        feats["ret_63d"] = eq.rolling(63).sum()
        if "vix_z" in macro.columns:
            feats["vix_z"] = macro["vix_z"].reindex(eq.index).ffill()
        if "yield_curve_z" in macro.columns:
            feats["yc_z"] = macro["yield_curve_z"].reindex(eq.index).ffill()
        return feats.dropna()

    def fit(self, returns: pd.DataFrame, macro: pd.DataFrame):
        from hmmlearn.hmm import GaussianHMM
        feats = self._build_features(returns, macro)
        self.model = GaussianHMM(
            n_components=self.n,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            random_state=42,
        )
        self.model.fit(feats.values)
        self._label_regimes(feats)
        logger.info(f"HMM entraîné : {self.n} régimes détectés")
        return self

    def _label_regimes(self, feats: pd.DataFrame):
        """Associe chaque état HMM à bull/bear/crisis selon la moyenne des rendements."""
        states = self.model.predict(feats.values)
        means = {}
        for s in range(self.n):
            mask = states == s
            means[s] = feats["ret_1d"].values[mask].mean()
        sorted_states = sorted(means, key=means.get, reverse=True)
        labels = ["bull", "bear", "crisis"]
        for i, s in enumerate(sorted_states):
            self.regime_labels[s] = labels[min(i, 2)]
        logger.info(f"Régimes : {self.regime_labels}")

    def predict(self, returns: pd.DataFrame,
                macro: pd.DataFrame) -> Tuple[str, np.ndarray]:
        """Retourne (régime_actuel, probas[bull, bear, crisis])."""
        feats = self._build_features(returns, macro)
        if len(feats) == 0 or self.model is None:
            return "bull", np.array([1.0, 0.0, 0.0])
        state = self.model.predict(feats.values[-1:].reshape(1, -1))[0]
        probas_raw = self.model.predict_proba(feats.values[-1:].reshape(1, -1))[0]
        # Réordonne en [bull, bear, crisis]
        ordered = np.zeros(3)
        label_to_idx = {"bull": 0, "bear": 1, "crisis": 2}
        for s, label in self.regime_labels.items():
            ordered[label_to_idx[label]] = probas_raw[s]
        regime = self.regime_labels.get(state, "bull")
        return regime, ordered

    def save(self, path: str = "data/cache/hmm_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "data/cache/hmm_model.pkl") -> "RegimeDetector":
        return joblib.load(path)


# ══════════════════════════════════════════════════════════════
#  2. XGBoost — Prédiction Rendements + SHAP
# ══════════════════════════════════════════════════════════════

class AlphaPredictor:
    """
    XGBoost qui prédit le rendement relatif à 21 jours par ticker.
    Features : momentum multi-horizon, quality proxy, macro, régime HMM.
    Interprétable via SHAP — montre POURQUOI chaque ticker est surpondéré.
    """

    def __init__(self, config: dict):
        self.cfg = config["ai"]["xgboost"]
        self.lookbacks = self.cfg["lookback_features_days"]
        self.models = {}       # {ticker: XGBRegressor}
        self.shap_values = {}  # pour visualisation

    def _make_features(self, returns: pd.DataFrame, macro: pd.DataFrame,
                       regime_probas: np.ndarray, ticker: str,
                       t: pd.Timestamp) -> Optional[np.ndarray]:
        """Construit le vecteur de features pour un ticker à une date."""
        ret = returns[ticker].loc[:t]
        mkt = returns.mean(axis=1).loc[:t]

        if len(ret) < max(self.lookbacks) + 22:
            return None

        feats = []
        # Momentum multi-horizon
        for lb in self.lookbacks:
            r = ret.iloc[-lb:].sum()
            feats.append(r)
            # Relative momentum vs marché
            m = mkt.iloc[-lb:].sum()
            feats.append(r - m)

        # Volatilité
        feats.append(ret.iloc[-21:].std() * np.sqrt(252))
        feats.append(ret.iloc[-63:].std() * np.sqrt(252))

        # Skewness / Kurtosis (tail risk)
        feats.append(float(ret.iloc[-63:].skew()))
        feats.append(float(ret.iloc[-63:].kurt()))

        # Max drawdown 63j
        cum = (1 + ret.iloc[-63:]).cumprod()
        roll_max = cum.cummax()
        dd = ((cum - roll_max) / roll_max).min()
        feats.append(float(dd))

        # Macro features
        if len(macro) > 0:
            macro_row = macro.asof(t) if hasattr(macro, 'asof') else macro.iloc[-1]
            for col in ["vix_z", "yield_curve_z", "eur_usd_z"]:
                val = macro_row.get(col, 0.0)
                feats.append(float(val) if pd.notna(val) else 0.0)

        # Régime HMM (probas bull/bear/crisis)
        feats.extend(regime_probas.tolist())

        return np.array(feats, dtype=np.float32)

    def _build_dataset(self, returns: pd.DataFrame, macro: pd.DataFrame,
                       regime_series: pd.Series, ticker: str,
                       horizon: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        """Construit X, y pour un ticker (walk-forward, pas de lookahead)."""
        X_list, y_list = [], []
        dates = returns.index[max(self.lookbacks) + 22: -horizon]

        for t in dates[::5]:   # sous-échantillonnage x5 pour vitesse
            regime_p = regime_series.get(t, np.array([1.0, 0.0, 0.0]))
            x = self._make_features(returns, macro, regime_p, ticker, t)
            if x is None:
                continue
            # Cible : rendement futur sur horizon jours
            future_idx = returns.index.searchsorted(t)
            if future_idx + horizon >= len(returns):
                continue
            future_ret = returns[ticker].iloc[future_idx:future_idx+horizon].sum()
            if pd.isna(future_ret):
                continue
            X_list.append(x)
            y_list.append(future_ret)

        if not X_list:
            return np.array([]), np.array([])
        return np.array(X_list), np.array(y_list)

    def fit(self, returns: pd.DataFrame, macro: pd.DataFrame,
            regime_series: Dict, tickers: List[str]):
        """Entraîne un XGBoost par ticker (ou un modèle global poolé)."""
        from xgboost import XGBRegressor

        logger.info("Entraînement XGBoost (modèle poolé tous tickers)...")
        X_all, y_all = [], []

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            X, y = self._build_dataset(returns, macro, regime_series, ticker)
            if len(X) > 10:
                # Ajout d'un feature "ticker_id" pour modèle poolé
                ticker_id = np.full((len(X), 1), hash(ticker) % 1000, dtype=np.float32)
                X_all.append(np.hstack([X, ticker_id]))
                y_all.append(y)

        if not X_all:
            logger.warning("XGBoost : données insuffisantes")
            return self

        X_pool = np.vstack(X_all)
        y_pool = np.concatenate(y_all)

        # Winsorize y (limite les outliers extrêmes)
        p1, p99 = np.percentile(y_pool, [1, 99])
        y_pool = np.clip(y_pool, p1, p99)

        model = XGBRegressor(
            n_estimators=self.cfg["n_estimators"],
            max_depth=self.cfg["max_depth"],
            learning_rate=self.cfg["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_pool, y_pool, eval_set=[(X_pool, y_pool)], verbose=False)
        self.models["global"] = model
        logger.info(f"XGBoost entraîné sur {len(y_pool)} samples")
        return self

    def predict_scores(self, returns: pd.DataFrame, macro: pd.DataFrame,
                       regime_probas: np.ndarray, tickers: List[str],
                       as_of: pd.Timestamp) -> pd.Series:
        """Retourne un score normalisé [-1, +1] par ticker."""
        if "global" not in self.models:
            return pd.Series(0.0, index=tickers)

        model = self.models["global"]
        preds = {}

        for ticker in tickers:
            if ticker not in returns.columns:
                continue
            x = self._make_features(returns, macro, regime_probas, ticker, as_of)
            if x is None:
                preds[ticker] = 0.0
                continue
            ticker_id = np.array([[hash(ticker) % 1000]], dtype=np.float32)
            x_full = np.hstack([x.reshape(1, -1), ticker_id])
            preds[ticker] = float(model.predict(x_full)[0])

        scores = pd.Series(preds)
        if scores.std() > 0:
            scores = (scores - scores.mean()) / scores.std()
        return scores.clip(-3, 3) / 3

    def explain(self, returns: pd.DataFrame, macro: pd.DataFrame,
                regime_probas: np.ndarray, tickers: List[str],
                as_of: pd.Timestamp) -> Optional[pd.DataFrame]:
        """SHAP values pour interprétabilité — montre les drivers du score."""
        try:
            import shap
            if "global" not in self.models:
                return None
            X_list = []
            valid_tickers = []
            for ticker in tickers:
                if ticker not in returns.columns:
                    continue
                x = self._make_features(returns, macro, regime_probas, ticker, as_of)
                if x is not None:
                    ticker_id = np.array([hash(ticker) % 1000], dtype=np.float32)
                    X_list.append(np.hstack([x, ticker_id]))
                    valid_tickers.append(ticker)
            if not X_list:
                return None
            X = np.array(X_list)
            explainer = shap.TreeExplainer(self.models["global"])
            shap_vals = explainer.shap_values(X)
            feature_names = (
                [f"mom_{lb}d" for lb in self.lookbacks] +
                [f"rel_mom_{lb}d" for lb in self.lookbacks] +
                ["vol_21d", "vol_63d", "skew", "kurt", "max_dd",
                 "vix_z", "yc_z", "fx_z",
                 "p_bull", "p_bear", "p_crisis", "ticker_id"]
            )
            df_shap = pd.DataFrame(shap_vals, index=valid_tickers,
                                   columns=feature_names[:shap_vals.shape[1]])
            return df_shap
        except ImportError:
            logger.warning("SHAP non disponible — pip install shap")
            return None

    def save(self, path: str = "data/cache/xgb_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "data/cache/xgb_model.pkl") -> "AlphaPredictor":
        return joblib.load(path)


# ══════════════════════════════════════════════════════════════
#  3. LSTM — Prédiction Volatilité
# ══════════════════════════════════════════════════════════════

class VolatilityLSTM:
    """
    LSTM PyTorch qui prédit la volatilité réalisée à 21 jours.
    Utilisé pour ajuster la matrice de covariance (scaling diagonal).
    """

    def __init__(self, config: dict):
        self.cfg = config["ai"]["lstm"]
        self.lookback = self.cfg["lookback_days"]
        self.model = None
        self.scaler_x = None
        self.scaler_y = None

    def _build_sequences(self, vol_series: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(self.lookback, len(vol_series) - 21):
            X.append(vol_series[i-self.lookback:i])
            y.append(vol_series[i:i+21].mean())
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, returns: pd.DataFrame):
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("PyTorch non disponible — LSTM désactivé")
            return self

        logger.info("Entraînement LSTM volatilité...")

        # Volatilité réalisée rolling 21j du portfolio équipondéré
        port_ret = returns.mean(axis=1).dropna()
        vol = port_ret.rolling(21).std().dropna().values * np.sqrt(252)

        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()

        X, y = self._build_sequences(vol)
        if len(X) < 50:
            logger.warning("LSTM : données insuffisantes")
            return self

        X_s = self.scaler_x.fit_transform(X)
        y_s = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        X_t = torch.FloatTensor(X_s).unsqueeze(-1)  # (N, T, 1)
        y_t = torch.FloatTensor(y_s)

        class LSTMNet(nn.Module):
            def __init__(self, hidden, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, layers, dropout=drop,
                                    batch_first=True)
                self.fc = nn.Linear(hidden, 1)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze()

        net = LSTMNet(self.cfg["hidden_size"], self.cfg["num_layers"],
                      self.cfg["dropout"])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        net.train()
        bs = self.cfg["batch_size"]
        for epoch in range(self.cfg["epochs"]):
            for i in range(0, len(X_t), bs):
                xb, yb = X_t[i:i+bs], y_t[i:i+bs]
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            if (epoch + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.cfg['epochs']} — loss {loss.item():.6f}")

        self.model = net
        logger.info("LSTM entraîné ✅")
        return self

    def predict_vol(self, returns: pd.DataFrame) -> float:
        """Prédit la volatilité du portfolio pour les 21 prochains jours."""
        if self.model is None:
            port_ret = returns.mean(axis=1).dropna()
            return float(port_ret.tail(21).std() * np.sqrt(252))

        try:
            import torch
            port_ret = returns.mean(axis=1).dropna()
            vol = port_ret.rolling(21).std().dropna().values * np.sqrt(252)
            if len(vol) < self.lookback:
                return float(vol[-1]) if len(vol) > 0 else 0.15
            x = vol[-self.lookback:].reshape(1, -1)
            x_s = self.scaler_x.transform(x)
            x_t = torch.FloatTensor(x_s).unsqueeze(-1)
            self.model.eval()
            with torch.no_grad():
                pred_s = self.model(x_t).item()
            pred = float(self.scaler_y.inverse_transform([[pred_s]])[0, 0])
            return max(pred, 0.05)
        except Exception as e:
            logger.debug(f"LSTM predict: {e}")
            return 0.15

    def save(self, path: str = "data/cache/lstm_model.pkl"):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str = "data/cache/lstm_model.pkl") -> "VolatilityLSTM":
        return joblib.load(path)


# ══════════════════════════════════════════════════════════════
#  4. FinBERT — Sentiment NLP
# ══════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """
    FinBERT (HuggingFace, gratuit) pour analyse de sentiment financier.
    En backtest : utilise des headlines synthétiques basées sur les données
    de marché (pas de vraies news historiques sans API payante).
    En live : peut être connecté à des flux RSS gratuits.
    """

    def __init__(self, config: dict):
        self.cfg = config["ai"]["finbert"]
        self.pipeline = None

    def load_model(self):
        if self.pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Chargement FinBERT ({self.cfg['model_name']})...")
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.cfg["model_name"],
                device=-1,           # CPU (0 = GPU si disponible)
                top_k=None,
            )
            logger.info("FinBERT chargé ✅")
        except Exception as e:
            logger.warning(f"FinBERT non disponible : {e}")
            self.pipeline = None

    def score_text(self, text: str) -> float:
        """Retourne score de sentiment [-1, +1] pour un texte."""
        if self.pipeline is None:
            return 0.0
        try:
            results = self.pipeline(text[:512])[0]
            score = 0.0
            for r in results:
                if r["label"] == "positive":
                    score += r["score"]
                elif r["label"] == "negative":
                    score -= r["score"]
            return float(np.clip(score, -1, 1))
        except Exception:
            return 0.0

    def _generate_synthetic_headlines(self, ticker: str, returns: pd.DataFrame,
                                      macro: pd.DataFrame,
                                      as_of: pd.Timestamp) -> List[str]:
        """
        Génère des titres synthétiques basés sur les données de marché
        pour simuler le sentiment en backtest (sans API news payante).
        """
        if ticker not in returns.columns:
            return []

        ret_1m = returns[ticker].loc[:as_of].tail(21).sum()
        ret_3m = returns[ticker].loc[:as_of].tail(63).sum()
        vol = returns[ticker].loc[:as_of].tail(21).std() * np.sqrt(252)

        headlines = []

        if ret_1m > 0.05:
            headlines.append(f"{ticker} surges on strong quarterly results and revenue beat")
        elif ret_1m < -0.05:
            headlines.append(f"{ticker} falls amid market concerns and revenue miss")
        else:
            headlines.append(f"{ticker} trades flat as investors await earnings guidance")

        if ret_3m > 0.10:
            headlines.append(f"{ticker} continues upward momentum with positive analyst revisions")
        elif ret_3m < -0.10:
            headlines.append(f"{ticker} faces headwinds with multiple analyst downgrades")

        # Macro context
        vix_z = macro["vix_z"].asof(as_of) if "vix_z" in macro.columns else 0
        if pd.notna(vix_z) and vix_z > 2:
            headlines.append("Markets face heightened volatility amid geopolitical tensions")
        elif pd.notna(vix_z) and vix_z < -1:
            headlines.append("Risk appetite improves as volatility drops to multi-month lows")

        return headlines

    def compute_scores(self, tickers: List[str], returns: pd.DataFrame,
                       macro: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """Score de sentiment agrégé par ticker."""
        self.load_model()
        scores = {}

        for ticker in tickers:
            headlines = self._generate_synthetic_headlines(ticker, returns, macro, as_of)
            if not headlines:
                scores[ticker] = 0.0
                continue
            ticker_scores = [self.score_text(h) for h in headlines]
            scores[ticker] = float(np.mean(ticker_scores))

        s = pd.Series(scores)
        if s.std() > 0:
            s = (s - s.mean()) / s.std()
        return s.clip(-3, 3) / 3


# ══════════════════════════════════════════════════════════════
#  Orchestrateur ML
# ══════════════════════════════════════════════════════════════

class MLEngine:
    """
    Orchestre les 4 modèles IA et produit les signaux consolidés
    pour Black-Litterman.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.bl_weights = config["black_litterman"]["signal_weights"]
        self.active = config["black_litterman"]["signals"]

        self.regime_detector = RegimeDetector(config)
        self.alpha_predictor = AlphaPredictor(config)
        self.vol_lstm = VolatilityLSTM(config)
        self.sentiment = SentimentAnalyzer(config)

        self.regime_series: Dict = {}   # cache des régimes par date
        self._fitted = False

    def fit(self, returns: pd.DataFrame, macro: pd.DataFrame,
            tickers: List[str]):
        """Entraîne tous les modèles sur les données historiques."""
        logger.info("══ ENTRAÎNEMENT MODÈLES IA ══")

        # 1. HMM
        if self.active.get("crisis_regime", True):
            logger.info("1/4 HMM Regime Detector...")
            self.regime_detector.fit(returns, macro)

            # Pré-calcul des régimes pour chaque date (pour XGBoost)
            feats_all = self.regime_detector._build_features(returns, macro)
            if len(feats_all) > 0 and self.regime_detector.model is not None:
                states = self.regime_detector.model.predict(feats_all.values)
                probas = self.regime_detector.model.predict_proba(feats_all.values)
                for i, date in enumerate(feats_all.index):
                    self.regime_series[date] = probas[i]

        # 2. XGBoost
        if self.active.get("momentum", True) or self.active.get("quality", True):
            logger.info("2/4 XGBoost Alpha Predictor...")
            self.alpha_predictor.fit(returns, macro, self.regime_series, tickers)

        # 3. LSTM
        logger.info("3/4 LSTM Volatilité...")
        self.vol_lstm.fit(returns)

        # 4. FinBERT — chargé à la demande (lourd)
        logger.info("4/4 FinBERT (chargement à la demande)...")

        self._fitted = True
        logger.info("══ ENTRAÎNEMENT TERMINÉ ✅ ══")
        return self

    def generate_signals(self, returns: pd.DataFrame, macro: pd.DataFrame,
                         tickers: List[str],
                         as_of: Optional[pd.Timestamp] = None) -> Dict:
        """
        Génère tous les signaux IA pour une date donnée.

        Retour
        ------
        dict avec :
          - scores      : pd.Series, score agrégé [-1,+1] par ticker
          - regime      : str, régime actuel
          - regime_prob : np.ndarray [p_bull, p_bear, p_crisis]
          - vol_forecast: float, volatilité prévue
          - shap_df     : pd.DataFrame ou None
          - detail      : dict des scores par source
        """
        if as_of is None:
            as_of = returns.index[-1]

        ret_to = returns.loc[:as_of]
        mac_to = macro.loc[:as_of] if len(macro) > 0 else macro

        # Régime
        regime, regime_prob = self.regime_detector.predict(ret_to, mac_to)

        # Volatilité LSTM
        vol_forecast = self.vol_lstm.predict_vol(ret_to)

        # Scores individuels
        detail = {}

        if self.active.get("momentum", True):
            xgb_scores = self.alpha_predictor.predict_scores(
                ret_to, mac_to, regime_prob, tickers, as_of)
            detail["xgboost"] = xgb_scores
        else:
            detail["xgboost"] = pd.Series(0.0, index=tickers)

        if self.active.get("sentiment", True):
            sent_scores = self.sentiment.compute_scores(tickers, ret_to, mac_to, as_of)
            detail["sentiment"] = sent_scores
        else:
            detail["sentiment"] = pd.Series(0.0, index=tickers)

        # Score macro : signal global selon régime
        regime_multiplier = {"bull": 0.5, "bear": -0.3, "crisis": -0.8}
        macro_score = regime_multiplier.get(regime, 0.0)
        detail["macro"] = pd.Series(macro_score, index=tickers)

        # Agrégation pondérée
        w_xgb = self.bl_weights.get("momentum", 0.25) + self.bl_weights.get("quality", 0.30)
        w_sent = self.bl_weights.get("sentiment", 0.10)
        w_mac = self.bl_weights.get("macro", 0.20) + self.bl_weights.get("crisis_regime", 0.15)
        total_w = w_xgb + w_sent + w_mac

        scores = (
            detail["xgboost"] * w_xgb +
            detail["sentiment"] * w_sent +
            detail["macro"] * w_mac
        ) / total_w

        # SHAP pour interprétabilité
        shap_df = self.alpha_predictor.explain(
            ret_to, mac_to, regime_prob, tickers, as_of)

        return {
            "scores": scores,
            "regime": regime,
            "regime_prob": regime_prob,
            "vol_forecast": vol_forecast,
            "shap_df": shap_df,
            "detail": detail,
        }

    def save(self, cache_path: str = "data/cache/"):
        path = Path(cache_path)
        self.regime_detector.save(str(path / "hmm.pkl"))
        self.alpha_predictor.save(str(path / "xgb.pkl"))
        self.vol_lstm.save(str(path / "lstm.pkl"))
        logger.info("Modèles ML sauvegardés ✅")

    def load(self, cache_path: str = "data/cache/"):
        path = Path(cache_path)
        try:
            self.regime_detector = RegimeDetector.load(str(path / "hmm.pkl"))
            self.alpha_predictor = AlphaPredictor.load(str(path / "xgb.pkl"))
            self.vol_lstm = VolatilityLSTM.load(str(path / "lstm.pkl"))
            self._fitted = True
            logger.info("Modèles ML chargés depuis cache ✅")
        except Exception as e:
            logger.warning(f"Chargement modèles échoué : {e} — réentraînement nécessaire")


if __name__ == "__main__":
    from data_pipeline import DataPipeline
    cfg = load_config()
    tickers = cfg["universe"]["eu_tickers"] + cfg["universe"]["us_tickers"]
    dp = DataPipeline(cfg)
    data = dp.prepare(tickers, use_cache=True)
    engine = MLEngine(cfg)
    engine.fit(data["returns"], data["macro"], tickers)
    signals = engine.generate_signals(data["returns"], data["macro"], tickers)
    print(f"\nRégime : {signals['regime']} | Vol prévue : {signals['vol_forecast']:.1%}")
    print(signals["scores"].sort_values(ascending=False))
