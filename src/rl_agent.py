"""
Module 7 : RL Agent — PPO Rebalancing (Stable-Baselines3)
==========================================================
Agent de Reinforcement Learning qui apprend à décider QUAND et COMBIEN
rebalancer en maximisant le Sharpe net après coûts.

State space  : dérive des poids, régime HMM, jours depuis dernier rebalancing,
               coûts estimés, PRU fiscal, VIX z-score
Action space : niveau de rebalancing [0=rien, 1=partiel, 2=complet]
Reward       : rendement net - coûts - pénalité dérive - pénalité concentration

100% gratuit — Stable-Baselines3 + Gymnasium
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class PortfolioEnv:
    """
    Environnement Gymnasium pour l'entraînement de l'agent RL.
    Compatible avec Stable-Baselines3.
    """

    def __init__(self, returns: pd.DataFrame, macro: pd.DataFrame,
                 config: dict, tickers: List[str],
                 regime_series: Optional[Dict] = None):
        try:
            import gymnasium as gym
            self._gym = gym
        except ImportError:
            raise ImportError("pip install gymnasium stable-baselines3")

        self.returns = returns.fillna(0.0)
        self.macro = macro
        self.cfg = config
        self.tickers = tickers
        self.n = len(tickers)
        self.regime_series = regime_series or {}

        # Paramètres coûts
        broker_name = config["broker"]["name"]
        b = config["broker"]["brokers"][broker_name]
        self.eu_fee = b.get("eu_fixed_fee", 2.0)
        self.us_fee = b.get("us_fixed_fee", 50.0)
        self.flat_tax = config["taxation"]["flat_tax_pct"]
        self.capital = config["portfolio"]["initial_capital"]

        # Spaces Gymnasium
        import gymnasium as gym
        from gymnasium import spaces

        # State : [poids_actuels (n), drift (n), régime (3), jours_depuis_reb,
        #          vix_z, yc_z, coût_estimé_normalisé]
        obs_dim = self.n * 2 + 3 + 1 + 2 + 1
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32)

        # Action : 0 = rien, 1 = rebalancing partiel (50%), 2 = rebalancing complet
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None):
        """Remet l'environnement à son état initial."""
        # Démarre à une date aléatoire dans les premières 3 années
        min_idx = 252 * 3
        max_idx = max(len(self.returns) - 252, min_idx + 1)
        self.current_idx = np.random.randint(min_idx, max_idx)
        self.start_idx = self.current_idx

        # Portefeuille équipondéré initial
        self.weights = np.ones(self.n) / self.n
        self.portfolio_value = self.capital
        self.days_since_rebalance = 0
        self.episode_returns = []
        self.episode_costs = []

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Construit le vecteur d'état."""
        date = self.returns.index[self.current_idx]

        # Poids actuels (après drift du marché)
        ret_today = self.returns.iloc[self.current_idx].reindex(self.tickers).fillna(0.0).values
        drift_weights = self.weights * (1 + ret_today)
        if drift_weights.sum() > 0:
            drift_weights /= drift_weights.sum()

        # Dérive par rapport aux poids cibles (équipondérés comme baseline)
        target = np.ones(self.n) / self.n
        drift = drift_weights - target

        # Régime
        regime_p = self.regime_series.get(date, np.array([1.0, 0.0, 0.0]))
        if len(regime_p) < 3:
            regime_p = np.array([1.0, 0.0, 0.0])

        # Macro
        vix_z = 0.0
        yc_z = 0.0
        if len(self.macro) > 0:
            macro_row = self.macro.asof(date) if hasattr(self.macro, 'asof') else self.macro.iloc[-1]
            vix_z = float(macro_row.get("vix_z", 0.0) or 0.0)
            yc_z = float(macro_row.get("yield_curve_z", 0.0) or 0.0)

        # Coût estimé rebalancing (normalisé par valeur portefeuille)
        est_cost = (sum(1 for t in self.tickers if "." in t) * self.eu_fee +
                    sum(1 for t in self.tickers if "." not in t) * self.us_fee)
        cost_normalized = est_cost / max(self.portfolio_value, 1)

        obs = np.concatenate([
            drift_weights.astype(np.float32),
            drift.astype(np.float32),
            regime_p.astype(np.float32),
            [self.days_since_rebalance / 365.0],
            [np.clip(vix_z, -5, 5)],
            [np.clip(yc_z, -5, 5)],
            [np.clip(cost_normalized * 100, 0, 5)],
        ]).astype(np.float32)

        return np.clip(obs, -5.0, 5.0)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Exécute une action de rebalancing.

        Actions
        -------
        0 : Ne rien faire
        1 : Rebalancing partiel (ramène à 50% entre actuel et target)
        2 : Rebalancing complet (retour aux poids cibles)
        """
        date = self.returns.index[self.current_idx]
        ret_today = self.returns.iloc[self.current_idx].reindex(self.tickers).fillna(0.0).values

        # Drift du marché
        self.weights = self.weights * (1 + ret_today)
        if self.weights.sum() > 0:
            self.weights /= self.weights.sum()
        self.portfolio_value *= (1 + np.dot(self.weights, ret_today))

        # Calcul des coûts de rebalancing selon l'action
        target = np.ones(self.n) / self.n
        rebalance_cost = 0.0

        if action == 1:   # partiel
            blend = 0.5
            new_weights = blend * target + (1 - blend) * self.weights
            drift = np.abs(new_weights - self.weights)
            rebalance_cost = self._estimate_cost(drift)
            self.weights = new_weights / new_weights.sum()
            self.days_since_rebalance = 0

        elif action == 2:   # complet
            drift = np.abs(target - self.weights)
            rebalance_cost = self._estimate_cost(drift)
            self.weights = target.copy()
            self.days_since_rebalance = 0

        # Pénalité concentration
        herfindahl = np.sum(self.weights ** 2)   # 1/n = parfait, 1 = concentré
        concentration_penalty = max(herfindahl - 2/self.n, 0) * 0.5

        # Reward : rendement net - coûts - pénalités
        port_return = np.dot(self.weights, ret_today)
        cost_pct = rebalance_cost / max(self.portfolio_value, 1)
        reward = float(port_return - cost_pct - concentration_penalty)

        self.episode_returns.append(port_return)
        self.episode_costs.append(cost_pct)

        self.current_idx += 1
        self.days_since_rebalance += 1

        done = self.current_idx >= len(self.returns) - 1
        truncated = False

        # Bonus de fin d'épisode : Sharpe ratio net
        if done and len(self.episode_returns) > 20:
            r = np.array(self.episode_returns)
            sharpe_bonus = float(r.mean() / (r.std() + 1e-6) * np.sqrt(252) / 100)
            reward += sharpe_bonus

        obs = self._get_obs() if not done else np.zeros_like(self._get_obs())
        return obs, reward, done, truncated, {}

    def _estimate_cost(self, drift: np.ndarray) -> float:
        """Estime le coût de rebalancement en euros."""
        total = 0.0
        for i, ticker in enumerate(self.tickers):
            amount = drift[i] * self.portfolio_value
            if amount < 10:
                continue
            # EU = ticker avec "." (ex: ASML.AS), US = sans point
            if "." in ticker:
                total += max(self.eu_fee, amount * 0.001)
            else:
                total += max(self.us_fee, amount * 0.0001)
        return total


class RLRebalancingAgent:
    """
    Agent PPO (Proximal Policy Optimization) pour le rebalancing intelligent.
    Entraîné en simulation, déployé en backtest et en live.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.model = None
        self.model_path = config["ai"]["rl_agent"]["model_path"]
        self.total_timesteps = config["ai"]["rl_agent"]["total_timesteps"]

    def train(self, returns: pd.DataFrame, macro: pd.DataFrame,
              tickers: List[str], regime_series: Optional[Dict] = None):
        """Entraîne l'agent PPO sur les données historiques."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.env_checker import check_env
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            logger.warning("stable-baselines3 non installé — pip install stable-baselines3")
            return self

        logger.info(f"Entraînement PPO ({self.total_timesteps:,} timesteps)...")

        def make_env():
            env = PortfolioEnv(returns, macro, self.cfg, tickers, regime_series)
            return env

        vec_env = DummyVecEnv([make_env])

        rl_cfg = self.cfg["ai"]["rl_agent"]
        self.model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=rl_cfg["learning_rate"],
            n_steps=rl_cfg["n_steps"],
            batch_size=rl_cfg["batch_size"],
            gamma=rl_cfg["gamma"],
            verbose=1,
            tensorboard_log="data/cache/tb_logs/",
        )

        self.model.learn(total_timesteps=self.total_timesteps)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.model_path)
        logger.info(f"Agent PPO sauvegardé : {self.model_path} ✅")
        return self

    def load(self) -> bool:
        """Charge un agent pré-entraîné."""
        try:
            from stable_baselines3 import PPO
            if Path(self.model_path).exists():
                self.model = PPO.load(self.model_path)
                logger.info("Agent PPO chargé depuis cache ✅")
                return True
        except Exception as e:
            logger.warning(f"Chargement agent PPO échoué : {e}")
        return False

    def decide(self, obs: np.ndarray) -> Tuple[int, str]:
        """
        Décide de l'action de rebalancing pour un état donné.

        Retour : (action, description)
          0 = "HOLD"     — ne rien faire
          1 = "PARTIAL"  — rebalancing partiel
          2 = "FULL"     — rebalancing complet
        """
        if self.model is None:
            return 0, "HOLD (agent non entraîné)"

        action, _ = self.model.predict(obs, deterministic=True)
        labels = {0: "HOLD", 1: "PARTIAL_REBALANCE", 2: "FULL_REBALANCE"}
        return int(action), labels.get(int(action), "HOLD")

    def backtest_decisions(self, returns: pd.DataFrame, macro: pd.DataFrame,
                           tickers: List[str],
                           regime_series: Optional[Dict] = None) -> pd.DataFrame:
        """
        Rejoue l'agent sur les données historiques et retourne les décisions.
        Utile pour évaluer et visualiser le comportement de l'agent.
        """
        env = PortfolioEnv(returns, macro, self.cfg, tickers, regime_series)
        obs, _ = env.reset()

        records = []
        done = False

        while not done and env.current_idx < len(returns):
            action, label = self.decide(obs)
            obs, reward, done, _, _ = env.step(action)

            if env.current_idx < len(returns):
                date = returns.index[env.current_idx - 1]
                records.append({
                    "date": date,
                    "action": label,
                    "action_code": action,
                    "reward": reward,
                    "portfolio_value": env.portfolio_value,
                    "days_since_rebalance": env.days_since_rebalance,
                })

        df = pd.DataFrame(records)
        if not df.empty:
            n_rebalances = (df["action_code"] > 0).sum()
            logger.info(f"RL Agent : {n_rebalances} rebalancings sur {len(df)} jours "
                        f"({n_rebalances/len(df)*100:.1f}%)")
        return df


if __name__ == "__main__":
    from data_pipeline import DataPipeline
    cfg = load_config()
    tickers = cfg["universe"]["eu_tickers"] + cfg["universe"]["us_tickers"]
    dp = DataPipeline(cfg)
    data = dp.prepare(tickers, use_cache=True)

    agent = RLRebalancingAgent(cfg)
    if not agent.load():
        logger.info("Démarrage entraînement PPO...")
        agent.train(data["returns"], data["macro"], tickers)

    decisions = agent.backtest_decisions(data["returns"], data["macro"], tickers)
    if not decisions.empty:
        print(decisions["action"].value_counts())
        print(f"Valeur finale simulée : {decisions['portfolio_value'].iloc[-1]:,.0f}€")
