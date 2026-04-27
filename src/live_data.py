"""
Module 10 : Live Data & Auto-Update
=====================================
Mise à jour automatique des données de marché :
  - Prix journaliers (incrémental — ne re-télécharge que le delta)
  - Filtres Shariah (annuel, vérifie si une mise à jour est due)
  - Données macro FRED (hebdomadaire)
  - Signaux FinBERT sur flux RSS gratuits (NewsAPI / Yahoo RSS)
  - Re-entraînement des modèles ML si données > seuil
  - Compatible GitHub Actions (cron job automatique)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import yaml
import logging
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from io import StringIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DataUpdater:
    """
    Gestionnaire de mises à jour incrémentales des données.
    Conçu pour tourner en cron job (GitHub Actions, local scheduler).
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.cache = Path(config["data"]["cache_path"])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.manifest_file = self.cache / "update_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        """Charge le manifeste des dernières mises à jour."""
        if self.manifest_file.exists():
            with open(self.manifest_file) as f:
                return json.load(f)
        return {
            "last_price_update": None,
            "last_macro_update": None,
            "last_shariah_update": None,
            "last_model_retrain": None,
            "last_news_update": None,
            "data_hash": None,
        }

    def _save_manifest(self):
        with open(self.manifest_file, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    # ── Prix Incrémentaux ──────────────────────────────────────────────────

    def update_prices(self, tickers: List[str]) -> Tuple[bool, int]:
        """
        Mise à jour incrémentale des prix.
        Ne télécharge que les jours manquants depuis la dernière MAJ.

        Retour : (mis_à_jour, nb_nouveaux_jours)
        """
        prices_file = self.cache / "prices.parquet"
        div_file = self.cache / "dividends.parquet"
        vol_file = self.cache / "volumes.parquet"

        today = pd.Timestamp.today().normalize()

        # Détermine la date de début du delta
        if prices_file.exists():
            existing = pd.read_parquet(prices_file)
            last_date = existing.index[-1]
            delta_start = last_date + pd.Timedelta(days=1)
            if delta_start >= today:
                logger.info(f"Prix déjà à jour (dernière date : {last_date.date()})")
                return False, 0
        else:
            existing = pd.DataFrame()
            delta_start = pd.Timestamp(self.cfg["data"]["start_date"])

        logger.info(f"Mise à jour prix : {delta_start.date()} → {today.date()}")

        new_prices, new_divs, new_vols = {}, {}, {}
        for ticker in tickers:
            try:
                h = yf.Ticker(ticker).history(
                    start=delta_start, end=today + pd.Timedelta(days=1),
                    auto_adjust=True, actions=True, progress=False)
                if not h.empty:
                    new_prices[ticker] = h["Close"]
                    new_divs[ticker] = h["Dividends"]
                    new_vols[ticker] = h["Volume"]
            except Exception as e:
                logger.debug(f"  {ticker}: {e}")

        if not new_prices:
            logger.info("Aucune nouvelle donnée de prix")
            return False, 0

        delta_df = pd.DataFrame(new_prices)
        n_days = len(delta_df)

        if not existing.empty:
            combined = pd.concat([existing, delta_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = delta_df

        combined.to_parquet(prices_file)

        # Dividendes et volumes
        if div_file.exists() and new_divs:
            ex_div = pd.read_parquet(div_file)
            delta_div = pd.DataFrame(new_divs).fillna(0)
            pd.concat([ex_div, delta_div]).sort_index().drop_duplicates().to_parquet(div_file)

        if vol_file.exists() and new_vols:
            ex_vol = pd.read_parquet(vol_file)
            delta_vol = pd.DataFrame(new_vols).fillna(0)
            pd.concat([ex_vol, delta_vol]).sort_index().drop_duplicates().to_parquet(vol_file)

        self.manifest["last_price_update"] = str(today)
        self._save_manifest()
        logger.info(f"✅ Prix mis à jour : +{n_days} jours")
        return True, n_days

    # ── Macro Incrémentale ─────────────────────────────────────────────────

    def update_macro(self, force: bool = False) -> bool:
        """Mise à jour hebdomadaire des données macro FRED."""
        last = self.manifest.get("last_macro_update")
        if last and not force:
            last_dt = pd.Timestamp(last)
            if (pd.Timestamp.today() - last_dt).days < 7:
                logger.info("Macro déjà à jour (< 7 jours)")
                return False

        logger.info("Mise à jour données macro FRED...")
        macro_file = self.cache / "macro.parquet"
        series_map = self.cfg["data"]["macro_series"]

        existing = pd.read_parquet(macro_file) if macro_file.exists() else pd.DataFrame()
        last_date = existing.index[-1] if not existing.empty else pd.Timestamp("2000-01-01")
        delta_start = last_date - pd.Timedelta(days=30)  # chevauchement 30j pour continuité

        new_data = {}
        for name, sid in series_map.items():
            try:
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
                s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                new_data[name] = s[str(delta_start.date()):]
            except Exception as e:
                logger.debug(f"FRED {sid}: {e}")

        if not new_data:
            return False

        delta_macro = pd.DataFrame(new_data).ffill()

        if not existing.empty:
            combined = pd.concat([existing, delta_macro]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = delta_macro

        if "us_yield_10y" in combined and "us_yield_2y" in combined:
            combined["yield_curve"] = combined["us_yield_10y"] - combined["us_yield_2y"]

        # Recalcul z-scores sur toute la série
        for col in [c for c in combined.columns if not c.endswith("_z")]:
            rm = combined[col].rolling(252).mean()
            rs = combined[col].rolling(252).std()
            combined[f"{col}_z"] = (combined[col] - rm) / rs.replace(0, np.nan)

        combined.to_parquet(macro_file)
        self.manifest["last_macro_update"] = str(pd.Timestamp.today())
        self._save_manifest()
        logger.info("✅ Macro mise à jour")
        return True

    # ── Filtre Shariah Annuel ─────────────────────────────────────────────

    def update_shariah_filter(self, tickers: List[str], force: bool = False) -> bool:
        """Mise à jour annuelle du filtre Shariah."""
        last = self.manifest.get("last_shariah_update")
        if last and not force:
            last_dt = pd.Timestamp(last)
            if (pd.Timestamp.today() - last_dt).days < 365:
                logger.info(f"Filtre Shariah déjà à jour (dernière MAJ : {last_dt.date()})")
                return False

        logger.info("Mise à jour filtre Shariah (annuelle)...")
        from universe import ShariahFilter
        sf = ShariahFilter(self.cfg)
        df = sf.build_universe(tickers, use_cache=False, cache_path=str(self.cache))

        self.manifest["last_shariah_update"] = str(pd.Timestamp.today())
        self._save_manifest()

        n = df["compliant"].sum()
        logger.info(f"✅ Filtre Shariah mis à jour : {n}/{len(df)} conformes")
        return True

    # ── News & Sentiment ──────────────────────────────────────────────────

    def update_news_sentiment(self, tickers: List[str]) -> pd.DataFrame:
        """
        Récupère les dernières news via Yahoo Finance RSS (gratuit).
        Alimente FinBERT pour les signaux de sentiment live.
        """
        last = self.manifest.get("last_news_update")
        if last:
            last_dt = pd.Timestamp(last)
            if (pd.Timestamp.today() - last_dt).total_seconds() < 3600:
                news_file = self.cache / "news_sentiment.parquet"
                if news_file.exists():
                    return pd.read_parquet(news_file)

        logger.info("Récupération news Yahoo Finance RSS...")
        news_records = []

        for ticker in tickers[:10]:  # limite à 10 tickers pour la vitesse
            try:
                url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
                r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 200:
                    # Parse XML simple
                    from xml.etree import ElementTree as ET
                    root = ET.fromstring(r.content)
                    for item in root.findall(".//item")[:5]:
                        title = item.findtext("title", "")
                        pub_date = item.findtext("pubDate", "")
                        if title:
                            news_records.append({
                                "ticker": ticker,
                                "title": title,
                                "pub_date": pub_date,
                                "date": pd.Timestamp.today(),
                            })
            except Exception as e:
                logger.debug(f"News {ticker}: {e}")

        if not news_records:
            return pd.DataFrame()

        news_df = pd.DataFrame(news_records)

        # Scoring FinBERT si disponible
        try:
            from ml_engine import SentimentAnalyzer
            sa = SentimentAnalyzer(self.cfg)
            sa.load_model()
            if sa.pipeline:
                news_df["sentiment_score"] = news_df["title"].apply(sa.score_text)
                ticker_sentiment = news_df.groupby("ticker")["sentiment_score"].mean()

                sent_file = self.cache / "news_sentiment.parquet"
                ticker_sentiment.to_frame("sentiment").to_parquet(sent_file)
                logger.info(f"✅ Sentiment calculé pour {len(ticker_sentiment)} tickers")
        except Exception as e:
            logger.debug(f"FinBERT scoring: {e}")

        self.manifest["last_news_update"] = str(pd.Timestamp.today())
        self._save_manifest()
        return news_df

    # ── Re-entraînement Conditionnel ──────────────────────────────────────

    def should_retrain(self, returns: pd.DataFrame,
                       threshold_days: int = 90) -> Tuple[bool, str]:
        """
        Décide si les modèles ML doivent être ré-entraînés.
        Critères : délai > 90 jours OU changement de régime de marché.
        """
        last = self.manifest.get("last_model_retrain")
        if last is None:
            return True, "jamais entraîné"

        last_dt = pd.Timestamp(last)
        days_since = (pd.Timestamp.today() - last_dt).days
        if days_since >= threshold_days:
            return True, f"délai : {days_since} jours"

        # Détection de changement de régime via volatilité
        recent_vol = returns.mean(axis=1).tail(21).std() * np.sqrt(252)
        hist_vol = returns.mean(axis=1).tail(252).std() * np.sqrt(252)
        if hist_vol > 0 and recent_vol / hist_vol > 1.5:
            return True, f"changement de régime (vol x{recent_vol/hist_vol:.1f})"

        return False, f"OK ({days_since} jours depuis dernier entraînement)"

    def mark_retrained(self):
        self.manifest["last_model_retrain"] = str(pd.Timestamp.today())
        self._save_manifest()

    # ── Update Complet ─────────────────────────────────────────────────────

    def run_full_update(self, tickers: List[str],
                        ml_engine=None, force: bool = False) -> Dict:
        """
        Lance toutes les mises à jour dans le bon ordre.
        Point d'entrée pour le cron job GitHub Actions.
        """
        logger.info("══ MISE À JOUR COMPLÈTE DES DONNÉES ══")
        report = {}

        # 1. Prix (quotidien)
        updated_prices, n_days = self.update_prices(tickers)
        report["prices"] = {"updated": updated_prices, "new_days": n_days}

        # 2. Macro (hebdomadaire)
        updated_macro = self.update_macro(force=force)
        report["macro"] = {"updated": updated_macro}

        # 3. Filtre Shariah (annuel)
        updated_shariah = self.update_shariah_filter(tickers, force=force)
        report["shariah"] = {"updated": updated_shariah}

        # 4. News (horaire en live)
        news_df = self.update_news_sentiment(tickers)
        report["news"] = {"n_articles": len(news_df)}

        # 5. Re-entraînement conditionnel
        if ml_engine is not None and (updated_prices or force):
            try:
                prices_file = self.cache / "prices.parquet"
                if prices_file.exists():
                    prices = pd.read_parquet(prices_file)
                    returns_proxy = prices.pct_change().dropna(how="all")
                    retrain, reason = self.should_retrain(returns_proxy)
                    report["retrain"] = {"needed": retrain, "reason": reason}
                    if retrain:
                        logger.info(f"Re-entraînement ML : {reason}")
                        # Note : le re-entraînement complet est lancé séparément
                        # pour ne pas bloquer la MAJ des données
            except Exception as e:
                logger.warning(f"Check retrain: {e}")
                report["retrain"] = {"needed": False, "reason": str(e)}

        logger.info("══ MISE À JOUR TERMINÉE ══")
        logger.info(json.dumps(report, indent=2, default=str))
        return report

    def get_status(self) -> Dict:
        """Résumé de l'état des données."""
        prices_file = self.cache / "prices.parquet"
        status = {
            "last_price_update": self.manifest.get("last_price_update", "jamais"),
            "last_macro_update": self.manifest.get("last_macro_update", "jamais"),
            "last_shariah_update": self.manifest.get("last_shariah_update", "jamais"),
            "last_model_retrain": self.manifest.get("last_model_retrain", "jamais"),
        }
        if prices_file.exists():
            p = pd.read_parquet(prices_file)
            status["price_data_from"] = str(p.index[0].date())
            status["price_data_to"] = str(p.index[-1].date())
            status["n_tickers"] = p.shape[1]
            status["n_days"] = p.shape[0]
        return status


if __name__ == "__main__":
    cfg = load_config()
    tickers = cfg["universe"]["eu_tickers"] + cfg["universe"]["us_tickers"]
    updater = DataUpdater(cfg)
    print("État actuel des données :")
    print(json.dumps(updater.get_status(), indent=2))
    report = updater.run_full_update(tickers)
    print("\nRapport de mise à jour :")
    print(json.dumps(report, indent=2, default=str))
