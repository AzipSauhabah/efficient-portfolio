"""
Module 2 : Data Pipeline
=========================
- Prix ajustés splits + dividendes (yfinance)
- FX EUR/USD (yfinance fallback si FRED indisponible)
- Données macro : VIX, yield curve, M2, Brent (FRED gratuit)
- Nettoyage, détection anomalies, calcul rendements nets CTO France
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import yaml
import logging
from pathlib import Path
from io import StringIO
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DataPipeline:
    def __init__(self, config: dict):
        self.cfg = config
        self.start = config["data"]["start_date"]
        self.end = config["data"]["end_date"]
        self.cache = Path(config["data"]["cache_path"])
        self.cache.mkdir(parents=True, exist_ok=True)
        self.eu_tickers = config["universe"]["eu_tickers"]
        self.us_tickers = config["universe"]["us_tickers"]
        self.tax = config["taxation"]
        self.purif_rate = config["shariah"]["purification_rate"]

    # ── Prix & Dividendes ──────────────────────────────────────────────────

    def fetch_prices(self, tickers: List[str], use_cache: bool = True
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        pf = self.cache / "prices.parquet"
        df_ = self.cache / "dividends.parquet"
        vf = self.cache / "volumes.parquet"

        if use_cache and pf.exists():
            logger.info("📂 Prix chargés depuis cache")
            return pd.read_parquet(pf), pd.read_parquet(df_), pd.read_parquet(vf)

        prices, dividends, volumes = {}, {}, {}
        failed = []

        for ticker in tickers:
            try:
                h = yf.Ticker(ticker).history(
                    start=self.start, end=self.end,
                    auto_adjust=True, actions=True
                )
                if h.empty or len(h) < 200:
                    failed.append(ticker); continue
                prices[ticker] = h["Close"]
                dividends[ticker] = h["Dividends"]
                volumes[ticker] = h["Volume"]
            except Exception as e:
                logger.warning(f"{ticker}: {e}"); failed.append(ticker)

        if failed:
            logger.warning(f"Échecs téléchargement : {failed}")

        p = pd.DataFrame(prices)
        d = pd.DataFrame(dividends).fillna(0.0)
        v = pd.DataFrame(volumes).fillna(0.0)
        p = self._clean(p)

        p.to_parquet(pf); d.to_parquet(df_); v.to_parquet(vf)
        logger.info(f"✅ Prix : {p.shape[1]} tickers × {p.shape[0]} jours")
        return p, d, v

    def _clean(self, prices: pd.DataFrame) -> pd.DataFrame:
        # Supprime tickers > 30% manquants
        bad = prices.columns[prices.isna().mean() > 0.30]
        if len(bad): logger.warning(f"Supprimés (>30% NaN) : {list(bad)}")
        prices = prices.drop(columns=bad)
        prices = prices.ffill(limit=5).replace(0, np.nan)
        # Anomalies > 90% en 1 jour (splits mal ajustés)
        r = prices.pct_change().abs()
        prices = prices.where(~(r > 0.90).shift(-1, fill_value=False))
        return prices

    # ── FX ─────────────────────────────────────────────────────────────────

    def fetch_fx(self, use_cache: bool = True) -> pd.Series:
        ff = self.cache / "fx_eurusd.parquet"
        if use_cache and ff.exists():
            return pd.read_parquet(ff)["EUR_per_USD"]

        logger.info("Téléchargement EUR/USD...")
        try:
            fx = yf.download("EURUSD=X", start=self.start, end=self.end,
                             auto_adjust=True, progress=False)["Close"]
            fx = fx.squeeze()
            # EURUSD=X = USD par EUR → EUR_per_USD = 1/fx mais yf donne déjà USD/EUR
            eur_per_usd = 1 / fx
        except Exception:
            logger.warning("FX yfinance échoué — taux fixe 1.10")
            idx = pd.date_range(self.start, self.end, freq="B")
            eur_per_usd = pd.Series(1/1.10, index=idx)

        df_fx = pd.DataFrame({"EUR_per_USD": eur_per_usd})
        df_fx.to_parquet(ff)
        return eur_per_usd

    def convert_to_eur(self, prices: pd.DataFrame, fx: pd.Series) -> pd.DataFrame:
        p = prices.copy()
        fx_aligned = fx.reindex(p.index).ffill().bfill()
        for t in self.us_tickers:
            if t in p.columns:
                p[t] = p[t] * fx_aligned
        return p

    # ── Macro ──────────────────────────────────────────────────────────────

    def fetch_macro(self, use_cache: bool = True) -> pd.DataFrame:
        mf = self.cache / "macro.parquet"
        if use_cache and mf.exists():
            return pd.read_parquet(mf)

        logger.info("Téléchargement données macro...")
        series_map = self.cfg["data"]["macro_series"]
        data = {}

        for name, sid in series_map.items():
            s = self._fred(sid)
            if s is not None:
                data[name] = s
            else:
                # Fallback yfinance pour VIX
                if name == "vix":
                    try:
                        v = yf.download("^VIX", start=self.start, end=self.end,
                                        progress=False)["Close"].squeeze()
                        data[name] = v
                    except Exception:
                        pass

        idx = pd.date_range(self.start, self.end, freq="B")
        macro = pd.DataFrame(data).reindex(idx).ffill().bfill()

        if "us_yield_10y" in macro and "us_yield_2y" in macro:
            macro["yield_curve"] = macro["us_yield_10y"] - macro["us_yield_2y"]

        # Z-scores rolling 252j
        for col in list(macro.columns):
            rm = macro[col].rolling(252).mean()
            rs = macro[col].rolling(252).std()
            macro[f"{col}_z"] = (macro[col] - rm) / rs.replace(0, np.nan)

        macro.to_parquet(mf)
        logger.info(f"✅ Macro : {macro.shape[1]} séries")
        return macro

    def _fred(self, sid: str) -> Optional[pd.Series]:
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text), index_col=0, parse_dates=True)
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
            return s[self.start:self.end]
        except Exception:
            return None

    # ── Rendements Nets ────────────────────────────────────────────────────

    def net_returns(self, prices: pd.DataFrame, dividends: pd.DataFrame,
                    volumes: pd.DataFrame) -> pd.DataFrame:
        """Rendements totaux après fiscalité CTO France + purification Shariah."""
        price_ret = prices.pct_change()
        div_yield = (dividends / prices.shift(1)).fillna(0.0)

        flat = self.tax["flat_tax_pct"]                      # 30%
        wh_us = self.tax["us_dividend_withholding"]          # 15%
        wh_eu = self.tax["eu_dividend_withholding"]          # 15%
        purif = self.purif_rate                              # 3%

        div_net = div_yield.copy()

        for t in self.eu_tickers:
            if t in div_net.columns:
                # retenue source → flat tax → purification
                d = div_net[t] * (1 - wh_eu) * (1 - flat) * (1 - purif)
                div_net[t] = d

        for t in self.us_tickers:
            if t in div_net.columns:
                # retenue 15% US imputée sur flat tax FR (crédit partiel)
                residual_tax = max(flat - wh_us, 0)
                d = div_net[t] * (1 - wh_us) * (1 - residual_tax) * (1 - purif)
                div_net[t] = d

        total = price_ret + div_net

        # Masque les titres illiquides (volume = 0 pendant > 20j)
        illiquid = (volumes == 0).rolling(20).sum() >= 20
        total = total.where(~illiquid, np.nan)

        return total

    def detect_extremes(self, returns: pd.DataFrame) -> pd.DataFrame:
        eq = returns.mean(axis=1)
        mu = eq.rolling(252).mean()
        sigma = eq.rolling(252).std()
        z = (eq - mu) / sigma.replace(0, np.nan)
        return pd.DataFrame({
            "port_return": eq,
            "zscore": z,
            "is_extreme": z < -3.0,
            "is_crisis": z < -5.0,
        })

    def prepare(self, tickers: List[str], use_cache: bool = True) -> dict:
        logger.info("═══ PIPELINE DONNÉES ═══")
        p_raw, div, vol = self.fetch_prices(tickers, use_cache)
        fx = self.fetch_fx(use_cache)
        macro = self.fetch_macro(use_cache)
        prices = self.convert_to_eur(p_raw, fx)
        returns = self.net_returns(prices, div, vol)
        extremes = self.detect_extremes(returns)
        logger.info("═══ PIPELINE OK ═══")
        return dict(prices=prices, dividends=div, volumes=vol,
                    returns=returns, macro=macro, fx=fx, extremes=extremes)


if __name__ == "__main__":
    cfg = load_config()
    tickers = cfg["universe"]["eu_tickers"] + cfg["universe"]["us_tickers"]
    dp = DataPipeline(cfg)
    data = dp.prepare(tickers, use_cache=True)
    print(f"Prix : {data['prices'].shape}, Macro : {data['macro'].shape}")
    print(data["prices"].tail(3))
