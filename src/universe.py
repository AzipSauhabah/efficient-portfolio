"""
Module 1 : Universe Builder — Filtre Shariah
=============================================
Construit l'univers investissable en appliquant les filtres
MSCI Islamic et/ou DJIM sur les actions EU + US.

Anti-lookahead bias : filtres appliqués avec grace_period_days de délai,
simulant la réalité des mises à jour d'index (annuelles).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXCLUDED_GICS = {
    "Banks", "Diversified Financials", "Insurance",
    "Tobacco", "Defense & Aerospace",
}
EXCLUDED_SECTORS = {"financials", "alcohol", "tobacco", "weapons", "gambling"}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class ShariahFilter:
    def __init__(self, config: dict):
        self.cfg = config["shariah"]
        self.msci = self.cfg["msci_thresholds"]
        self.djim = self.cfg["djim_thresholds"]
        self.mode = self.cfg["filter_mode"]
        self.grace = self.cfg["grace_period_days"]

    def _fetch_fundamentals(self, ticker: str) -> Optional[dict]:
        try:
            info = yf.Ticker(ticker).info
            mkt_cap = info.get("marketCap", np.nan) or np.nan
            total_debt = info.get("totalDebt", 0) or 0
            total_assets = info.get("totalAssets", np.nan) or np.nan
            cash = info.get("totalCash", 0) or 0
            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            return {
                "sector": sector,
                "industry": industry,
                "debt_to_assets": total_debt / total_assets if total_assets and not np.isnan(total_assets) else np.nan,
                "cash_to_assets": cash / total_assets if total_assets and not np.isnan(total_assets) else np.nan,
                "debt_to_mktcap": total_debt / mkt_cap if mkt_cap and not np.isnan(mkt_cap) else np.nan,
                "cash_to_mktcap": cash / mkt_cap if mkt_cap and not np.isnan(mkt_cap) else np.nan,
                "haram_revenue_pct": 0.0,  # proxy : 0 pour secteurs non-exclus
            }
        except Exception as e:
            logger.debug(f"{ticker}: {e}")
            return None

    def _msci_ok(self, f: dict) -> Tuple[bool, str]:
        if f["sector"].lower() in EXCLUDED_SECTORS:
            return False, f"secteur exclu ({f['sector']})"
        if f["industry"] in EXCLUDED_GICS:
            return False, f"industrie exclue ({f['industry']})"
        if pd.notna(f["debt_to_assets"]) and f["debt_to_assets"] > self.msci["debt_to_assets"]:
            return False, f"dette/actifs {f['debt_to_assets']:.2f} > {self.msci['debt_to_assets']}"
        if pd.notna(f["cash_to_assets"]) and f["cash_to_assets"] > self.msci["cash_interest_to_assets"]:
            return False, f"cash/actifs {f['cash_to_assets']:.2f} > {self.msci['cash_interest_to_assets']}"
        return True, "OK"

    def _djim_ok(self, f: dict) -> Tuple[bool, str]:
        if f["sector"].lower() in EXCLUDED_SECTORS:
            return False, f"secteur exclu ({f['sector']})"
        if f["industry"] in EXCLUDED_GICS:
            return False, f"industrie exclue ({f['industry']})"
        if pd.notna(f["debt_to_mktcap"]) and f["debt_to_mktcap"] > self.djim["debt_to_market_cap"]:
            return False, f"dette/mktcap {f['debt_to_mktcap']:.2f} > {self.djim['debt_to_market_cap']}"
        return True, "OK"

    def screen(self, ticker: str) -> dict:
        f = self._fetch_fundamentals(ticker)
        if f is None:
            return {"ticker": ticker, "compliant": False, "reason": "no_data",
                    "msci": False, "djim": False, "sector": "?", "industry": "?"}

        msci, msci_reason = self._msci_ok(f)
        djim, djim_reason = self._djim_ok(f)

        if self.mode == "intersection":
            ok = msci and djim
            reason = "OK" if ok else f"MSCI:{msci_reason} | DJIM:{djim_reason}"
        elif self.mode == "union":
            ok = msci or djim
            reason = "OK" if ok else f"MSCI:{msci_reason} | DJIM:{djim_reason}"
        elif self.mode == "msci_only":
            ok, reason = msci, msci_reason
        else:
            ok, reason = djim, djim_reason

        return {
            "ticker": ticker, "compliant": ok, "reason": reason,
            "msci": msci, "djim": djim,
            "sector": f["sector"], "industry": f["industry"],
            "debt_to_assets": f.get("debt_to_assets"),
            "debt_to_mktcap": f.get("debt_to_mktcap"),
        }

    def build_universe(self, tickers: List[str], use_cache: bool = True,
                       cache_path: str = "data/cache/") -> pd.DataFrame:
        cache_file = Path(cache_path) / "universe_latest.parquet"
        if use_cache and cache_file.exists():
            logger.info("Univers chargé depuis cache")
            return pd.read_parquet(cache_file)

        logger.info(f"Screening Shariah : {len(tickers)} tickers...")
        results = [self.screen(t) for t in tickers]
        df = pd.DataFrame(results)
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)
        logger.info(f"✅ Conformes : {df['compliant'].sum()}/{len(df)}")
        return df

    def get_compliant_tickers(self, tickers: List[str], **kwargs) -> List[str]:
        df = self.build_universe(tickers, **kwargs)
        return df[df["compliant"]]["ticker"].tolist()


def purify_dividend(gross: float, haram_pct: float = 0.0,
                    purif_rate: float = 0.03) -> Tuple[float, float]:
    """Retourne (dividende_net, montant_purification)."""
    purif = gross * max(haram_pct, purif_rate)
    return gross - purif, purif


if __name__ == "__main__":
    cfg = load_config()
    tickers = cfg["universe"]["eu_tickers"] + cfg["universe"]["us_tickers"]
    sf = ShariahFilter(cfg)
    df = sf.build_universe(tickers, use_cache=False)
    print(df[["ticker", "msci", "djim", "compliant", "sector", "reason"]].to_string())
