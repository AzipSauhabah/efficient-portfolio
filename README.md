# 🕌 Shariah Efficient Portfolio — IA Quantamentale

> **Portefeuille efficient Shariah (EU + US) avec stack IA complète — 100% gratuit, lancable en 1 clic depuis Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzipSauhabah/shariah-portfolio/blob/main/notebooks/shariah_portfolio.ipynb)
[![Daily Update](https://github.com/AzipSauhabah/shariah-portfolio/actions/workflows/update_data.yml/badge.svg)](https://github.com/AzipSauhabah/shariah-portfolio/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Lancer la démo en 1 clic

**Cliquez sur le badge Colab ci-dessus** — aucune installation requise.

> ⚡ **Recommandé** : `Runtime → Changer le type de runtime → GPU T4` (gratuit, 3x plus rapide)

---

## 🧠 Stack IA

| Modèle | Rôle | Technologie |
|--------|------|-------------|
| **HMM** | Détection régimes marché (bull/bear/crisis) | `hmmlearn` |
| **XGBoost + SHAP** | Prédiction alpha par titre (interprétable) | `xgboost`, `shap` |
| **LSTM** | Prédiction volatilité 21 jours | `PyTorch` |
| **FinBERT** | Sentiment NLP sur news géopolitiques | `HuggingFace Transformers` |
| **PPO Agent** | Rebalancing intelligent (Reinforcement Learning) | `Stable-Baselines3` |
| **Black-Litterman** | Optimisation avec views ML | `cvxpy` |

---

## 📊 Stratégies disponibles

```python
from src.strategies import get_strategy

strategies = {
    "bl_quantamental": get_strategy("bl_quantamental", cfg, ml_engine=engine, optimizer=opt),
    "max_sharpe":      get_strategy("max_sharpe", cfg),
    "min_variance":    get_strategy("min_variance", cfg),
    "risk_parity":     get_strategy("risk_parity", cfg),
    "momentum":        get_strategy("momentum", cfg),
    "equal_weight":    get_strategy("equal_weight", cfg),
}
```

---

## 🏗️ Architecture du Projet

```
shariah-portfolio/
├── 📓 notebooks/
│   └── shariah_portfolio.ipynb    ← Colab principal (bouton ci-dessus)
│
├── 🧠 src/
│   ├── universe.py                ← Filtre Shariah (MSCI Islamic ∩ DJIM)
│   ├── data_pipeline.py           ← Prix ajustés + dividendes + macro
│   ├── ml_engine.py               ← HMM + XGBoost + LSTM + FinBERT
│   ├── rl_agent.py                ← Agent PPO rebalancing (Stable-Baselines3)
│   ├── optimizer.py               ← Black-Litterman + covariance ensemble
│   ├── strategies.py              ← 6 stratégies comparables
│   ├── costs.py                   ← Broker + fiscalité CTO France
│   ├── backtest.py                ← Walk-forward engine + métriques
│   └── live_data.py               ← Mise à jour incrémentale automatique
│
├── ⚙️ config.yaml                 ← TOUS les paramètres ici
├── 📋 requirements.txt
├── 🔄 .github/workflows/
│   └── update_data.yml            ← Cron job GitHub Actions (données live)
└── 📖 README.md
```

---

## ⚙️ Paramètres configurables (config.yaml)

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `portfolio.initial_capital` | 30000 | Capital initial (€) |
| `broker.name` | `fortuneo` | Broker (fortuneo/boursorama/ibkr/swissquote) |
| `shariah.filter_mode` | `intersection` | Mode filtre (intersection/union/msci_only/djim_only) |
| `covariance.method` | `ensemble` | Méthode cov (ensemble/ledoit_wolf/dcc_garch) |
| `ai.*.enabled` | `true` | Active/désactive chaque module IA |
| `extreme_events.method` | `ensemble` | Modèle extrêmes (ensemble/copula/monte_carlo/stress) |

---

## 💰 Comparaison des brokers (capital 30 000€)

| Broker | Achat EU | Achat US | Drag total/an |
|--------|----------|----------|---------------|
| **Fortuneo** | 2€ fixe | **50€ fixe** ⚠️ | ~3-5% |
| **Boursorama** | 1.99€ | 0.99% (min 7.99€) | ~1-2% |
| **IBKR** | 0.05% | 0.005$/action | ~0.2-0.5% |
| **Swissquote** | 9€ | 9€ | ~1-2% + garde |

> ⚠️ **Important** : avec 30k€, Fortuneo US est très pénalisant.
> Le moteur cost-aware limite automatiquement les ordres US.

---

## 📥 Mise sur GitHub (instructions)

### Option A — Première fois (nouveau repo)

```bash
# 1. Téléchargez le repo depuis Claude
# 2. Créez un repo sur github.com (bouton "New repository")
# 3. Dans votre terminal :

cd shariah-portfolio

git init
git add .
git commit -m "🕌 Initial commit — Shariah Portfolio IA"
git branch -M main
git remote add origin https://github.com/AzipSauhabah/shariah-portfolio.git
git push -u origin main
```

### Option B — Mise à jour (repo existant)

```bash
cd shariah-portfolio
git add .
git commit -m "✨ Update $(date +'%Y-%m-%d')"
git push
```

### Activer GitHub Actions (données live)

```bash
# Dans github.com → votre repo → Settings → Secrets and variables → Actions
# Ajouter (optionnel) :
#   FRED_API_KEY  → clé gratuite sur https://fred.stlouisfed.org/docs/api/api_key.html
```

---

## 🔄 Données Live & Auto-Update

Les données sont mises à jour automatiquement via **GitHub Actions** :
- **Prix** : tous les jours ouvrés à 18h UTC
- **Macro** : chaque semaine
- **Filtre Shariah** : chaque année
- **Modèles ML** : sur déclenchement manuel ou si régime change

```python
# Mise à jour manuelle depuis Python
from src.live_data import DataUpdater
updater = DataUpdater(cfg)
updater.run_full_update(tickers)
```

---

## 🚀 Lancer en local

```bash
# Cloner
git clone https://github.com/AzipSauhabah/shariah-portfolio.git
cd shariah-portfolio

# Environnement virtuel
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Installation
pip install -r requirements.txt

# Lancer le backtest complet
python src/backtest.py

# Mise à jour des données
python src/live_data.py
```

---

## 📋 Fiscalité CTO France (intégrée)

- **Flat tax 30%** (PFU) sur chaque cession en plus-value
- **Retenue à la source US 15%** (convention fiscale FR-US), imputée sur la flat tax
- **Purification Shariah 3%** sur les dividendes
- **PRU (Prix de Revient Unitaire)** tracké par position pour calcul exact

---

## 📚 Références

- Black, F. & Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal*
- Ledoit, O. & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix
- Engle, R. (2002). Dynamic Conditional Correlation. *JBES*
- FinBERT: [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)
- AAOIFI Shariah Standards — [aaoifi.com](https://aaoifi.com)
- MSCI Islamic Index Methodology — [msci.com](https://msci.com)

---

## 📄 License

MIT — libre d'utilisation, modification et distribution.

---
