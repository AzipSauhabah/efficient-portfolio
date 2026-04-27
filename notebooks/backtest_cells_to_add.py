# ═══════════════════════════════════════════════════════════════════
# CELLULES À AJOUTER DANS shariah_portfolio.ipynb
# Copiez ces blocs APRÈS la cellule 9 (Black-Litterman)
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# CELLULE 9b : Installation Backtrader
# ─────────────────────────────────────────────────────────────────
"""
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install',
                'backtrader', '-q'], check=False)
print('✅ Backtrader installé')
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 10 : ⚙️ Configuration du Backtest Complet
# ─────────────────────────────────────────────────────────────────
"""
from backtesting_framework import (
    BacktestOrchestrator, fetch_ohlcv, compute_metrics
)
from strategies import (
    EqualWeightStrategy, MinVarianceStrategy, MaxSharpeStrategy,
    MomentumStrategy, RiskParityStrategy, BLQuantamentalStrategy,
)

# ── Paramètres du backtest ─────────────────────────────────────
EXECUTION_MODE      = 'vwap'          # 'vwap' | 'twap' | 'market'
VALIDATION_METHODS  = ['walk_forward', 'monte_carlo', 'cpcv']
                                       # Retirez 'cpcv' si trop lent sur CPU
N_MCCV_SPLITS       = 30              # Nb de splits Monte Carlo (30-100)
CPCV_N_GROUPS       = 6               # Groupes CPCV
CPCV_K_TEST         = 2               # Groupes de test simultanés
CPCV_MAX_COMBOS     = 10              # Max combinaisons (réduit si lent)

# ── Stratégies à comparer ──────────────────────────────────────
tickers_ok = [t for t in compliant_tickers if t in data['returns'].columns]

strategies = {
    'BL_Quantamental': BLQuantamentalStrategy(cfg, ml_engine=engine, optimizer=optimizer),
    'MaxSharpe':       MaxSharpeStrategy(cfg),
    'MinVariance':     MinVarianceStrategy(cfg),
    'RiskParity':      RiskParityStrategy(cfg),
    'Momentum_12_1':   MomentumStrategy(cfg, window_months=12, skip_months=1),
    'EqualWeight':     EqualWeightStrategy(cfg),
}

print(f'✅ {len(strategies)} stratégies configurées :')
for name in strategies:
    print(f'  • {name}')
print(f'\\n⚙️  Mode exécution  : {EXECUTION_MODE.upper()}')
print(f'   Validations     : {VALIDATION_METHODS}')
print(f'   Splits MCCV     : {N_MCCV_SPLITS}')
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 11 : 📥 Téléchargement données OHLCV (pour VWAP/TWAP)
# ─────────────────────────────────────────────────────────────────
"""
print('📥 Téléchargement données OHLCV (Open/High/Low/Close/Volume)...')
print('   Nécessaire pour simulation VWAP/TWAP réaliste')
print('   (peut prendre 2-5 minutes la première fois)\\n')

ohlcv_data = fetch_ohlcv(
    tickers_ok,
    cfg['data']['start_date'],
    cfg['data']['end_date'],
    cfg['data']['cache_path'],
)

print(f'\\n✅ OHLCV disponible : {len(ohlcv_data)} tickers')
for ticker, df in list(ohlcv_data.items())[:3]:
    print(f'  {ticker}: {df.index[0].date()} → {df.index[-1].date()} ({len(df)} jours)')
print('  ...')
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 12 : 🚀 BACKTEST COMPLET — Toutes stratégies × 3 méthodes
# ─────────────────────────────────────────────────────────────────
"""
from backtesting_framework import MonteCarloCVValidator, CPCVValidator

print('🚀 Lancement du backtesting framework complet...')
print('   Backtrader event-driven + VWAP + WF + MCCV + CPCV')
print('   Durée estimée : 15-40 min (GPU accélère LSTM/FinBERT)\\n')

# Instanciation avec paramètres personnalisés
orch = BacktestOrchestrator(cfg)
orch.mccv = MonteCarloCVValidator(cfg, n_splits=N_MCCV_SPLITS)
orch.cpcv = CPCVValidator(
    cfg,
    n_groups=CPCV_N_GROUPS,
    k_test=CPCV_K_TEST,
    max_combos=CPCV_MAX_COMBOS,
)

backtest_results = orch.run_all(
    strategies,
    data['returns'],
    ohlcv_data,
    tickers_ok,
    execution_mode=EXECUTION_MODE,
    validation_methods=VALIDATION_METHODS,
)

# ── Tableau comparatif complet ─────────────────────────────────
comp = backtest_results['comparison']
print('\\n' + '═'*80)
print('  TABLEAU COMPARATIF — TOUTES STRATÉGIES × MÉTHODES DE VALIDATION')
print('═'*80)

display_cols = ['strategy', 'validation', 'ann_return', 'volatility',
                'sharpe', 'sortino', 'max_dd', 'calmar',
                'hit_rate', 'prob_overfit', 'deflated_sharpe', 'overfit_wf']
available_cols = [c for c in display_cols if c in comp.columns]

import pandas as pd
pd.options.display.float_format = '{:.3f}'.format
print(comp[available_cols].to_string(index=False))

print(f'\\n🏆 Meilleure stratégie OOS : {backtest_results[\"best\"].upper()}')
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 13 : 📊 Dashboard de comparaison des stratégies
# ─────────────────────────────────────────────────────────────────
"""
# Collecte les rendements OOS de chaque stratégie (walk-forward)
returns_by_strategy = {}
for strat_name, strat_results in backtest_results['results'].items():
    # Préférence : WF > MCCV > CPCV pour les rendements cumulés
    for method in ['walk_forward', 'monte_carlo', 'cpcv']:
        if method in strat_results and len(strat_results[method].returns) > 0:
            returns_by_strategy[strat_name] = strat_results[method].returns
            break

orch.plot_comparison(backtest_results, returns_by_strategy)
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 14 : 🔬 Analyse approfondie — CPCV & Overfitting
# ─────────────────────────────────────────────────────────────────
"""
import matplotlib.pyplot as plt
import numpy as np

comp = backtest_results['comparison']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Deflated Sharpe Ratio ─────────────────────────────────────
ax1 = axes[0]
cpcv_data = comp[comp['validation'] == 'cpcv'].dropna(subset=['deflated_sharpe'])
if not cpcv_data.empty:
    colors = ['#27ae60' if v > 0 else '#e74c3c'
              for v in cpcv_data['deflated_sharpe']]
    bars = ax1.barh(cpcv_data['strategy'], cpcv_data['deflated_sharpe'],
                    color=colors, alpha=0.85)
    ax1.axvline(0, color='black', linewidth=1.2, linestyle='--')
    ax1.set_title('Deflated Sharpe Ratio (CPCV)\\n'
                  'Verde = robuste | Rouge = suspect', fontsize=11)
    ax1.set_xlabel('Sharpe moyen - 1σ')
    for bar, val in zip(bars, cpcv_data['deflated_sharpe']):
        ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}', va='center', fontsize=9)

# ── IS vs OOS Sharpe (WFA) ────────────────────────────────────
ax2 = axes[1]
wf_data = comp[comp['validation'] == 'walk_forward']
if not wf_data.empty and 'overfit_wf' in wf_data.columns:
    valid = wf_data.dropna(subset=['overfit_wf', 'sharpe'])
    if not valid.empty:
        sc = ax2.scatter(
            valid['sharpe'],
            1 - valid['overfit_wf'],  # OOS/IS ratio
            s=150,
            c=valid['overfit_wf'],
            cmap='RdYlGn_r',
            vmin=0, vmax=1,
            zorder=5,
        )
        for _, row in valid.iterrows():
            ax2.annotate(row['strategy'],
                         (row['sharpe'], 1 - row['overfit_wf']),
                         textcoords='offset points', xytext=(5, 5),
                         fontsize=8)
        ax2.axhline(1.0, color='green', linestyle='--', alpha=0.5,
                    label='OOS = IS (parfait)')
        ax2.axhline(0.5, color='orange', linestyle='--', alpha=0.5,
                    label='OOS = 50% IS (seuil)')
        plt.colorbar(sc, ax=ax2, label='Score overfitting')
        ax2.set_xlabel('Sharpe OOS')
        ax2.set_ylabel('Ratio OOS/IS Sharpe')
        ax2.set_title('IS vs OOS Sharpe\\n(Walk-Forward Analysis)', fontsize=11)
        ax2.legend(fontsize=8)

plt.suptitle('🔬 Analyse Robustesse & Overfitting',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('data/cache/overfitting_analysis.png', bbox_inches='tight', dpi=150)
plt.show()
"""

# ─────────────────────────────────────────────────────────────────
# CELLULE 15 : 💾 Export complet des résultats backtest
# ─────────────────────────────────────────────────────────────────
"""
import json
import pandas as pd
from pathlib import Path

Path('data/cache').mkdir(exist_ok=True)

# Export tableau comparatif
comp.to_csv('data/cache/strategy_comparison.csv', index=False)

# Export métriques JSON par stratégie
all_metrics = {}
for strat_name, strat_results in backtest_results['results'].items():
    all_metrics[strat_name] = {}
    for method, result in strat_results.items():
        m = {k: (float(v) if isinstance(v, (float, int, np.floating)) else str(v))
             for k, v in result.metrics.items()}
        all_metrics[strat_name][method] = m

with open('data/cache/all_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Export rendements OOS par stratégie
for strat_name, strat_results in backtest_results['results'].items():
    for method, result in strat_results.items():
        if len(result.returns) > 0:
            fname = f'data/cache/returns_{strat_name}_{method}.csv'
            result.returns.to_csv(fname)

print('✅ Exports sauvegardés dans data/cache/ :')
print('  📊 strategy_comparison.csv')
print('  📋 all_metrics.json')
print('  📈 returns_<strategie>_<methode>.csv (x N)')
print('  🖼️  strategy_comparison.png')
print('  🖼️  overfitting_analysis.png')

# Téléchargement depuis Colab
try:
    from google.colab import files
    for fname in ['strategy_comparison.csv', 'all_metrics.json',
                  'strategy_comparison.png', 'overfitting_analysis.png']:
        try:
            files.download(f'data/cache/{fname}')
        except Exception:
            pass
except ImportError:
    pass
"""
