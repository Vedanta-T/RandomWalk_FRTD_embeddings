"""
Synthetic structural benchmark.
Use the EXPERIMENTS dictionary to configure the experiments to run.
Usage:  python benchmark_synthetic.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import datasets as ds
import embeddings as em


from benchmark import evaluate


# ============================================================================
# FORMATTING
# ============================================================================

BOLD = '\033[1m'
RESET = '\033[0m'

np.seterr(all='ignore')

def fmt(value, best):
    cell = f'{value:.3f}'
    return f'{BOLD}{cell}{RESET}' if np.isclose(value, best) else cell

def hellinger(p, q):
    return float(np.sqrt(0.5) *
                 np.linalg.norm(np.sqrt(np.abs(p)) - np.sqrt(np.abs(q))))


# ============================================================================
# METHODS
# ============================================================================

ALGORITHMS = {
    'RolX':      (em.rolx,       hellinger),
    'GraphWave': (em.graphwave,  'l2'),
    'struc2vec': (em.struc2vec,  'l2'),
    'FRTD':      (em.frtd,       hellinger),
    'degree':    (em.degree,     'l2'),
    'spectral':  (em.spectral_embedding, 'l2')
}


# ============================================================================
# OUTPUT
# ============================================================================

OUTDIR = 'role_benchmarks/synthetic_results'
os.makedirs(OUTDIR, exist_ok=True)


# ============================================================================
# EXPERIMENTS
# ============================================================================

EXPERIMENTS = {
    'RGG_vs_ER_scaling': {
        'loader': ds.synthetic_rgg_vs_er,
        'parameter': 'n1',
        'values': [32, 64, 128, 256, 512, 1024],
        'fixed': {
            'n2': None,
            'k': 8,
        },
        'repeats': 100,
    },
}


'''

    # ----------------------------------------------------------------------
    # Another example
    # ----------------------------------------------------------------------

    'PA_vs_DD_scaling': {
        'loader': ds.synthetic_pa_vs_dd,
        'parameter': 'N',
        'values': [64, 128, 256, 512, 1024],
        'fixed': {
            'm': 6,
            'q': None,
            'alpha': 0.5,
        },
        'repeats': 100,
    },
    'SBM_scaling': {
        'loader': ds.sbm,
        'parameter': 'n',
        'values': [32, 64, 128, 256, 512, 1024, 2048],
        'fixed': {
            'c_in': 8,
            'c_out': 2,
        },
        'repeats': 100,
    },
'''

# ============================================================================
# STORAGE
# ============================================================================

all_rows = []


# ============================================================================
# MAIN LOOP
# ============================================================================

for exp_name, cfg in EXPERIMENTS.items():

    print('\n' + '=' * 80)
    print(exp_name)
    print('=' * 80)

    loader = cfg['loader']
    parameter = cfg['parameter']
    values = cfg['values']
    fixed = cfg['fixed']
    repeats = cfg['repeats']

    for val in values:

        kwargs = dict(fixed)
        kwargs[parameter] = val

        if exp_name == 'RGG_vs_ER_scaling':
            kwargs['n2'] = val

        print(f'\n{parameter} = {val}')

        rows = []

        for rep in tqdm(range(repeats)):

            kwargs['seed'] = np.random.randint(0, 1_000_000)

            G, labels = loader(**kwargs)

            rep_results = {}

            for alg_name, (fn, metric) in ALGORITHMS.items():

                t0 = time.time()
                emb, nodes = fn(G)
                _, f1 = evaluate(emb, nodes, labels, metric=metric)

                rep_results[alg_name] = (f1, time.time() - t0)

            rows.append(rep_results)

        # ------------------------------------------------------------------
        # Aggregate
        # ------------------------------------------------------------------

        summary = {}

        for alg_name in ALGORITHMS:

            f1s = [r[alg_name][0] for r in rows]
            runtimes = [r[alg_name][1] for r in rows]

            summary[alg_name] = {
                'f1_mean': np.mean(f1s),
                'f1_std': np.std(f1s),
                'runtime_mean': np.mean(runtimes),
                'runtime_std': np.std(runtimes),
            }

        best = max(
            summary[a]['f1_mean']
            for a in summary
        )

        print(
            f'{"algorithm":<12s} '
            f'{"f1(mean)":>10s} '
            f'{"std":>8s} '
            f'{"runtime":>10s}'
        )

        for alg_name in ALGORITHMS:

            s = summary[alg_name]

            print(
                f'{alg_name:<12s} '
                f'{fmt(s["f1_mean"], best):>10s} '
                f'{s["f1_std"]:>8.3f} '
                f'{s["runtime_mean"]:>10.2f}'
            )

            all_rows.append({
                'experiment': exp_name,
                'parameter': parameter,
                'value': val,
                'algorithm': alg_name,
                'f1_mean': s['f1_mean'],
                'f1_std': s['f1_std'],
                'runtime_mean': s['runtime_mean'],
                'runtime_std': s['runtime_std'],
            })


# ============================================================================
# SAVE CSV
# ============================================================================

df = pd.DataFrame(all_rows)

csv_path = os.path.join(
    OUTDIR,
    'synthetic_results.csv'
)

df.to_csv(csv_path, index=False)

print('\nSaved:')
print(csv_path)


# ============================================================================
# PLOTTING
# ============================================================================

plt.rcParams['figure.dpi'] = 200

colors = ['blue', 'red', 'green', 'purple', 'k', 'magenta']
markers = ['x', 's', '^', 'o', '*']


for exp_name in df['experiment'].unique():

    sub = df[df['experiment'] == exp_name]

    parameter = sub['parameter'].iloc[0]

    # ----------------------------------------------------------------------
    # F1 plot
    # ----------------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    i=0

    for alg_name in ALGORITHMS:

        d = sub[sub['algorithm'] == alg_name]

        if len(d) == 0:
            continue

        x = d['value'].values
        y = d['f1_mean'].values
        e = d['f1_std'].values

        plt.plot(
            x,
            y,
            color=colors[i],
            marker=markers[i],
            label=alg_name,
        )
        #Uncomment below for errorbars
        #plt.fill_between(
        #    x,
        #    y - e,
        #    y + e,
        #    color=colors[i],
        #    alpha=0.15,
        #)
        i+=1

    plt.xlabel('n', fontsize=23, font='serif')
    plt.ylabel('Macro-F1 score', fontsize=23, font='serif')
    #plt.title(exp_name)
    #plt.legend()
    plt.xscale('log')
    plt.legend(fontsize=22, prop={'family':'serif', 'size':18})
    plt.tight_layout()

    path = os.path.join(
        OUTDIR,
        f'{exp_name}_f1.pdf'
    )

    plt.savefig(path, bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------------------
    # Runtime plot
    # ----------------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    i=0
    for alg_name in ALGORITHMS:

        d = sub[sub['algorithm'] == alg_name]

        if len(d) == 0:
            continue

        x = d['value'].values
        y = d['runtime_mean'].values
        e = d['runtime_std'].values

        plt.plot(
            x,
            y,
            color=colors[i],
            marker=markers[i],
            label=alg_name,
        )
        #Uncomment below for errorbars
        #plt.fill_between(
        #    x,
        #    y - e,
        #    y + e,
        #    color=colors[i],
        #    alpha=0.15,
        #)
        i+=1

    plt.xlabel('n', fontsize=23, font='serif')
    plt.ylabel('Runtime (s)', fontsize=23, font='serif')
    plt.title(exp_name)
    #plt.legend()
    plt.legend(fontsize=22, prop={'family':'serif', 'size':18})
    plt.yscale('log')
    plt.xscale('log')
    plt.title(f'{exp_name} runtime')
    #plt.legend()
    plt.tight_layout()

    path = os.path.join(
        OUTDIR,
        f'{exp_name}_runtime.pdf'
    )

    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print('\n' + '=' * 80)
print('AVERAGE PERFORMANCE')
print('=' * 80)

for exp_name in df['experiment'].unique():

    print(f'\n{exp_name}')

    sub = df[df['experiment'] == exp_name]

    avg = (
        sub
        .groupby('algorithm')['f1_mean']
        .mean()
        .sort_values(ascending=False)
    )

    best = avg.max()

    for alg_name, val in avg.items():

        print(
            f'  {alg_name:<12s} '
            f'{fmt(val, best)}'
        )

print('\nPlots written to:')
print(OUTDIR)