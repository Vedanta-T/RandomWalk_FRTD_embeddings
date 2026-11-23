"""Role-extraction benchmark on real-world networks (Table 1 of the paper).

Five node embeddings (degree, RolX, GraphWave, struc2vec, FRTD) are scored by how
well a 5-nearest-neighbour classifier recovers node metadata ("roles") from them,
measured by macro-F1 under 5-fold stratified cross validation.

Distances for the k-NN classifier:
  * FRTD and RolX: total variation distance, TVD(p, q) = 1/2 sum_t |p(t) - q(t)|.
    Implemented as metric='manhattan': TVD is half the L1 distance, so the
    nearest neighbours (and hence the scores) are identical.
  * degree, GraphWave, struc2vec: Euclidean (L2).

Averaging (a single CV split / struc2vec run is not reliable: several winners in a
single run are decided by margins far below the run-to-run noise):
  * The 5-fold CV is repeated with N_CV_SEEDS different shuffles; all methods are
    evaluated on the same splits. This also averages over k-NN distance ties.
  * struc2vec (the only stochastic method) is run N_S2V_RUNS times with seeds
    0..N_S2V_RUNS-1; every run is evaluated on every split.
  * One "repeat" = mean macro-F1 over the 5 folds of one split (and one struc2vec
    run). Reported: mean over repeats and its standard deviation, in concise
    notation, e.g. 0.535(109) = 0.535 +- 0.109; P(best) = share of
    (struc2vec run, split) combinations in which the method has the top score.

Hyperparameters are the defaults in embeddings.py (FRTD: K = 128 return-time
steps). Times are embedding times only (mean over runs).

Edit EXPERIMENTS and ALGORITHMS below to change the benchmark.
Usage:
  python benchmark.py                          # 10 random CV shuffle seeds (printed and saved)
  python benchmark.py --standard-seeds            # CV shuffle seeds 0..9
  python benchmark.py --cv-seed-list 3 17 42   # exactly these CV shuffle seeds
struc2vec always uses seeds 0..N_S2V_RUNS-1.
Writes one score per repeat to results/benchmark_repeats.csv and the summary table to
results/benchmark_table.md.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

import datasets as ds
import embeddings as em

N_CV_SEEDS = 10
N_S2V_RUNS = 5
TVD = 'manhattan'   # TVD = L1 / 2
L2 = 'euclidean'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

# name -> (embedding function, k-NN metric, stochastic?)
ALGORITHMS = {
    'degree':    (em.degree,    L2,  False),
    'RolX':      (em.rolx,      TVD, False),
    'GraphWave': (em.graphwave, L2,  False),
    'struc2vec': (em.struc2vec, L2,  True),
    'FRTD':      (em.frtd,      TVD, False),
}

DATASETS = {
    'adjnoun':         ds.adjnoun,
    'celegans':        ds.celegans_neurons,
    'webkb-cornell':   lambda: ds.webkb('cornell'),
    'webkb-texas':     lambda: ds.webkb('texas'),
    'webkb-wisconsin': lambda: ds.webkb('wisconsin'),
    'airports-brazil': lambda: ds.airports('brazil'),
    'airports-europe': lambda: ds.airports('europe'),
    'openflights':     ds.ATN,
}


def evaluate(emb, nodes, labels, metric, n_splits=5, seed=0):
    """5-NN classifier, stratified k-fold CV. Returns (mean accuracy, mean macro-F1)."""
    y = np.array([labels[v] for v in nodes])
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    clf = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='brute')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(clf, emb, y, cv=cv, scoring=('accuracy', 'f1_macro'), n_jobs=1)
    return scores['test_accuracy'].mean(), scores['test_f1_macro'].mean()


def concise(value, unc, dec=3):
    """0.535 +- 0.109 -> '0.535(109)' (uncertainty in units of the last digit, just for reporting)."""
    return f'{value:.{dec}f}({int(round(unc * 10 ** dec))})'


STANDARD_CV_SEEDS = list(range(10))   # Standard CV shuffle seeds (0..9)


def cv_seed_list(n, seed_list=None, standard_seeds=False):
    """Seeds for the CV shuffles: an explicit list, standard seeds 0-9, or (default) n fresh random seeds."""
    if seed_list:
        return list(seed_list)
    if standard_seeds:
        return list(STANDARD_CV_SEEDS)
    return np.random.default_rng().integers(0, 2**31 - 1, size=n).tolist()


def run(names, cv_seeds, n_runs):
    rows = []
    for dname in names:
        G, labels = DATASETS[dname]()
        print(f'\n{dname}: n={G.number_of_nodes()}, e={G.number_of_edges()}, '
              f'roles={len(set(labels[v] for v in G))}', flush=True)
        for aname, (fn, metric, stochastic) in ALGORITHMS.items():
            for r in (range(n_runs) if stochastic else [-1]):
                t0 = time.time()
                emb, nodes = fn(G, seed=r) if stochastic else fn(G)
                t_embed = time.time() - t0
                for s in cv_seeds:
                    _, f1 = evaluate(np.asarray(emb, float), nodes, labels, metric, seed=s)
                    rows.append(dict(dataset=dname, n=G.number_of_nodes(), algorithm=aname,
                                     s2v_run=r, cv_seed=s, f1_macro=f1, embed_time=t_embed))
            print(f'  {aname:<10s} done', flush=True)
    return pd.DataFrame(rows)


def summarise(df):
    lines = ['| dataset | n | ' + ' | '.join(ALGORITHMS) + ' |', '|---|---|' + '---|' * len(ALGORITHMS)]
    print('\nmean macro-F1 (std over repeats), embedding time [s], P(best)')
    for dname, sub in df.groupby('dataset', sort=False):
        stats = sub.groupby('algorithm').agg(f1=('f1_macro', 'mean'), std=('f1_macro', 'std'),
                                             t=('embed_time', 'mean'))
        # P(best) over (struc2vec run, CV split) combinations
        det = sub[sub.s2v_run == -1].pivot_table(index='cv_seed', columns='algorithm', values='f1_macro')
        sto = sub[sub.s2v_run >= 0]
        wins = dict.fromkeys(ALGORITHMS, 0)
        for (r, s), g in sto.groupby(['s2v_run', 'cv_seed']):
            scores = det.loc[s].to_dict()
            scores.update(dict(zip(g.algorithm, g.f1_macro)))
            wins[max(scores, key=scores.get)] += 1
        total = sum(wins.values())
        best = stats.f1.max()
        print(f'\n{dname} (n={sub.n.iloc[0]})')
        cells = []
        for a in ALGORITHMS:
            st = stats.loc[a]
            mark = '*' if np.isclose(st.f1, best) else ' '
            print(f'  {mark} {a:<10s} {concise(st.f1, st["std"]):>12s} {st.t:9.2f}s  P(best) {wins[a] / total:4.0%}')
            c = f'{concise(st.f1, st["std"])}<br>{st.t:.2f}'
            cells.append(f'**{c}**' if mark == '*' else c)
        lines.append(f'| {dname} | {sub.n.iloc[0]} | ' + ' | '.join(cells) + ' |')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument('--cv-seeds', type=int, default=N_CV_SEEDS,
                    help='number of random CV shuffles (default %(default)s); the seeds are printed and '
                         'saved, so a run can be repeated with --cv-seed-list')
    seeds = ap.add_mutually_exclusive_group()
    seeds.add_argument('--standard-seeds', action='store_true',
                       help='use the CV shuffle seeds (0..9)')
    seeds.add_argument('--cv-seed-list', type=int, nargs='+', metavar='SEED',
                       help='use exactly these CV shuffle seeds')
    ap.add_argument('--s2v-runs', type=int, default=N_S2V_RUNS,
                    help='number of struc2vec runs, seeds 0..R-1 (default %(default)s)')
    args = ap.parse_args()
    names = args.datasets
    cv_seeds = cv_seed_list(args.cv_seeds, args.cv_seed_list, args.standard_seeds)
    print('CV shuffle seeds:', ' '.join(map(str, cv_seeds)))

    df = run(names, cv_seeds, args.s2v_runs)
    os.makedirs(OUTDIR, exist_ok=True)
    df.to_csv(os.path.join(OUTDIR, 'benchmark_repeats.csv'), index=False)
    table = summarise(df)
    with open(os.path.join(OUTDIR, 'benchmark_table.md'), 'w') as f:
        f.write('Mean macro-F1 over repeats, with the std over repeats in concise notation '
                '(0.535(109) = 0.535 ± 0.109); second line: embedding time (s). Bold = best mean.\n\n'
                f'CV shuffle seeds: {" ".join(map(str, cv_seeds))}; '
                f'struc2vec seeds: 0..{args.s2v_runs - 1}\n\n')
        f.write('\n'.join(table) + '\n')
    print(f'\nWrote {OUTDIR}/benchmark_repeats.csv and benchmark_table.md')


if __name__ == '__main__':
    main()
