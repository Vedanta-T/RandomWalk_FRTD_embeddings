"""Minimal role-extraction benchmark.

Runs five embedding methods on a set of role-labelled graphs and prints
macro-F1 from a 5-NN classifier (per-algorithm distance metric: Hellinger
for non-negative / PMF embeddings, L2 for the rest) under stratified
5-fold CV.

Usage:  python benchmark.py
Edit EXPERIMENTS and ALGORITHMS below to change the benchmark.
"""
import sys
import time


import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier

import datasets as ds
import embeddings as em


BOLD = '\033[1m' if sys.stdout.isatty() else ''
RESET = '\033[0m' if sys.stdout.isatty() else ''


def fmt(value, best, width=7):
    cell = f'{value:{width}.3f}'
    return f'{BOLD}{cell}{RESET}' if value == best else cell


def hellinger(p, q):
    return float(np.sqrt(0.5) *
                 np.linalg.norm(np.sqrt(np.abs(p)) - np.sqrt(np.abs(q))))

L1_distance = lambda p, q : np.sum(np.abs(p-q))


def evaluate(emb, nodes, labels, metric, n_splits=5):
    y = np.array([labels[v] for v in nodes])
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    clf = KNeighborsClassifier(n_neighbors=5, metric=metric, algorithm='brute')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_validate(clf, emb, y, cv=cv,
                            scoring=('accuracy', 'f1_macro'), n_jobs=1)
    return (scores['test_accuracy'].mean(),
            scores['test_f1_macro'].mean())


# (embedding fn, KNN distance metric).
# Hellinger only on the non-negative PMF-ish embeddings (RolX, FRTD);
# L2 for the signed/dense ones (degree, GraphWave, struc2vec).
ALGORITHMS = {
    'degree':    (em.degree,    'l2'),
    'FRTD':      (em.frtd,      hellinger),
    'struc2vec': (em.struc2vec, 'l2'),
    'RolX':      (em.rolx,      hellinger),
    'GraphWave': (em.graphwave, 'l2'),
}



DATASETS = {
    'adjnoun': ds.adjnoun,
    'celegans': ds.celegans_neurons,
    'webkb-cornell':   lambda: ds.webkb('cornell'),
    'webkb-texas':     lambda: ds.webkb('texas'),
    'webkb-wisconsin': lambda: ds.webkb('wisconsin'),
    'airports-brazil': lambda: ds.airports('brazil'),
    'airports-europe': lambda: ds.airports('europe'),
    'Openflights': lambda: ds.ATN()
}

def main():
    header = f'{"algorithm":<12s} {"f1":>7s} {"time(s)":>8s}'
    for dname, loader in DATASETS.items():
        G, labels = loader()
        n_roles = len(set(labels.values()))
        print(f'\n{dname} (n={G.number_of_nodes()}, '
              f'e={G.number_of_edges()}, roles={n_roles})')
        print(header)
        rows = []
        for aname, (fn, metric) in ALGORITHMS.items():
            t0 = time.time()
            emb, nodes = fn(G)
            _, f1 = evaluate(emb, nodes, labels, metric=metric)
            rows.append((aname, f1, time.time() - t0))
        best_f1 = max(r[1] for r in rows)
        for aname, f1, dt in rows:
            print(f'  {aname:<12s} {fmt(f1, best_f1)} {dt:>8.2f}')


if __name__ == '__main__':
    main()
