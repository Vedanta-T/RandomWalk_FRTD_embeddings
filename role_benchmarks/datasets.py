"""Role-labelled graph datasets for the benchmark.

Each loader returns (G, labels) where G is an undirected networkx graph
and labels is a {node: int} dict. Files are downloaded into ./data/ on
first use.

Upstream data sources referenced in https://arxiv.org/abs/2512.02694. Please cite the upstream sources if you use these datasets
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import shutil
import urllib.request
import zipfile
from collections import Counter
import pickle
import numpy as np
import networkx as nx


CACHE = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(CACHE, exist_ok=True)


def _download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


def _download_gz(url, gz_path, txt_path):
    if os.path.exists(txt_path):
        return
    if not os.path.exists(gz_path):
        urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, 'rb') as f_in, open(txt_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def synthetic_pa_vs_dd(N=512, m=5, alpha=0.5, q=None, seed=42):
    """Growing network where each new node is one of two mechanisms:
      - PA (preferential attachment): connects to m existing nodes with prob ∝ degree
      - DD (duplication-divergence): picks a uniform random parent w, adds
        an anchor edge to w, then copies each of w's other neighbours with
        retention probability q

    Role = which mechanism (0 = PA, 1 = DD).

    When q is None it defaults to (m-1)/(2m), which (with alpha=0.5) matches
    the expected mean degree across the two roles. Discriminating signal:
    DD nodes inherit a parent's neighbourhood → triangle-rich; PA nodes pick
    spread-out high-degree targets → triangle-poor.
    """
    if q is None:
        q = (m - 1) / (2 * m)
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    seed_size = m + 1                                  # clique seed
    for i in range(seed_size):
        for j in range(i + 1, seed_size):
            G.add_edge(i, j)
    labels = {i: 0 for i in range(seed_size)}

    for t in range(seed_size, N):
        existing = np.array(list(G.nodes()))
        if rng.random() < alpha:
            degs = np.array([G.degree(v) for v in existing], dtype=float)
            chosen = rng.choice(existing, size=min(m, len(existing)),
                                replace=False, p=degs / degs.sum())
            G.add_node(t)
            for c in chosen:
                G.add_edge(t, int(c))
            labels[t] = 0
        else:
            parent = int(rng.choice(existing))
            G.add_node(t)
            G.add_edge(t, parent)                       # anchor edge
            for nb in list(G.neighbors(parent)):
                if nb != t and rng.random() < q:
                    G.add_edge(t, nb)
            labels[t] = 1
    return G, labels


def synthetic_rgg_vs_er(n1=256, n2=256, k=8, seed=42):
    """Mix of n1 random-geometric-graph nodes and n2 Erdős–Rényi nodes,
    parameterised by target mean degree `k`. Both populations have the
    same expected degree, so degree can't distinguish them — the only
    discriminating signal is higher-order structure (clustering, triangles).

    Returns (G, labels) with label 0 = RGG node, 1 = ER node.
    """
    N = n1 + n2
    p = k / (N - 1)
    r = float(np.sqrt(k / (np.pi * (N - 1))))
    G = nx.random_geometric_graph(n1, r, seed=seed)
    rng = np.random.default_rng(seed)
    er = range(n1, N)
    G.add_nodes_from(er)
    for i in er:
        for j in range(N):
            if j <= i and j >= n1:
                continue  # avoid double-counting within ER
            if rng.random() < p:
                G.add_edge(i, j)
    return G, {v: (0 if v < n1 else 1) for v in G.nodes()}

def sbm(n=512, c_in=8, c_out=2, seed=42):
    """Stochastic block model with two equal-sized communities, where
    each node has expected degree c_in + c_out. Role = community label.
    """
    rng = np.random.default_rng(seed)
    sizes = [n // 2, n - n // 2]
    p_in = c_in / (sizes[0] - 1)
    p_out = c_out / sizes[1]
    G = nx.stochastic_block_model(sizes, [[p_in, p_out], [p_out, p_in]], seed=seed)
    old_labels = {
        v: 0 if v < sizes[0] else 1
        for v in G.nodes()
    }
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    # Relabel remaining nodes consecutively
    mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
    G = nx.relabel_nodes(G, mapping)
    # Update labels using the mapping
    labels = {
        mapping[old]: old_labels[old]
        for old in mapping
    }
    if len(isolates) > 0:
        print(f"Found {len(isolates)} isolated nodes.")
    return G, labels





def airports(country='brazil'):
    assert country in {'brazil', 'europe', 'usa'}
    base = 'https://raw.githubusercontent.com/leoribeiro/struc2vec/master/graph'
    e = os.path.join(CACHE, f'{country}-airports.edgelist')
    l = os.path.join(CACHE, f'labels-{country}-airports.txt')
    _download(f'{base}/{country}-airports.edgelist', e)
    _download(f'{base}/labels-{country}-airports.txt', l)
    G = nx.read_edgelist(e, nodetype=int)
    labels = {}
    with open(l) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels[int(parts[0])] = int(parts[1])
    G = G.subgraph([v for v in G.nodes() if v in labels]).copy()
    return G, labels


def webkb(school='cornell', min_class_size=2):
    assert school in {'cornell', 'texas', 'wisconsin'}
    base = (f'https://raw.githubusercontent.com/graphdml-uiuc-jlu/'
            f'geom-gcn/master/new_data/{school}')
    e = os.path.join(CACHE, f'webkb-{school}-edges.txt')
    l = os.path.join(CACHE, f'webkb-{school}-labels.txt')
    _download(f'{base}/out1_graph_edges.txt', e)
    _download(f'{base}/out1_node_feature_label.txt', l)
    G = nx.Graph()
    with open(e) as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) == 2:
                G.add_edge(int(p[0]), int(p[1]))
    labels = {}
    with open(l) as f:
        next(f)
        for line in f:
            p = line.split('\t')
            if len(p) >= 3:
                labels[int(p[0])] = int(p[2])
    counts = Counter(labels.values())
    keep = {c for c, n in counts.items() if n >= min_class_size}
    labels = {v: c for v, c in labels.items() if c in keep and v in G.nodes()}
    G = G.subgraph(labels.keys()).copy()
    return G, labels


def _skewed_csv(net_name, sub_name=None):
    sub = sub_name or net_name
    z = os.path.join(CACHE, f'skewed-{sub}.csv.zip')
    out = os.path.join(CACHE, f'skewed-{sub}')
    if not os.path.isdir(out):
        if not os.path.exists(z):
            url = f'https://networks.skewed.de/net/{net_name}/files/{sub}.csv.zip'
            urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    return os.path.join(out, 'nodes.csv'), os.path.join(out, 'edges.csv')


def adjnoun():
    """Newman's word-adjacency network: nouns (1) vs adjectives (0)."""
    nodes_csv, edges_csv = _skewed_csv('adjnoun')
    labels = {}
    with open(nodes_csv) as f:
        next(f)
        for row in csv.reader(f):
            if len(row) >= 3:
                labels[int(row[0])] = int(row[2])
    G = nx.Graph()
    with open(edges_csv) as f:
        next(f)
        for row in csv.reader(f):
            if len(row) >= 2:
                G.add_edge(int(row[0]), int(row[1]))
    return G, labels


def celegans_neurons():
    """C. elegans 2019 hermaphrodite chemical synapse network, cell-type labels."""
    nodes_csv, edges_csv = _skewed_csv('celegans_2019', 'hermaphrodite_chemical')
    labels = {}
    type_to_id = {}
    with open(nodes_csv) as f:
        next(f)
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            t = row[1].strip()
            type_to_id.setdefault(t, len(type_to_id))
            labels[int(row[0])] = type_to_id[t]
    G = nx.Graph()
    with open(edges_csv) as f:
        next(f)
        for row in csv.reader(f):
            if len(row) >= 2:
                G.add_edge(int(row[0]), int(row[1]))
    return G, labels



def ATN():
    """OpenFlights airport network (3188 nodes) with airport-type labels.

    Label 3 is an 'other' category, not a meaningful role, so its
    147 nodes are removed; the benchmark runs on the remaining 3041 nodes
    (3 classes), 28 of which become isolated after the removal.
    """
    e = os.path.join(CACHE, 'ATN.edgelist')
    l = os.path.join(CACHE, 'ATN_labels.pkl')

    G = nx.read_edgelist(e, nodetype=int)

    with open(l, 'rb') as file:
        labels = pickle.load(file)

    remove_nodes = [v for v, lab in labels.items() if lab == 3]

    G.remove_nodes_from(remove_nodes)

    labels = {
        v: lab
        for v, lab in labels.items()
        if lab != 3
    }

    return G, labels
    
