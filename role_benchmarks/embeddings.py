"""Five role / structural embeddings for benchmarking.

Each function returns (emb, nodes), where emb is a 2-D NumPy array of shape
(n_nodes, dim) and `nodes` is the corresponding node list.

  - degree(G)     :    trivial baseline; emb[i] = [deg(node_i)]
  - rolx(G)       :    Henderson et al. 2012, NMF on ReFeX features (k=4)
  - graphwave(G)  :    Donnat et al. 2018, paper defaults (n_samples=50)
  - struc2vec(G)  :    Ribeiro et al. 2017, via shenweichen/GraphEmbedding
  - frtd(G)       :    first-return-time distribution (absorbing-walker form)
"""
from __future__ import annotations

import contextlib
import io
import os
import tempfile
import warnings

import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.decomposition import NMF

from ge.models.struc2vec import Struc2Vec as _Struc2Vec
import ge


import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
)

sys.path.insert(0, PROJECT_ROOT)

from helpers import generate_embedding


# --- degree (trivial baseline) ----------------------------------------------

def degree(G):
    nodes = list(G.nodes())
    emb = np.array([[G.degree(v)] for v in nodes], dtype=float)
    return emb, nodes


# --- ReFeX (helper for RolX) ------------------------------------------------

def _base_features(G):
    nodes = list(G.nodes())
    feats = np.zeros((len(nodes), 3))
    for i, v in enumerate(nodes):
        nbrs = set(G.neighbors(v))
        ego = nbrs | {v}
        internal = sum(1 for u in ego for w in G.neighbors(u) if w in ego) // 2
        external = sum(1 for u in ego for w in G.neighbors(u) if w not in ego)
        feats[i] = [len(nbrs), internal, external]
    return feats, nodes


def _prune(X, threshold=0.95):
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        redundant = False
        for k in keep:
            c = np.corrcoef(col, X[:, k])[0, 1]
            if np.isfinite(c) and abs(c) > threshold:
                redundant = True
                break
        if not redundant:
            keep.append(j)
    return X[:, keep] if keep else X[:, :1]


def _refex(G, n_iter=3):
    X, nodes = _base_features(G)
    X = _prune(X)
    idx = {v: i for i, v in enumerate(nodes)}
    for _ in range(n_iter):
        n, d = X.shape
        sums = np.zeros((n, d))
        means = np.zeros((n, d))
        for i, v in enumerate(nodes):
            nbr = [idx[u] for u in G.neighbors(v)]
            if not nbr:
                continue
            block = X[nbr]
            sums[i] = block.sum(axis=0)
            means[i] = block.mean(axis=0)
        X = _prune(np.hstack([X, sums, means]))
    return X, nodes


# --- RolX (Henderson et al. 2012) -------------------------------------------

def rolx(G, n_roles=4, n_iter=10):
    """NMF on ReFeX features. Fixed k = n_roles."""
    X, nodes = _refex(G, n_iter=n_iter)
    X_pos = np.log1p(np.clip(X, 0, None))
    model = NMF(n_components=n_roles, init=None, max_iter=2000, random_state=0)
    return model.fit_transform(X_pos), nodes


# --- GraphWave (Donnat et al. 2018, paper defaults) -------------------------

def graphwave(G, scales=(1.0, 5.0), n_samples=50, t_max=100.0):
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)
    n = A.shape[0]
    deg = A.sum(axis=1)
    safe = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(safe), 0.0)
    L = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L = (L + L.T) / 2
    lam, U = np.linalg.eigh(L)
    ts = np.linspace(0, t_max, n_samples)

    parts = []
    for s in scales:
        Psi = (U * np.exp(-s * lam)[None, :]) @ U.T
        phi_real = np.empty((n, n_samples))
        phi_imag = np.empty((n, n_samples))
        for k, t in enumerate(ts):
            z = np.exp(1j * t * Psi).mean(axis=0)
            phi_real[:, k] = z.real
            phi_imag[:, k] = z.imag
        parts.extend([phi_real, phi_imag])
    return np.hstack(parts), nodes


# --- struc2vec (Ribeiro et al. 2017, via shenweichen/GraphEmbedding) --------

def struc2vec(G, dimensions=128, walk_length=40, num_walks=10, window=10,
              k_max=None):
    nodes = list(G.nodes())
    with tempfile.TemporaryDirectory() as tmp, \
         contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()), \
         warnings.catch_warnings(), \
         np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        temp_path = os.path.join(tmp, 's2v') + os.sep
        model = _Struc2Vec(graph=G, walk_length=walk_length,
                           num_walks=num_walks,
                           opt3_num_layers=k_max, temp_path=temp_path, 
                           reuse=False)
        # ge's walker emits int node IDs; gensim>=4 wants strings.
        model.sentences = [[str(t) for t in walk] for walk in model.sentences]
        model.train(embed_size=dimensions, window_size=window,
                    workers=1, iter=5)
        wv = model.w2v_model.wv
        emb = np.array([wv[str(v)] for v in nodes])
        #sg = ge.SkipGramModel(
        #rw.totalNodes, rw.embedDim
        #)
    
        #rw.learnNodeEmbedding(sg)
    
        #emb = np.array([
        #    rw.getNodeEmbedding(v)
        #    for v in nodes
        #])

    return emb, nodes


def frtd(G, t_max=200, include_tail=True):
    return generate_embedding(G=G, #Networkx graph
                       A=None, #Sparse adjacency matrix
                       M=t_max,  #Max number of steps
                       progress_bar=False, #Print progress bar
                       normalise_embedding=include_tail, #Normalise embedding by appending remaining probability mass to last step
                       teleport=False, #Teleport to consider directed networks
                       alpha = 0.1, #Teleportation probability
                       nodelist=list(G.nodes)
                             ), list(G.nodes)