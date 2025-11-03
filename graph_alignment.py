import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import scipy
from scipy.optimize import linear_sum_assignment
import time

from helpers import generate_embedding, compute_distance_matrix, combined_embeddings

import sys
custom_path = 'FUGAL-FRT/helpers'
sys.path.append(custom_path)
from pred import predict_alignment


#### Key functions #####

def FRTD_alignment(graph1, graph2, M=100, teleport=False, alpha=0.1, normalise_embedding=True, directed=False, **kwargs):
    t1 = -time.time()
    if not directed:
        embedding1 = generate_embedding(G=graph1, M = M, normalise_embedding=normalise_embedding, teleport=teleport, alpha=alpha)[:, 2:]
        embedding2 = generate_embedding(G=graph2, M = M,  normalise_embedding=normalise_embedding, teleport=teleport, alpha=alpha)[:, 2:]
    else:
        embedding1 = combined_embeddings(nx.to_scipy_sparse_array(graph1, nodelist=range(graph1.number_of_nodes())),
                                         M=M, alpha=alpha, normalise_embedding=normalise_embedding)
        embedding2 = combined_embeddings(nx.to_scipy_sparse_array(graph2, nodelist=range(graph2.number_of_nodes())),
                                         M=M, alpha=alpha, normalise_embedding=normalise_embedding)

    row, col, _ = hungarian_matching(embedding1, embedding2, metric='TVD')
    t1 += time.time()
    return row, col, t1


def FUGAL_alignment(graph1, graph2, mu=1, use_normalised_adjacency=False, sink_method='sinkhorn'):
    t1 = -time.time()
    rowcol, _ = predict_alignment([graph1], [graph2], use_normalised_adjacency=use_normalised_adjacency, mu=mu, sink_method=sink_method)
    row = [] ; col = []
    for i in range(len(rowcol[0])):
        row.append(rowcol[0][i][0])
        col.append(rowcol[0][i][1])
    t1 += time.time()
    
    return row, col, t1

def FUGAL_FRT_alignment(graph1, graph2, mu=1, use_normalised_adjacency=False, sink_method='sinkhorn',
                       M=100, teleport=False, alpha=0.1, normalise_embedding=True,
                       metric='TVD', 
                       **kwargs):
    t1 = -time.time()
    rowcol, times = predict_alignment([graph1], [graph2], use_frts=True, M=M, mu=mu, normalise_embedding=normalise_embedding,
                                      teleport=teleport, alpha=alpha, metric=metric, use_normalised_adjacency=use_normalised_adjacency,
                                      sink_method=sink_method)
    row = [] ; col = []
    for i in range(len(rowcol[0])):
        row.append(rowcol[0][i][0])
        col.append(rowcol[0][i][1])
    t1 += time.time()
    
    return row, col, t1

def ApproximateQA_alignment(graph1, graph2, use_normalised_adjacency=False, sink_method='sinkhorn'):
    t1 = -time.time()
    rowcol, times = predict_alignment([graph1], [graph2], no_embedding=True, mu=0, sink_method=sink_method)
    row = [] ; col = []
    for j in range(len(rowcol[0])):
        row.append(rowcol[0][j][0])
        col.append(rowcol[0][j][1])
    t1+=time.time()
    return row, col, t1


### Helper ####

def hungarian_matching(embeddings1, embeddings2, metric='TVD'):
    """
    Matches node embeddings using the Hungarian algorithm.

    Parameters:
    - embeddings1 (ndarray): Embedding matrix for graph 1 (n x d).
    - embeddings2 (ndarray): Embedding matrix for graph 2 (m x d).
    - metric (str): Distance metric for computing the cost matrix. Options: 'euclidean', 'manhattan'.

    Returns:
    - row_ind (ndarray): Row indices of matched nodes from graph 1.
    - col_ind (ndarray): Column indices of matched nodes from graph 2.
    - cost_matrix (ndarray): Cost matrix used for matching.
    """
    # Validate input dimensions
    assert embeddings1.shape[1] == embeddings2.shape[1], "Embeddings must have the same dimensionality."

    # Compute the cost matrix
    if metric == 'euclidean':
        cost_matrix = compute_distance_matrix(embeddings1, embeddings2, use_second=True, metric='L2')
    elif metric == 'TVD' or metric=='manhattan':
        cost_matrix = compute_distance_matrix(embeddings1, embeddings2, use_second=True, metric='TVD')
    elif metric == 'cosine':
        cost_matrix = compute_distance_matrix(embeddings1, embeddings2, use_second=True, metric='cosine')
    elif metric == 'hellinger':
        cost_matrix =  compute_distance_matrix(embeddings1, embeddings2, use_second=True, metric='hellinger')
    elif metric == 'JSD':
        cost_matrix =  compute_distance_matrix(embeddings1, embeddings2, use_second=True, metric='JSD')   
        cost_matrix = np.where(np.isnan(cost_matrix) == True, 0, cost_matrix)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Solve the assignment problem using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind, cost_matrix
    
