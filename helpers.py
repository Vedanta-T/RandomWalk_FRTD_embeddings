import networkx as nx
import numpy as np
import scipy
from tqdm import tqdm


##### Key functions ######

def generate_embedding(G=None, #Networkx graph
                       A=None, #Sparse adjacency matrix
                       M=100,  #Max number of steps
                       progress_bar=False, #Print progress bar
                       normalise_embedding=False, #Normalise embedding by appending remaining probability mass to last step
                       teleport=False, #Teleport to consider directed networks
                       alpha = 0.1, #Teleportation probability 
                       **kwargs):
    '''
    Requires : Graph or Adjacency matrix
    Optional Inputs:
                    M : Maximum number of steps
                    progress_bar : Flag which prints the calculation progress_bar (default is true)

    Returns : NxM matrix containing the node embeddings
    '''
    if A is None:
        if G is None:
            raise ValueError('Please provide either networkx graph or adjacency matrix.')
        A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()))
    matrix = first_return_dist(A, K=M, teleport=teleport, alpha=alpha, progress_bar=progress_bar)
    if normalise_embedding:
        matrix = np.hstack([matrix, 1-np.sum(matrix, axis=1)[:, np.newaxis]])


    return matrix


# Combine embeddings for A, A.T when working with directed networks
def combined_embeddings(A, M=40, alpha=0.1, normalise_embedding=True):
    outward = generate_embedding(A=A, M=M, normalise_embedding=normalise_embedding, teleport=True, alpha=alpha)
    inward = generate_embedding(A=A.T, M=M, normalise_embedding=normalise_embedding, teleport=True, alpha=alpha)
    return np.concatenate((outward, inward), axis=1)

### Base functions ####

def first_return_dist(A, K=40, teleport=False, alpha = 0.1, progress_bar=False):
    p = np.eye(A.shape[0], dtype=np.float64)
    X = np.zeros((K, A.shape[0]))
    if not teleport:
        W = scipy.sparse.csr_matrix(A / A.sum(axis=0))
        if progress_bar:
            for t in tqdm(range(1, K)):
                p = W.dot(p)
                X[t] = p.diagonal()
                np.fill_diagonal(p, 0.0)
        else:
            for t in range(1, K):
                p = W.dot(p)
                X[t] = p.diagonal()
                np.fill_diagonal(p, 0.0)
    else:
        W = (1-alpha)*scipy.sparse.csr_matrix(A / A.sum(axis=0))
        if progress_bar:
            for t in tqdm(range(1, K)):
                p = W.dot(p) + p.sum(axis=0)* (alpha / A.shape[0])
                X[t] = p.diagonal()
                np.fill_diagonal(p, 0.0)
        else:
            for t in range(1, K):
                p = W.dot(p) + p.sum(axis=0)* (alpha / A.shape[0])
                X[t] = p.diagonal()
                np.fill_diagonal(p, 0.0)
    return X.T



#### Additional helpers #####

def compute_mean_std(G):
    '''
    Function which takes in a networkx graph and returns the exact mean and standard deviation of FRTDs (using the analytical expression)
    
    Requires : Networkx Graph

    Returns : Nx2 matrix containing the mean and std of the FRT distributions
    '''
    N  = len(G.nodes)
    embedding = np.zeros((N, 2))        
    A = np.identity(N)-nx.normalized_laplacian_matrix(G).todense()
    vals, vecs = np.linalg.eigh(A)
    embedding[:, 0] = 1/vecs[:, -1]**2
    S = np.zeros(N)
    for i in range(N-1):
        S += vecs[:, i]**2 * 1/(1-vals[i])
    embedding[:, 1] = np.sqrt(2*embedding[:, 0]**2*S + embedding[:, 0] - embedding[:, 0]**2)

    return embedding


#Permutation invariant distance measures

def earth_movers_distance(embedding1, embedding2, weights1=None, weights2=None):
    import ot
    """
    Compute the Earth Mover's Distance (EMD) between two graph embeddings.

    Parameters:
    - embedding1 (ndarray): Embedding of the first graph (m x d).
    - embedding2 (ndarray): Embedding of the second graph (n x d).
    - weights1 (ndarray): Optional weights for the first graph's nodes (m,). Default is uniform weights.
    - weights2 (ndarray): Optional weights for the second graph's nodes (n,). Default is uniform weights.

    Returns:
    - emd (float): Earth Mover's Distance between the two embeddings.
    """
    # Number of points in each embedding
    m, n = embedding1.shape[0], embedding2.shape[0]
    
    # Assign uniform weights if none are provided
    if weights1 is None:
        weights1 = np.ones(m) / m
    if weights2 is None:
        weights2 = np.ones(n) / n
    
    # Compute the cost matrix (pairwise distances between points in embedding1 and embedding2)
    cost_matrix = np.linalg.norm(embedding1[:, None, :] - embedding2[None, :, :], axis=2)
    
    # Solve the optimal transport problem
    emd = ot.emd2(weights1, weights2, cost_matrix)
    return emd


def hausdorff_distance(embeddings_A, embeddings_B, metric='cityblock'):
    from scipy.spatial.distance import cdist
    """
    Compute the Hausdorff distance between two sets of embeddings.

    Parameters:
    - embeddings_A (ndarray): Node embeddings of the first graph (n x d).
    - embeddings_B (ndarray): Node embeddings of the second graph (m x d).
    - metric (str): Distance metric to use ('l1', 'l2', etc.).

    Returns:
    - float: Hausdorff distance between the two embedding sets.
    """
    # Compute pairwise distances
    pairwise_dist = cdist(embeddings_A, embeddings_B, metric=metric)

    # Compute directed distances
    d_AB = np.max(np.min(pairwise_dist, axis=1))  # A -> B
    d_BA = np.max(np.min(pairwise_dist, axis=0))  # B -> A

    # Hausdorff distance
    return max(d_AB, d_BA)



def compute_distance_matrix(embedding, embedding2=None, use_second=False, metric='TVD', alpha=0, beta=0, G_0=None):
    import scipy.sparse as sp
    import scipy.spatial.distance as sp_distance
    from scipy.spatial.distance import cdist
    
    N=len(embedding)
    if not use_second:
        if metric == 'l1' or metric=='L1' or metric=='manhattan' or metric=='TVD':
            return 0.5*sp_distance.cdist(embedding, embedding, metric='cityblock')
        if metric == 'l2' or metric=='L2' or metric=='euclidean':
            embedding_sq = np.sum(embedding**2, axis=1)
            return np.sqrt(embedding_sq[:, None] - 2 * embedding @ embedding.T + embedding_sq[None, :])
        if metric == 'cosine':
            return cdist(embedding, embedding, metric="cosine")
        if metric == 'hellinger' or metric=='Hellinger':
            dist = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(i+1, N):
                    dist[i, j] = np.sqrt(np.abs(1-np.sum(np.sqrt(np.abs(embedding[i]*embedding[j])))))
            return dist + dist.T
        if metric=='JSD' or metric=='jensen-shannon' or metric=='Jensen Shannon':
            dist = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(i+1, N):
                    dist[i, j] = JSD(embedding[i], embedding[j])
            return dist + dist.T
    else:
        if metric == 'l1' or metric=='L1' or metric=='manhattan' or metric=='TVD':
            return 0.5*sp_distance.cdist(embedding, embedding2, metric='cityblock')
        if metric == 'l2' or metric=='L2' or metric=='euclidean':
            return np.sqrt(np.sum(embedding**2, axis=1)[:, None] - 2 * embedding @ embedding2.T + np.sum(embedding2**2, axis=1)[None, :])
        if metric == 'cosine':
            return cdist(embedding, embedding2, metric="cosine")
        if metric == 'hellinger' or metric=='Hellinger':
            dist = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    dist[i, j] = np.sqrt(np.abs(1-np.sum(np.sqrt(np.abs(embedding[i]*embedding2[j])))))
            return dist
        if metric=='JSD' or metric=='jensen-shannon' or metric=='Jensen Shannon':
            from scipy.spatial.distance import jensenshannon as JSD
            dist = np.zeros((N, N), dtype=float)
            for i in range(N):
                for j in range(N):
                    dist[i, j] = JSD(embedding[i], embedding2[j])
            return dist








