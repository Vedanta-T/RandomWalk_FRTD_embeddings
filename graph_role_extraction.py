import networkx as nx
import numpy as np
import scipy
import pandas as pd
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

from helpers import generate_embedding, combined_embeddings


####### Key functions #########

def find_k_clusters(G = None, embedding=None, HellingerPC = None, n_clusters=2, method='GMM',
                    M=100, normalise_embedding=True, teleport=False, alpha=0.1, directed=False,
                    n_components=10,
                    **kwargs):
    '''
    Function takes in either Networkx Graph or precomputed embeddings and clusters according to a chosen number of clusters and method
    Returns:
        clusters : List of cluster assignments
        roles : Dictionary assigning each node to its cluster assignment
        group_to_nodes : Dictionary assigning each cluster to a list of nodes in it

    These returns are all equivalent, depending on the downstream application either format can be used.
    '''
    
    if embedding is None and HellingerPC is None:
        if G is None:
            raise ValueError('Specify Graph or input precalculated embeddings')
        if not directed:
            embedding = generate_embedding(G=G, M=M, normalise_embedding=normalise_embedding, teleport=teleport, alpha=alpha)
        else:
            embedding = combined_embeddings(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes())), 
                                            M = M, normalise_embedding=normalise_embedding, alpha=alpha)
    if HellingerPC is None:
        HellingerPC = HellingerPCs(embedding, n_components=n_components)

    EC = EmbeddingClustering(HellingerPC)
    if method == 'gmm' or method == 'GMM' or method == 'Gaussian Mixture Model':
        clusters = EC.gmm(n_clusters=n_clusters, **kwargs)
    elif method=='kmeans' or method=='kMeans':
        clusters = EC.kmeans(k=n_clusters, **kwargs)
    elif method == 'BayesianGMM' or  method=='bayesian-gmm' or method=='bgmm':
        clusters = EC.bayesian_gmm(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError('Please input valid clustering method - default is GMM')

    if G is not None:
        keys = list(G.nodes())
    else:
        keys = range(len(HellingerPC))
    roles = {k:v for k, v in zip(keys, clusters)}
    groups = sorted(set(roles.values()))
    group_to_nodes = {g: [n for n in keys if roles[n] == g] for g in groups}
    
    return clusters, roles, group_to_nodes


def find_optimal_clusters(embedding=None, G=None, HellingerPC=None, method='gmm', max_clusters=10, M=100,
                          normalise_embedding=True, teleport=False, alpha=0.1, directed=False,
                          n_components=10, *args, **kwargs):
    '''
    Optional Inputs:
                    embedding : The graph embedding 
                    G : The graph (either this or the above is required)
                    method : either kmeans of gmm (Gaussian Mixture model)
                    max_clusters : Maximum number of clusters to check upto
                    M : Number of steps to calculate embedding to if not provided
                    *args, **kwargs : Arguments for clustering methods

    Returns: Optimal cluster labels, optimal number of clusters, silhouette coefficient for all checked cluster numbers
    '''
    
    if embedding is None and HellingerPC is None:
        if G is None:
            raise ValueError('Specify Graph or input precalculated embeddings')
        if not directed:
            embedding = generate_embedding(G=G, M=M, normalise_embedding=normalise_embedding, teleport=teleport, alpha=alpha)
        else:
            embedding = combined_embeddings(nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes())), 
                                            M = M, normalise_embedding=normalise_embedding, alpha=alpha)
    if HellingerPC is None:
        HellingerPC = HellingerPCs(embedding, n_components=n_components)

    Cluster = EmbeddingClustering(HellingerPC)
    optimal = None ; previous = 0
    scores = []
    for k in range(2, max_clusters + 1):  # Start from 2 since silhouette is undefined for 1 cluster
        if method=='kmeans':
            cluster_labels = Cluster.kMeans(k=k, *args, **kwargs)
            score = silhouette_score(HellingerPC, cluster_labels)
            if score > previous:
                optimal = cluster_labels.copy()
                previous = score  
            scores.append(score)
        elif method == 'gmm' or method == 'GMM' or method == 'Gaussian Mixture Model' or method=='community detection' or method=='bayesian_gmm':
            break
        else:
            raise ValueError('Enter valid clustering method')

    if method == 'gmm' or method == 'GMM' or method == 'Gaussian Mixture Model':
        model = Cluster.gmm_select_model(max_clusters=max_clusters)
        df = pd.DataFrame(model.cv_results_)[
            ["param_n_components", "mean_test_score"]
        ]
        df["mean_test_score"] = -df["mean_test_score"]
        df = df.rename(
            columns={
                "param_n_components": "Number of components",
                "mean_test_score": "BIC score",
            }
        )
        cluster_labels = model.predict(HellingerPC)
        if G is not None:
            keys = list(G.nodes())
        else:
            keys = range(len(HellingerPC))
        roles = {k:v for k, v in zip(keys, cluster_labels)}
        groups = sorted(set(roles.values()))
        group_to_nodes = {g: [n for n in keys if roles[n] == g] for g in groups}
        return cluster_labels, roles, group_to_nodes, model.best_params_, df
    elif method=='bayesian_gmm' or method=='bgmm':
        return Cluster.bayesian_gmm(max_clusters, *args, **kwargs)
        

    if method == 'community detection':
        return Cluster.community_detection(*args, **kwargs)

        
    return optimal, scores.index(max(scores)) + 2, scores  # Optimal number of clusters


########################################################
### Helpers #####

def HellingerPCs(embedding, n_components=10):
    sqrt_embedding = np.sqrt(embedding)
    visualise = VisualiseEmbedding(sqrt_embedding)
    pca = visualise.compute_pca_embedding(n_components=n_components)[0]
    return pca

########################################################

class EmbeddingClustering:
    def __init__(self, embedding):
        self.embedding = embedding
        self.N, self.M = embedding.shape
        
    def kMeans(self, k, *args, **kwargs):
        kmeans = KMeans(n_clusters=k, *args, **kwargs)
        labels = kmeans.fit_predict(self.embedding)
        return labels

    def gmm(self, n_clusters, *args, **kwargs):
        # Step 3: Fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, *args, **kwargs)
        gmm.fit(self.embedding)
    
        # Step 4: Predict cluster labels
        cluster_labels = gmm.predict(self.embedding)
    
        return cluster_labels      

    def gmm_select_model(self, max_clusters, *args, **kwargs):
        def gmm_bic_score(estimator, X):
            """Callable to pass to GridSearchCV that will use the BIC score."""
            # Make it negative since GridSearchCV expects a score to maximize
            return -estimator.bic(X)
        param_grid = {
            "n_components": range(1, max_clusters)
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
        grid_search.fit(self.embedding)

        return grid_search

    def bayesian_gmm(self, n_clusters, *args, **kwargs):
        gmm = BayesianGaussianMixture(n_components=n_clusters, *args, **kwargs)
        cluster_labels = gmm.fit_predict(self.embedding)
    
        return cluster_labels, gmm.get_params()

    def community_detection(self, measure = 'exponential', gamma=10):
        if measure=='exponential':
            S = np.exp(-gamma*0.5*np.abs(self.embedding[:, None, :] - self.embedding[None, :, :]).sum(axis=-1))
        else:
            S = 1 - 0.5*np.abs(self.embedding[:, None, :] - self.embedding[None, :, :]).sum(axis=-1)
        
        S_G = nx.from_numpy_array(S)
        S_G.remove_edges_from(nx.selfloop_edges(S_G))
        partition = community_louvain.best_partition(S_G)
        return partition, S_G

    def find_silhouette_score(self, kmeans_labels=None, k =None, *args, **kwargs):
        if not kmeans_labels:
            if not k:
                raise ValueError('Input k')
            kmeans_labels = self.kMeans(k, *args, **kwargs)

        return silhouette_score(embedding1, kmeans_labels)


########################################################

class VisualiseEmbedding:

    '''
    Class to visualise the FRT distribution embeddings either using tSNE or PCA
    '''
    
    def __init__(self, embedding, G=None, labels=None):
        """
        Initialize the visualizer with node embeddings and optional labels.

        Parameters:
        - embeddings (ndarray): Node embeddings (n x d), where each row corresponds to a node.
        - labels (list or None): Optional list of labels for nodes. Default is None.
        """
        self.node_embeddings = embedding
        self.node_labels = labels
        self.G = G

    def compute_tsne_embedding(self, n_components = 2, perplexity=None, learning_rate=200, random_state=42, metric='manhattan', **kwargs):
        """
        Compute the 2D t-SNE embedding for a given matrix of node embeddings.
    
        Parameters:
        - node_embeddings (array-like): Input matrix where each row corresponds to a node's embedding.
        - perplexity (int, optional): Perplexity parameter for t-SNE. If None, set dynamically as min(30, n_samples // 2).
        - learning_rate (float, optional): Learning rate for t-SNE. Default is 200.
        - random_state (int, optional): Random seed for reproducibility. Default is 42.
    
        Returns:
        - embedding_2d (ndarray): A 2D array with the t-SNE embeddings for each node.
        """
        n_samples = self.node_embeddings.shape[0]
        if perplexity is None:
            perplexity = min(30, n_samples // 2)  # Adjust perplexity dynamically
        
        if perplexity >= n_samples:
            raise ValueError(f"Perplexity ({perplexity}) must be less than the number of samples ({n_samples}).")
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state, metric=metric, **kwargs)
        embedding_2d = tsne.fit_transform(self.node_embeddings)
        return embedding_2d
        
    def compute_pca_embedding(self, n_components=2, instance=False, other_embedding=None):
        """
        Compute the 2D PCA embedding for a given matrix of node embeddings and visualize it.
    
        Parameters:
        - node_embeddings (array-like): Input matrix where each row corresponds to a node's embedding.
        - node_labels (list, optional): Labels for each node to annotate the plot. Default is None (no labels).
    
        Returns:
        - embedding_2d (ndarray): A 2D array with the PCA embeddings for each node.
        """
        # Compute PCA
        pca = PCA(n_components=n_components)
        if instance:
            embedding_2d = instance.transform(other_embedding)
        else:
            instance = pca.fit(self.node_embeddings)
            embedding_2d = instance.transform(self.node_embeddings)
            return embedding_2d, instance
    
        return embedding_2d


    def plot_embedding(self, fig = None, embedding_2d = None, type=None, xlabel='', ylabel='', grid=True, s=40, fontsize=12, color='r'):
        """
        Plot the 2D visualization of the embeddings.

        Parameters: xlabel, ylabel, grid, choice of embedding
        - 
        """
        if not fig:
            fig, ax = plt.subplots()
        else:
            ax = fig.get_axes()[0]
        

        if embedding_2d is not None:
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        elif type is None:
            raise ValueError('Enter valid type')
            return None
        elif type == 'tSNE' or type=='tsne':
            embedding_2d = self.compute_tsne_embedding()
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        elif type == 'PCA' or type=='pca':
            embedding_2d = self.compute_pca_embedding()
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=s, alpha=1, color=color)
            
            if self.node_labels is not None:
                for i, label in enumerate(self.node_labels):
                    ax.annotate(label, (embedding_2d[i, 0], embedding_2d[i, 1]), fontsize=fontsize, alpha=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(grid)
            return fig
        else:
            raise ValueError('Input valid embedding visualisation method or input 2d embedding')
            return None

