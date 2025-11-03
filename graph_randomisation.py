import networkx as nx
import numpy as np
import random
from tqdm import tqdm
import pathos
import scipy

from helpers import generate_embedding, combined_embeddings


L1_distance = lambda x,y : np.sum(np.abs(x-y))

class MCMC_Chain():
    '''
    MCMC chain class which initalises a chain with a start edgelist and a given target embedding, to obtain randomised samples.
    Requires:
        current_edgelist : Edgelist to initialise chain at
        beta : Inverse temperature
        target_embedding : Embedding of target network
        normalised : Whether embedding should be normalised
        alpha : Teleportation probability (None if no teleportation)
        directed : Whether we are working with directed networks
        combine_embeddings : If embeddings for A, A.T should be combined (should be true for directed networks)
        store_history : If chain history should be stored
    '''

    def __init__(self, current_edgelist, beta, target_embedding, normalised=True, alpha=None, directed=False, combine_embeddings=False,
                store_history=False):
        if not combine_embeddings:
            self.N, self.K = target_embedding.shape[0], target_embedding.shape[1]-normalised
        else:
            self.N, self.K = target_embedding.shape[0], int((target_embedding.shape[1]-normalised)/2)
        self.beta = beta
        self.target_embedding = target_embedding
        self.normalised=normalised
        self.directed=directed
        self.store_history = store_history
        if alpha is None:
            self.frtd = lambda A : generate_embedding(A=A, M=self.K, normalise_embedding=self.normalised, teleport=False)
        else:
            if combine_embeddings:
                self.frtd = lambda A : combined_embeddings(A, M=self.K, normalise_embedding=self.normalised, alpha=alpha)
            else:
                self.frtd = lambda A : generate_embedding(A=A, M=self.K, normalise_embedding=self.normalised, teleport=True, alpha=alpha)
        
        self.set_attributes(current_edgelist, self.directed)

    def set_attributes(self, current_edgelist, directed):
        self.edges = np.array(current_edgelist)
        self.A = scipy.sparse.dok_matrix((self.N,self.N), dtype=bool)
        self.A[self.edges[:,0],self.edges[:,1]] = 1
        if not directed:
            self.A[self.edges[:,1],self.edges[:,0]] = 1
        self.current_embedding = self.frtd(self.A)
        self.similarity = L1_distance(self.target_embedding, self.current_embedding)
        if self.store_history:
            self.history = [self.similarity]
        self.acceptance_count = 0

    #Edge move proposal steps
    def edge_move_step(self):
        u = np.random.randint(self.edges.shape[0])
        i,j = self.edges[u]
        if not self.directed and (self.A[i].sum() == 1 or self.A[j].sum() == 1):
            return False
            
        a,b = np.random.randint(self.N, size=2)
        while (a==b) or self.A[a, b] == 1:
            a,b = np.random.randint(self.N, size=2)

        if not self.directed:
            self.A[i,j] = self.A[j,i] = 0
            self.A[a,b] = self.A[b,a] = 1
            self.edges[u] = a, b
        else:
            self.A[i,j] = 0
            self.A[a,b] = 1
            self.edges[u] = a, b

        new_embedding = self.frtd(self.A)
        new_similarity = L1_distance(self.target_embedding, new_embedding)

        if np.log(np.random.random()) < self.beta * (self.similarity - new_similarity):
            self.current_embedding = new_embedding
            self.similarity = new_similarity
            if self.store_history:
                self.history.append(self.similarity)
            self.new_history.append(self.similarity)
            self.acceptance_count +=1
            return True
        else:
            if not self.directed:
                self.A[a,b] = self.A[b,a] = 0
                self.A[i,j] = self.A[j,i] = 1
                self.edges[u] = i, j
            else:
                self.A[i,j] = 1
                self.A[a,b] = 0
                self.edges[u] = i, j
                
            if self.store_history:
                self.history.append(self.similarity)
            self.new_history.append(self.similarity)
            return False

    # Degree preserving proposal steps
    def directed_edge_swap_step(self):
        '''
        Swap (i-->j, a-->b) to (i-->b, a-->j)
        '''
        i,j,a,b = 0,0,0,0
        while len(np.unique([i,j,a,b])) < 4 or (self.A[i, b]==1) or (self.A[a, j]==1):
            u = np.random.randint(self.edges.shape[0])
            i,j = self.edges[u]
            v = np.random.randint(self.edges.shape[0])
            a,b = self.edges[v]
            
        self.A[i,j] = 0
        self.A[a,b] = 0
        self.A[i,b] = 1
        self.A[a,j] = 1
        
        self.edges[u] = i, b
        self.edges[v] = a, j

        return True, i, j, a, b, u, v

    def directed_triangle_reverse_step(self):
        '''
        Swap (i-->j, j-->k, k-->i) to (i-->k, k-->j, j-->i)
        '''
        for _ in range(100):
            u = np.random.choice(len(self.edges))
            i, j = self.edges[u]
            if self.A[j, i]:
                continue
            # Get all outgoing from j
            j_out = self.A[j,:].nonzero()[1]
            for k in j_out:
                if self.A[k,i] and not self.A[i, k] and not self.A[k, j]:  # Found triangle i->j->k->i
                    # Reverse all three edges
                    self.A[i,j] = self.A[j,k] = self.A[k,i] = 0
                    self.A[j,i] = self.A[k,j] = self.A[i,k] = 1
                    # Update edge list
                    self.edges[u] = j, i
                    for v in range(len(self.edges)):
                        if self.edges[v][0] == j and self.edges[v][1] == k:
                            v1 = v
                            self.edges[v] = (k,j)  
                        if self.edges[v][0] == k and self.edges[v][1] == i:
                            v2 = v
                            self.edges[v] = (i,k)
                        else:
                            continue
                    return True, i, j, k, u, v1, v2

        return False, i, j, None, u, None, None

    def directed_combined_step(self):
        i,j,a,b = 0,0,0,0
        while len(np.unique([i,j,a,b])) < 3 or (self.A[i, b]==1) or (self.A[a, j]==1):
            u = np.random.randint(self.edges.shape[0])
            i,j = self.edges[u]
            v = np.random.randint(self.edges.shape[0])
            a,b = self.edges[v]
            
        if i==b and self.A[j, a] and not self.A[i, a] and not self.A[j, i]:
            self.A[a, i] = self.A[i, j] = self.A[j, a] = 0
            self.A[i, a] = self.A[a, j] = self.A[j, i] = 1
            self.edges[u] = j, i ; self.edges[v] = b, a
            for v1 in range(len(self.edges)):
                if self.edges[v1][0] == j and self.edges[v1][1] == a:
                    self.edges[v1] = a, j
                    return True, i, j, a, b, u, v, v1
                else: continue
        elif a==j and self.A[b, i] and not self.A[b, j] and not self.A[j, i]:
            self.A[i, j] = self.A[j, b] = self.A[b, i] = 0
            self.A[j, i] = self.A[b, j] = self.A[i, b] = 1
            self.edges[u] = j, i ; self.edges[v] = b, a
            for v1 in range(len(self.edges)):
                if self.edges[v1][0] == b and self.edges[v1][1] == i:
                    self.edges[v1] = i, b
                    return True, i, j, a, b, u, v, v1 
                else: continue
        elif len(np.unique([i,j,a,b])) == 4:
            self.A[i,j] = 0
            self.A[a,b] = 0
            self.A[i,b] = 1
            self.A[a,j] = 1
            
            self.edges[u] = i, b
            self.edges[v] = a, j
    
            return True, i, j, a, b, u, v, None
        else: return self.directed_combined_step()
                    

    def degree_preserving_step(self):
        if self.directed:
            swapped, i, j, a, b, u, v, v1 = self.directed_combined_step()
            if v1 is None:
                new_embedding = self.frtd(self.A)
                new_similarity = L1_distance(self.target_embedding, new_embedding)
        
                if np.log(np.random.random()) < self.beta * (self.similarity - new_similarity):
                    self.current_embedding = new_embedding
                    self.similarity = new_similarity
                    if self.store_history:
                        self.history.append(self.similarity)
                    self.new_history.append(self.similarity)
                    self.acceptance_count +=1
                    return True
                else:
                    self.A[i,j] = 1
                    self.A[a,b] = 1
                    self.A[i,b] = 0
                    self.A[a,j] = 0
                    
                    self.edges[u] = i, j
                    self.edges[v] = a, b
                    if self.store_history:
                        self.history.append(self.similarity)
                    self.new_history.append(self.similarity)
                    return False
            else:
                new_embedding = self.frtd(self.A)
                new_similarity = L1_distance(self.target_embedding, new_embedding)
        
                if np.log(np.random.random()) < self.beta * (self.similarity - new_similarity):
                    self.current_embedding = new_embedding
                    self.similarity = new_similarity
                    if self.store_history:
                        self.history.append(self.similarity)
                    self.new_history.append(self.similarity)
                    self.acceptance_count +=1
                    return True
                else:
                    if i==b : k=a
                    elif j==a : k=b
                    self.A[j, i] = self.A[k, j] = self.A[i, k] =  0
                    self.A[i,j] = self.A[j,k] = self.A[k,i] = 1
                    self.edges[u] = i, j
                    self.edges[v] = j, k
                    self.edges[v1] = k, i
                    if self.store_history:
                        self.history.append(self.similarity)
                    self.new_history.append(self.similarity)
                    return False
        else:
            i,j,a,b = 0,0,0,0 ; flag = True
            while len(np.unique([i,j,a,b])) < 4 or flag:
                u = np.random.randint(self.edges.shape[0])
                i,j = self.edges[u]
                v = np.random.randint(self.edges.shape[0])
                a,b = self.edges[v]
                if (self.A[i,b]==1) or (self.A[a,j]==1):
                    if self.A[i,a]==1 or self.A[j,b]==1:
                        flag=True
                    else:
                        flag=False
                        cross_rewire=False
                else:
                    flag=False
                    cross_rewire=True

            if cross_rewire:
                self.A[i,j] = self.A[j,i] = 0
                self.A[a,b] = self.A[b,a] = 0
                self.A[i,b] = self.A[b,i] = 1
                self.A[j,a] = self.A[a,j] = 1
                self.edges[u] = i, b
                self.edges[v] = a, j
            else:
                self.A[i,j] = self.A[j,i] = 0
                self.A[a,b] = self.A[b,a] = 0
                self.A[i,a] = self.A[a,i] = 1
                self.A[j,b] = self.A[b,j] = 1
                self.edges[u] = i, a
                self.edges[v] = j, b
    
            new_embedding = self.frtd(self.A)
            new_similarity = L1_distance(self.target_embedding, new_embedding)
    
            if np.log(np.random.random()) < self.beta * (self.similarity - new_similarity):
                self.current_embedding = new_embedding
                self.similarity = new_similarity
                if self.store_history:
                    self.history.append(self.similarity)
                self.new_history.append(self.similarity)
                self.acceptance_count +=1
                return True
            else:
                if cross_rewire:
                    self.A[i,b] = self.A[b,i] = 0
                    self.A[j,a] = self.A[a,j] = 0
                    self.A[i,j] = self.A[j,i] = 1
                    self.A[a,b] = self.A[b,a] = 1
            
                    self.edges[u] = i, j
                    self.edges[v] = a, b
                else:
                    self.A[i,a] = self.A[a,i] = 0
                    self.A[j,b] = self.A[b,j] = 0
                    self.A[i,j] = self.A[j,i] = 1
                    self.A[a,b] = self.A[b,a] = 1
                    self.edges[u] = i, j
                    self.edges[v] = a, b
                if self.store_history:
                    self.history.append(self.similarity)
                self.new_history.append(self.similarity)
                return False

    def N_steps(self, N, p=0):
        '''
        Run N steps of the chain
        Set p=0 for degree preserving steps, and p=1 for random edge moves
        '''
        acc = 0
        self.new_history = []
        for _ in range(N):
            if np.random.random() < p:
                acc += self.edge_move_step()
            else:
                acc += self.degree_preserving_step()
        return acc / N


#Temperature swaps for parallel tempering
def temp_swaps(N, states):
    acc = 0
    for _ in range(N):
        s,t = np.random.randint(len(states), size=2)
        H1, H2 = states[s].similarity, states[t].similarity
        b1, b2 = states[s].beta, states[t].beta
        if np.log(np.random.random()) < (H1 - H2) * (b1 - b2):
            states[s].beta = b2
            states[t].beta = b1
            acc += 1
    return acc


##### Run parallel tempering #####
def run_parallel_tempering(target_graph, #Provide target graph (networkx graph)
                  starting_graph=None, #Provide starting graph (if None then start from tarrget)
                  n_steps =100000, #Total number of steps
                  burn_in=50000,  #Burn in period (sample after this)
                  K = 40, #Depth of embedding
                  directed=False, #If networks are directed
                  alpha=None, #Teleportation probability (if None then no teleportation)
                  normalise_embedding=True, #If embeddings should be normalised
                  operation='degree-preserving', #Choice of operation (move-edge or degree-preserving)
                  betas=None, #List of temperatures for replicas (length of list corresponds to number of replicas)
                  n_states=40, #If betas list is not provided then number of replicas
                  beta_min = 0, #Minimum temp (only needed if betas is not provided)
                  beta_max=40, #Max temp (only needed if betas is not provided)
                  swap_steps=1000, #Number of temperature swap attempts
                  swap_interval = 500, #Interval at which to attempt swaps (sampling is also done at this interval)
                  print_outputs=False, #Print minimum energy at each swap interval
                  combine_embeddings=False, #Combine embeddings for directed graphs
                  sample=True #If graphs other than the minimum should be sampled (sampling is done at the swap interval to ensure mixing between samples)
                  ):

    '''
    Function which runs parallel tempering to obtain randomised graph samples at a range of temperatures and perform minimisation

    Returns:
        MCMC state objects
        minimum_edgelists : Edgelist corresponding to minimum energy at each sampling point
        min_diffs : List of minimum energy at each iteration
        diffs_by_temp : Dictionary of energies by inverse temperature over the iterations
        betas : Inverse temperature schedule
        Optional : samples - Sampled graphs at various temperatures post burn in
    '''

    if not directed:
        target_embedding = generate_embedding(G=target_graph, M=K, normalise_embedding=normalise_embedding)
    else:
        if combine_embeddings:
            A_target = nx.to_scipy_sparse_array(target_graph, nodelist=range(target_graph.number_of_nodes()))
            target_embedding = combined_embeddings(A_target, M=K, normalise_embedding=normalise_embedding, alpha=alpha)
        else:
            target_embedding = generate_embedding(G=target_graph, M=K, normalise_embedding=normalise_embedding, teleport=True, alpha=alpha)

    if starting_graph is None:
        starting_graph = target_graph.copy()
    if betas is None:
        betas = np.linspace(beta_min, beta_max, n_states)
        
    states = []
    for b in betas:
        states.append(MCMC_Chain(current_edgelist = list(starting_graph.edges), beta = b, target_embedding =target_embedding, normalised=normalise_embedding, alpha=alpha, directed=directed, combine_embeddings=combine_embeddings))

    samples = {beta: [] for beta in betas} if sample else None
    min_diffs = []
    minimum_edgelists = []
    diffs_by_temp = {beta: [] for beta in betas}

    def run_N_steps(state, p):
        state.N_steps(swap_interval, p)
        return state

    run_move_edges = lambda state: run_N_steps(state, 1)
    run_degree_preserving = lambda state: run_N_steps(state, 0.0)
    np.seterr(all='ignore')
    pool = pathos.pools.ProcessPool() 
    step_number = 0
    print(np.amin([states[j].similarity for j in range(len(states))]))
    n_its = int(n_steps/swap_interval)
    for i in tqdm(range(n_its)):
        if operation=='move-edge':
            states = pool.map(run_move_edges, states, chunksize=1)
        elif operation=='degree-preserving':
            states = pool.map(run_degree_preserving, states, chunksize=1)

        step_number += swap_interval

        if step_number >= burn_in:
            print('Collecting samples...')
            minimum_state = np.argmin([states[j].similarity for j in range(len(states))])
            minimum_edgelists.append(states[minimum_state].edges)
            if sample:
                for state in states:
                    samples[state.beta].append(state.edges)
                

        min_diffs.append(np.amin([states[j].similarity for j in range(len(states))])) 
        
        for state in states:
            diffs_by_temp[state.beta] = diffs_by_temp[state.beta] + state.new_history
            state.new_history = []

        if print_outputs:
            print(np.amin([states[j].similarity for j in range(len(states))]))
        
        
        temp_swaps(swap_steps, states)
    
            
    if sample:
        return states, minimum_edgelists, min_diffs, diffs_by_temp, samples, betas
    return states, minimum_edgelists, min_diffs, diffs_by_temp, betas



##### Entropy computation ######

def compute_entropy_mean(sample_energies, S0=0):
    r'''
    Compute entropy using mean energies:
    S(\beta) = S(0) + \beta <U(\beta)> - \int_0^\beta <U(\beta')> d\beta'
    '''
    betas = sorted(sample_energies.keys())
    #var_E = {b: np.var(sample_energies[b]) for b in betas}
    mean_E = {b: np.mean(sample_energies[b]) for b in betas}
    S = {betas[0]: S0}
    integral=betas[0]*mean_E[betas[0]]
    for i in range(1, len(betas)):
        b0, b1 = betas[i-1], betas[i]
        #trap = 0.5 * (b1*var_E[b1] + b0*var_E[b0]) * (b1 - b0)
        trap = 0.5 * (mean_E[b1] + mean_E[b0]) * (b1 - b0)
        integral += trap
        #S[b1] = S[b0] - trap
        S[b1] = S0 + b1 * mean_E[b1] - integral
    
    return S


def compute_entropy_variance(sample_energies, S0=0):
    r'''
    Compute entropy using energy variances:
    S(\beta) = S(0) - \int_0^\beta \beta' Var[U(\beta')] d\beta'
    '''
    betas = sorted(sample_energies.keys())
    var_E = {b: np.var(sample_energies[b]) for b in betas}
    #mean_E = {b: np.mean(sample_energies[b]) for b in betas}
    S = {0.0: S0}
    #integral=0
    for i in range(1, len(betas)):
        b0, b1 = betas[i-1], betas[i]
        trap = 0.5 * (b1*var_E[b1] + b0*var_E[b0]) * (b1 - b0)
        #trap = 0.5 * (mean_E[b1] + mean_E[b0]) * (b1 - b0)
        #integral += trap
        S[b1] = S[b0] - trap
        #S[b1] = b1 * mean_E[b1] - integral
    
    return S

def compute_entropy_free_energy(sample_energies, S0=0):
    r'''
    Compute entropy using finite differences
    S(\beta) = S(0) + \beta <U(\beta)> - \sum_{i=1}^N log <exp(U(x) \beta_i - \beta_{i-11})>_{\beta_i}
    '''
    betas = sorted(sample_energies.keys())
    mean_E = {b: np.mean(sample_energies[b]) for b in betas}
    S = {0.0: S0} ; term = 0
    for i in range(1, len(betas)):
        b0, b1 = betas[i-1], betas[i]
        prob = np.mean(np.exp(np.where(np.array(sample_energies[b1])*(b1-b0) > 700, 700, np.array(sample_energies[b1])*(b1-b0))))
        term += np.log(prob)
        S[b1] = b1 * mean_E[b1] - term + S0
    
    return S


def infinite_temp_entropy(degrees):
    '''
    Estimate infinite temperature for an undirected graph
    Reference:
    McKay and N. C. Wormald, Asymptotic enumeration by degree sequence of graphs of high degree,644 
    European Journal of Combinatorics, 11 (1990), pp. 565–5
    '''
    
    from scipy.special import gammaln
    degrees = np.array(degrees, dtype=int)
    E = np.sum(degrees) // 2
    
    # lambda term
    lamda = np.sum(degrees * (degrees - 1)) / (4 * E)
    
    # log of factorial terms (using gammaln for stability)
    log_num = gammaln(2*E + 1)  # log((2E)!)
    log_den = gammaln(E + 1) + E*np.log(2) + np.sum(gammaln(degrees + 1))
    
    # log of asymptotic count
    log_G = log_num - log_den - lamda - lamda**2
    
    return log_G   # this is the entropy

def infinite_temp_entropy_directed(in_degrees, out_degrees):
    '''
    Estimate infinite temperature for a directed graph
    Reference:
    A. Liebenau, N. Wormald, Asymptotic enumeration of digraphs and bipartite graphs by degree sequence, Random Struct. Algorithms. (2023), 62 (2023), 259–286.
    '''
    
    in_degrees = np.array(in_degrees, dtype=int)
    out_degrees = np.array(out_degrees, dtype=int)
    m = np.sum(in_degrees)
    n = len(in_degrees)

    # log of normalisation
    log_normalisation = log_nCr(n * (n - 1), m)

    # sum of log binomials for out-degrees
    log_term1 = sum(log_nCr(n - 1, d) for d in out_degrees)

    # sum of log binomials for in-degrees
    log_term2 = sum(log_nCr(n - 1, d) for d in in_degrees)

    # covariance structure
    cov = np.cov(out_degrees, in_degrees)
    sigma_s = cov[0, 0]
    sigma_t = cov[1, 1]
    sigma_st = cov[0, 1]

    s = np.mean(out_degrees)
    t = np.mean(in_degrees)
    mu = m / (n * (n - 1))

    H = -0.5 * (1 - sigma_s / (s * (1 - mu))) * (1 - sigma_t / (t * (1 - mu))) - sigma_st / (s * (1 - mu))

    # stay in log domain as long as possible
    log_G = -log_normalisation + log_term1 + log_term2 + H

    return log_G  # this is already log(G), i.e. the entropy


def compute_specific_heat_mean(sample_energies):
    betas = sorted(sample_energies.keys())
    mean_E = {b: np.mean(sample_energies[b]) for b in betas}
    C = {}
    for i in range(len(betas)):
        if 0 < i < len(betas)-1:
            b1, b0 = betas[i+1], betas[i-1]
            dE_db = (mean_E[b1] - mean_E[b0]) / (b1 - b0)   # central diff
        elif i == 0:
            b1, b0 = betas[i], betas[i+1]
            dE_db = (mean_E[b1] - mean_E[b0]) / (b1 - b0)           # forward diff
        else:
            b1, b0 = betas[i], betas[i-1]
            dE_db = (mean_E[b1] - mean_E[b0]) / (b1 - b0)        # backward diff
        C[betas[i]] = -(betas[i]**2) * dE_db
    return C

def compute_specific_heat_variance(sample_energies):
    """
    Compute specific heat C(beta) = beta^2 * Var(E)_beta
    """
    C = {}
    for beta, energies in sample_energies.items():
        C[beta] = (beta**2) * np.var(energies)
    return C