# RandomWalk_FRTD_embeddings
Repository accompanying https://arxiv.org/abs/2512.02694

## File description:

1. helpers.py : Contains the base first return time distribution (FRTD) embedding functions along with other helpers
2. graph_role_extraction.py : Contains functions for role extraction from graphs, equivalent to clustering embeddings
3. graph_alignment.py : Contains functions for graph alignment using FRTDs alone and FUGAL-FRTD method.
4. graph_randomisation.py : Containts functions for graph randomization, i.e. MCMC sampling from the FRTD distance ensemble using parallel tempering

The jupyter notebook files contain examples and code to reproduce the results in the paper and demonstrate how to use the method, however these require upstream data sources mentioned in the respective notebook. An example notebook is included for role extraction for the lord of the rings inter character network for which the data is provided.

The role benchmarks folder contains the scripts to generate the role extraction results in the paper the description is as follows:
1. datasets.py : Obtains the various real network datasets and labels from upstream sources. Please cite the upstream sources (described in the paper) if you use this data in your work.
2. embeddings.py : Uses the methods provided by the GraphEmbedding repository (https://github.com/shenweichen/GraphEmbedding/blob/master/README.md) to construct the baselines.
3. benchmark.py : Runs the role-extraction benchmark on the 8 real datasets (Table 1): 5-NN macro-F1 under 5-fold stratified cross validation, averaged over 10 cross-validation splits and 5 seeded struc2vec runs, reporting mean (std), embedding time and how often each method is best. FRTD and RolX are compared with the total variation distance. The cross-validation splits are random by default.
4. benchmark_synthetic.py : Runs a benchmark on the synthetic network models described in the paper. 

The FUGAL_FRT folder relies on the reference code originally written for the FUGAL algorithm [1, 2] modified to also include the FUGAL-FRTD algorithm, this can be called more easily from the graph_alignment.py wrapper. 


## Requirements:

The FRTD computation function requires only numpy and scipy, however the more advanced optional applications rely on networkx, matplotlib, sklearn, tqdm, ot, and pathos. The comparison scripts have additional dependencies for the GraphEmbedding repository to implement struc2vec.


## References:

[1] A. Bommakanti, H. R. Vonteri, K. Skitsas, S. Ranu, D. Mottin, and P. Karras, FUGAL:
Feature-fortified unrestricted graph alignment, in The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024, https://openreview.net/forum?id=SdLOs1FR4h.
[2] https://github.com/idea-iitd/Fugal
