# RandomWalk_FRTD_embeddings
Repository accompanying [paper link] 

## File description:

1. helpers.py : Contains the base first return time distribution (FRTD) embedding functions along with other helpers
2. graph_role_extraction.py : Contains functions for role extraction from graphs, equivalent to clustering embeddings
3. graph_alignment.py : Contains functions for graph alignment using FRTDs alone and FUGAL-FRTD method.
4. graph_randomisation.py : Containts functions for graph randomization, i.e. MCMC sampling from the FRTD distance ensemble using parallel tempering

The jupyter notebook files contain examples and code to reproduce the results in the paper and demonstrate how to use the method, however these require upstream data sources
mentioned in the respective notebooks. The LOTR network data is provided.

The FUGAL_FRT folder relies on the reference code originally written for the FUGAL algorithm [1, 2] modified to also include the FUGAL-FRTD algorithm, this can be called more easily from the graph_alignment.py wrapper. 


## Requirements:

The FRTD computation function requires only numpy and scipy, however the more advanced optional applications rely on networkx, matplotlib, sklearn, tqdm, ot, and pathos.


## References:

[1] A. Bommakanti, H. R. Vonteri, K. Skitsas, S. Ranu, D. Mottin, and P. Karras, FUGAL:
Feature-fortified unrestricted graph alignment, in The Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024, https://openreview.net/forum?id=SdLOs1FR4h.
[2] https://github.com/idea-iitd/Fugal
