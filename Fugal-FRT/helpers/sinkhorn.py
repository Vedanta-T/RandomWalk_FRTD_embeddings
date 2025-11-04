# -*- coding: utf-8 -*-
"""
Rewrite ot.bregman.sinkhorn in Python Optimal Transport (https://pythonot.github.io/_modules/ot/bregman.html#sinkhorn)
using pytorch operations.
Bregman projections for regularized OT (Sinkhorn distance).
"""

import torch
import time

M_EPS = 1e-16


def sinkhorn(a, b, C, reg=1e-1, method='sinkhorn', maxIter=1000, tau=1e3,
             stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    method : str
        method used for the solver either 'sinkhorn', 'greenkhorn', 'sinkhorn_stabilized' or
        'sinkhorn_epsilon_scaling', see those function for specific parameters
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    if method.lower() == 'sinkhorn':
        return sinkhorn_knopp(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    elif method.lower() == 'sinkhorn_safe':
        return sinkhorn_knopp_safe(a, b, C, reg, maxIter=maxIter,
                              stopThr=stopThr, verbose=verbose, log=log,
                              warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                              **kwargs)
    elif method.lower() == 'sinkhorn_stabilized':
        return sinkhorn_stabilized(a, b, C, reg, maxIter=maxIter, tau=tau,
                                   stopThr=stopThr, verbose=verbose, log=log,
                                   warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                   **kwargs)
    elif method.lower() == 'sinkhorn_epsilon_scaling':
        return sinkhorn_epsilon_scaling(a, b, C, reg,
                                        maxIter=maxIter, maxInnerIter=100, tau=tau,
                                        scaling_base=0.75, scaling_coef=None, stopThr=stopThr,
                                        verbose=False, log=log, warm_start=warm_start, eval_freq=eval_freq,
                                        print_freq=print_freq, **kwargs)
    elif method.lower() == 'sinkhorn_logdomain':
        return sinkhorn_logdomain(a, b, C, reg=reg, maxIter=maxIter, stopThr=stopThr, eval_freq=eval_freq, dtype=torch.float64)
    else:
        raise ValueError("Unknown method '%s'." % method)


def sinkhorn_knopp(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                   verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization optimal transport
    The input should be PyTorch tensors
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1].

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        u = warm_start['u']
        v = warm_start['v']
    else:
        u = torch.ones(na, dtype=a.dtype).to(device) / na
        v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    torch.div(C, -reg, out=K)
    torch.exp(K, out=K)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)

    # t1 = time.time()
    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        if torch.any(torch.isnan(u)) or torch.any(torch.isnan(v)) or \
                torch.any(torch.isinf(u)) or torch.any(torch.isinf(v)):
            print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            # below is equivalent to:
            # b_hat = torch.sum(u.reshape(-1, 1) * K * v.reshape(1, -1), 0)
            # but with more memory efficient
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            # err = (b - b_hat).abs().sum().item()

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1
    # t2 = time.time()
    # print("Sinkhorn loop: ", t2 - t1)

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = reg * torch.log(u + M_EPS)
        log['beta'] = reg * torch.log(v + M_EPS)

    # transport plan
    P = u.reshape(-1, 1) * K * v.reshape(1, -1)
    if log:
        return P, log
    else:
        return P


def sinkhorn_stabilized(a, b, C, reg=1e-1, maxIter=1000, tau=1e3, stopThr=1e-9,
                        verbose=False, log=False, warm_start=None, eval_freq=10, print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix scaling algorithm as proposed in [1]
    but with the log stabilization proposed in [3] an defined in [2] (Algo 3.1)

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    device = a.device
    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    if log:
        log = {'err': []}

    if warm_start is not None:
        alpha = warm_start['alpha']
        beta = warm_start['beta']
    else:
        alpha = torch.zeros(na, dtype=a.dtype).to(device)
        beta = torch.zeros(nb, dtype=b.dtype).to(device)

    u = torch.ones(na, dtype=a.dtype).to(device) / na
    v = torch.ones(nb, dtype=b.dtype).to(device) / nb

    def update_K(alpha, beta):
        """log space computation"""
        """memory efficient"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=K)
        torch.add(K, -C, out=K)
        torch.div(K, reg, out=K)
        torch.exp(K, out=K)

    def update_P(alpha, beta, u, v, ab_updated=False):
        """log space P (gamma) computation"""
        torch.add(alpha.reshape(-1, 1), beta.reshape(1, -1), out=P)
        torch.add(P, -C, out=P)
        torch.div(P, reg, out=P)
        if not ab_updated:
            torch.add(P, torch.log(u + M_EPS).reshape(-1, 1), out=P)
            torch.add(P, torch.log(v + M_EPS).reshape(1, -1), out=P)
        torch.exp(P, out=P)

    K = torch.empty(C.shape, dtype=C.dtype).to(device)
    update_K(alpha, beta)

    b_hat = torch.empty(b.shape, dtype=C.dtype).to(device)

    it = 1
    err = 1
    ab_updated = False

    # allocate memory beforehand
    KTu = torch.empty(v.shape, dtype=v.dtype).to(device)
    Kv = torch.empty(u.shape, dtype=u.dtype).to(device)
    P = torch.empty(C.shape, dtype=C.dtype).to(device)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)
        v = torch.div(b, KTu + M_EPS)
        torch.matmul(K, v, out=Kv)
        u = torch.div(a, Kv + M_EPS)

        ab_updated = False
        # remove numerical problems and store them in K
        if u.abs().sum() > tau or v.abs().sum() > tau:
            alpha += reg * torch.log(u + M_EPS)
            beta += reg * torch.log(v + M_EPS)
            u.fill_(1. / na)
            v.fill_(1. / nb)
            update_K(alpha, beta)
            ab_updated = True

        if log and it % eval_freq == 0:
            # we can speed up the process by checking for the error only all
            # the eval_freq iterations
            update_P(alpha, beta, u, v, ab_updated)
            b_hat = torch.sum(P, 0)
            err = (b - b_hat).pow(2).sum().item()
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['u'] = u
        log['v'] = v
        log['alpha'] = alpha + reg * torch.log(u + M_EPS)
        log['beta'] = beta + reg * torch.log(v + M_EPS)

    # transport plan
    alpha_tilde = alpha + reg * torch.log(u.clamp_min(M_EPS))
    beta_tilde  = beta  + reg * torch.log(v.clamp_min(M_EPS))
    P = torch.exp((alpha_tilde[:, None] + beta_tilde[None, :] - C) / reg)
    #update_P(alpha, beta, u, v, False)

    if log:
        return P, log
    else:
        return P


def sinkhorn_epsilon_scaling(a, b, C, reg=1e-1, maxIter=100, maxInnerIter=100, tau=1e3, scaling_base=0.75,
                             scaling_coef=None, stopThr=1e-9, verbose=False, log=False, warm_start=None, eval_freq=10,
                             print_freq=200, **kwargs):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:

    .. math::
        \gamma = arg\min_\gamma <\gamma,C>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - C is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are target and source measures (sum to 1)

    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [1] but with the log stabilization
    proposed in [3] and the log scaling proposed in [2] algorithm 3.2

    Parameters
    ----------
    a : torch.tensor (na,)
        samples measure in the target domain
    b : torch.tensor (nb,)
        samples in the source domain
    C : torch.tensor (na,nb)
        loss matrix
    reg : float
        Regularization term > 0
    tau : float
        thershold for max value in u or v for log scaling
    maxIter : int, optional
        Max number of iterations
    stopThr : float, optional
        Stop threshol on error ( > 0 )
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    gamma : (na x nb) torch.tensor
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters

    References
    ----------
    [1] M. Cuturi, Sinkhorn Distances : Lightspeed Computation of Optimal Transport, Advances in Neural Information Processing Systems (NIPS) 26, 2013
    [2] Bernhard Schmitzer. Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems. SIAM Journal on Scientific Computing, 2019
    [3] Chizat, L., Peyré, G., Schmitzer, B., & Vialard, F. X. (2016). Scaling algorithms for unbalanced transport problems. arXiv preprint arXiv:1607.05816.

    See Also
    --------

    """

    na, nb = C.shape

    assert na >= 1 and nb >= 1, 'C needs to be 2d'
    assert na == a.shape[0] and nb == b.shape[0], "Shape of a or b does't match that of C"
    assert reg > 0, 'reg should be greater than 0'
    assert a.min() >= 0. and b.min() >= 0., 'Elements in a or b less than 0'

    def get_reg(it, reg, pre_reg):
        if it == 1:
            return scaling_coef
        else:
            if (pre_reg - reg) * scaling_base < M_EPS:
                return reg
            else:
                return (pre_reg - reg) * scaling_base + reg

    if scaling_coef is None:
        scaling_coef = C.max() + reg

    it = 1
    err = 1
    running_reg = scaling_coef

    if log:
        log = {'err': []}

    warm_start = None

    while (err > stopThr and it <= maxIter):
        running_reg = get_reg(it, reg, running_reg)
        P, _log = sinkhorn_stabilized(a, b, C, running_reg, maxIter=maxInnerIter, tau=tau,
                                      stopThr=stopThr, verbose=False, log=True,
                                      warm_start=warm_start, eval_freq=eval_freq, print_freq=print_freq,
                                      **kwargs)

        warm_start = {}
        warm_start['alpha'] = _log['alpha']
        warm_start['beta'] = _log['beta']

        primal_val = (C * P).sum() + reg * (P * torch.log(P)).sum() - reg * P.sum()
        dual_val = (_log['alpha'] * a).sum() + (_log['beta'] * b).sum() - reg * P.sum()
        err = primal_val - dual_val
        if log:
            log['err'].append(err)

        if verbose and it % print_freq == 0:
            print('iteration {:5d}, constraint error {:5e}'.format(it, err))

        it += 1

    if log:
        log['alpha'] = _log['alpha']
        log['beta'] = _log['beta']
        return P, log
    else:
        return P

def sinkhorn_logdomain(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9, eval_freq=10, dtype=torch.float64):
    """
    Stable Sinkhorn in pure log-domain:
      alpha <- reg * (log a - logsumexp((beta[None,:] - C)/reg, dim=1))
      beta  <- reg * (log b - logsumexp((alpha[:,None] - C)/reg, dim=0))
    Returns transport plan P.
    """
    device = C.device
    a = a.to(dtype); b = b.to(dtype); C = C.to(dtype)
    eps = torch.finfo(dtype).eps

    # ensure strictly positive and normalized
    a = (a + eps); a = a / a.sum()
    b = (b + eps); b = b / b.sum()

    na, nb = C.shape
    alpha = torch.zeros(na, dtype=dtype, device=device)
    beta  = torch.zeros(nb, dtype=dtype, device=device)

    def lse_row(beta):
        # logsumexp over columns j
        return torch.logsumexp((beta[None, :] - C) / reg, dim=1)

    def lse_col(alpha):
        # logsumexp over rows i
        return torch.logsumexp((alpha[:, None] - C) / reg, dim=0)

    err = float('inf')
    for it in range(1, maxIter + 1):
        # update alpha
        alpha = reg * (torch.log(a) - lse_row(beta))
        # recenter (shift invariance): keep numbers moderate
        m = alpha.mean()
        alpha = alpha - m
        beta  = beta + m

        # update beta
        beta = reg * (torch.log(b) - lse_col(alpha))

        if (it % eval_freq) == 0 or it == maxIter:
            # compute b_hat = sum_j P_ij over i using log-domain
            # log P = (alpha[:,None] + beta[None,:] - C)/reg
            logP = (alpha[:, None] + beta[None, :] - C) / reg
            b_hat = torch.exp(torch.logsumexp(logP, dim=0))
            err = torch.sum((b - b_hat) ** 2).item()
            if err < stopThr:
                break

    # final plan
    logP = (alpha[:, None] + beta[None, :] - C) / reg
    P = torch.exp(logP)
    return P


def sinkhorn_knopp_safe(a, b, C, reg=1e-1, maxIter=1000, stopThr=1e-9,
                         verbose=False, log=False, warm_start=None,
                         eval_freq=10, print_freq=200, M_EPS=0.0):
    # keep dtype/device exactly as given
    device, dtype = C.device, C.dtype
    na, nb = C.shape
    assert na == a.shape[0] and nb == b.shape[0]
    assert reg > 0 and a.min() >= 0 and b.min() >= 0

    if log: hist = {'err': []}

    if warm_start is not None:
        u = warm_start['u'].to(device=device, dtype=dtype)
        v = warm_start['v'].to(device=device, dtype=dtype)
    else:
        u = torch.full((na,), 1.0/na, dtype=dtype, device=device)
        v = torch.full((nb,), 1.0/nb, dtype=dtype, device=device)

    # exact same kernel construction
    K = torch.exp(-C / reg)

    it, err = 1, float('inf')
    # prealloc like yours
    KTu = torch.empty_like(v)
    Kv  = torch.empty_like(u)

    while (err > stopThr and it <= maxIter):
        upre, vpre = u, v
        torch.matmul(u, K, out=KTu)                 # u^T K
        v = b / (KTu + M_EPS)
        torch.matmul(K, v, out=Kv)                  # K v
        u = a / (Kv + M_EPS)

        if torch.isnan(u).any() or torch.isnan(v).any() or torch.isinf(u).any() or torch.isinf(v).any():
            if verbose:
                print('Warning: numerical errors at iteration', it)
            u, v = upre, vpre
            break

        if (it % eval_freq) == 0:
            # identical error to your code: check columns only
            b_hat = torch.matmul(u, K) * v
            err = (b - b_hat).pow(2).sum().item()
            if log: hist['err'].append(err)
            if verbose and (it % print_freq) == 0:
                print(f'iteration {it:5d}, constraint error {err:5e}')
        it += 1

    #P = u[:, None] * K * v[None, :]
    tiny = torch.finfo(C.dtype).tiny  # e.g., ~1e-308 for float64
    alpha = reg * torch.log(u.clamp_min(tiny))
    beta  = reg * torch.log(v.clamp_min(tiny))
    
    # recentre (alpha+beta shift invariance keeps exponents moderate)
    m = alpha.mean()
    alpha = alpha - m
    beta  = beta + m
    
    Z = (alpha[:, None] + beta[None, :] - C) / reg
    P = torch.exp(Z)  # numerically stable, no 0*Inf
    if log:
        hist.update({'u': u, 'v': v,
                     'alpha': reg * torch.log(u + 1e-300),
                     'beta':  reg * torch.log(v + 1e-300)})
        return P, hist
    return P

