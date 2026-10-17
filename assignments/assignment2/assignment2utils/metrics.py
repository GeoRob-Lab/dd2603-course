"""
PyTorch helpers for Parts 3 to 5: SPD matrix functions, geodesics of a metric field, evaluation.

A *metric field* is any callable  metric(x) -> M  with x a tensor (N, 3) of positions and M a
tensor (N, 3, 3) of symmetric positive definite matrices, written with torch operations so that
autograd can differentiate M with respect to x. All tensors are float64.

SPD matrices (batched over leading dimensions)
    sym(A)                      (A + A^T) / 2
    spd_fun(S, f, df)           U f(L) U^T for symmetric S = U L U^T, with a safe gradient
    spd_log(P), spd_sqrtm(P), spd_invsqrtm(P)
    spd_dist(A, B)              affine-invariant distance |log(A^-1/2 B A^-1/2)|_F
    vec_to_sym(v, n)            R^(n(n+1)/2) -> Sym_n, an isometry (off-diagonal entries / sqrt 2)

Geodesics and evaluation
    tensors(paths, idx)         X, V, A of some demos as tensors (D, T, 3)
    christoffel_fd(metric, x)   Christoffel symbols by finite differences (a slow reference)
    geodesic_residual(...)      the per-point geodesic residual of demos under a metric
    geodesic_bvp(metric, x0, x1)  the geodesic between two points (energy minimisation)
    resample(X, n)              a path resampled to n points equally spaced in arc length
    path_error_mm(path, demo)   mean distance between two paths after resampling, in mm
    evaluate(metric, paths, idx)  geodesics between the end points of demos and their path errors
    euclidean(x)                the Euclidean metric field, M(x) = I
"""
import math

import numpy as np
import torch

DTYPE = torch.float64


# ======================================================================= SPD matrices
def sym(A):
    """The symmetric part (A + A^T) / 2 of a batch of square matrices."""
    return 0.5 * (A + A.transpose(-1, -2))


class _SpectralFunction(torch.autograd.Function):
    """
    F(S) = U diag(f(l)) U^T for symmetric S = U diag(l) U^T. The backward pass uses the
    Daleckii-Krein formula with the divided differences (f(l_i) - f(l_j)) / (l_i - l_j),
    replaced by f'(l_i) when l_i = l_j. torch.linalg.eigh alone returns NaN gradients
    when two eigenvalues are equal, for example at the identity.
    """

    @staticmethod
    def forward(ctx, S, f, df):
        lam, U = torch.linalg.eigh(S)
        flam = f(lam)
        ctx.save_for_backward(lam, U, flam)
        ctx.df = df
        return U @ torch.diag_embed(flam) @ U.transpose(-1, -2)

    @staticmethod
    def backward(ctx, G):
        lam, U, flam = ctx.saved_tensors
        dl = lam[..., :, None] - lam[..., None, :]
        dfl = flam[..., :, None] - flam[..., None, :]
        dflam = ctx.df(lam)
        scale = 1.0 + torch.maximum(lam[..., :, None].abs(), lam[..., None, :].abs())
        close = dl.abs() <= 1e-9 * scale
        divided = torch.where(close, 0.5 * (dflam[..., :, None] + dflam[..., None, :]),
                              dfl / torch.where(close, torch.ones_like(dl), dl))
        Ut = U.transpose(-1, -2)
        return U @ (divided * (Ut @ sym(G) @ U)) @ Ut, None, None


def spd_fun(S, f, df):
    """U f(L) U^T for a batch of symmetric matrices S = U L U^T; df is the derivative of f."""
    return _SpectralFunction.apply(S, f, df)


def spd_log(P):
    """Matrix logarithm of SPD matrices."""
    return spd_fun(P, torch.log, lambda l: 1.0 / l)


def spd_sqrtm(P):
    """Matrix square root P^(1/2) of SPD matrices."""
    return spd_fun(P, torch.sqrt, lambda l: 0.5 / torch.sqrt(l))


def spd_invsqrtm(P):
    """P^(-1/2) of SPD matrices."""
    return spd_fun(P, lambda l: l ** -0.5, lambda l: -0.5 * l ** -1.5)


def spd_dist(A, B):
    """Affine-invariant distance between SPD matrices: |log(A^-1/2 B A^-1/2)|_F."""
    Ai = spd_invsqrtm(A)
    return torch.linalg.matrix_norm(spd_log(sym(Ai @ B @ Ai)))


def vec_to_sym(v, n):
    """
    Map vectors (..., n(n+1)/2) to symmetric matrices (..., n, n): the first n entries are the
    diagonal, the others the upper triangle row by row, divided by sqrt(2) so that
    |vec_to_sym(v)|_F = |v|.
    """
    iu = torch.triu_indices(n, n, 1)
    S = torch.diag_embed(v[..., :n])
    off = torch.zeros_like(S)
    off[..., iu[0], iu[1]] = v[..., n:] / math.sqrt(2.0)
    return S + off + off.transpose(-1, -2)


# ======================================================================= metric fields
def euclidean(x):
    """The Euclidean metric field M(x) = I (it still depends on x for autograd, with zero derivative)."""
    return torch.eye(x.shape[-1], dtype=x.dtype) + 0.0 * x[..., :, None] * x[..., None, :]


def tensors(paths, idx=None):
    """Positions, velocities and accelerations of the demos idx, as float64 tensors (D, T, 3)."""
    idx = slice(None) if idx is None else idx
    return tuple(torch.as_tensor(np.asarray(a)[idx], dtype=DTYPE) for a in (paths.x, paths.v, paths.a))


def christoffel_fd(metric, x, h=1e-5):
    """
    Christoffel symbols G[n, k, i, j] = Gamma^k_ij(x_n) of a metric field, with the derivatives of
    M taken by central finite differences of step h. A slow reference for checking; no gradients.
    """
    x = torch.as_tensor(x, dtype=DTYPE)
    d = x.shape[-1]
    with torch.no_grad():
        M = metric(x)
        dM = torch.stack([(metric(x + h * e) - metric(x - h * e)) / (2 * h)
                          for e in torch.eye(d, dtype=DTYPE)], 1)        # dM[n, l, i, j] = d_l M_ij
        S = dM.permute(0, 2, 1, 3) + dM.permute(0, 2, 3, 1) - dM        # [n, l, i, j]: d_i M_lj + d_j M_li - d_l M_ij
        return 0.5 * torch.einsum("nkl,nlij->nkij", torch.linalg.inv(M), S)


def geodesic_residual(metric, X, V, A, christoffel, norm="euclidean"):
    """
    The geodesic residual of curves under a metric field, one value per sample point.

    X, V, A: positions, velocities, accelerations (D, T, 3) of D curves at T times.
    christoffel(metric, x): a function returning the Christoffel symbols (N, 3, 3, 3).
    Returns r (D, T):
        a     = A + Gamma(V, V)                          zero along a geodesic with constant speed
        a_perp = a - (<a, V> / <V, V>) V                 drops the part that only changes the speed
        r     = L^2 |a_perp|^2 / |V|^4                   L = length of the curve
    With norm="euclidean" the inner products, norms and L are Euclidean; with norm="metric" they
    are those of M(x). r is dimensionless and does not change when a curve is re-timed.
    """
    D, T, d = X.shape
    x, v, acc = X.reshape(-1, d), V.reshape(-1, d), A.reshape(-1, d)
    G = christoffel(metric, x)
    a = acc + torch.einsum("nkij,ni,nj->nk", G, v, v)
    if norm == "euclidean":
        Mv, Ma = v, a
    elif norm == "metric":
        M = metric(x)
        Mv, Ma = (M @ v[..., None])[..., 0], (M @ a[..., None])[..., 0]
    else:
        raise ValueError("norm must be 'euclidean' or 'metric'")
    vv = (v * Mv).sum(-1)
    coef = (a * Mv).sum(-1) / vv
    a_perp_sq = (a * Ma).sum(-1) - 2 * coef * (a * Mv).sum(-1) + coef ** 2 * vv
    L = vv.sqrt().reshape(D, T).mean(1, keepdim=True)       # t runs over [0, 1]
    return L ** 2 * a_perp_sq.reshape(D, T) / vv.reshape(D, T) ** 2


def geodesic_bvp(metric, x0, x1, n=64, iters=300):
    """
    The geodesic from x0 to x1 (positions (3,)), as n points: minimises the discrete energy
    (n - 1) * sum_i dx_i^T M(midpoint_i) dx_i over the n - 2 inner points with L-BFGS,
    starting from the straight line. Returns a NumPy array (n, 3).
    """
    x0, x1 = torch.as_tensor(np.asarray(x0), dtype=DTYPE), torch.as_tensor(np.asarray(x1), dtype=DTYPE)
    t = torch.linspace(0, 1, n, dtype=DTYPE)[1:-1, None]
    inner = (x0 + t * (x1 - x0)).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([inner], lr=1, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)

    def energy():
        opt.zero_grad()
        P = torch.cat([x0[None], inner, x1[None]])
        dP = P[1:] - P[:-1]
        E = (n - 1) * (dP[:, None, :] @ metric(0.5 * (P[1:] + P[:-1])) @ dP[:, :, None]).sum()
        E.backward()
        return E

    opt.step(energy)
    with torch.no_grad():
        return torch.cat([x0[None], inner, x1[None]]).numpy()


def resample(X, n=100):
    """The path X (T, 3) resampled to n points equally spaced in arc length."""
    X = np.asarray(X, dtype=float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))]
    u = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(u, s, X[:, j]) for j in range(X.shape[1])], 1)


def path_error_mm(path, demo, n=100):
    """Mean distance between two paths after resampling both to n points by arc length, in mm."""
    return 1000.0 * float(np.linalg.norm(resample(path, n) - resample(demo, n), axis=1).mean())


def evaluate(metric, paths, idx, n=64):
    """
    Geodesics of `metric` between the first and last point of the demos idx (positions in metres),
    and their path errors against the demos.

    Returns a dict with
        geodesics  list of (n, 3) arrays
        error_mm   (len(idx),) path error of each geodesic
        line_mm    (len(idx),) path error of the straight line (the Euclidean geodesic)
    """
    idx = np.atleast_1d(np.arange(len(paths.x))[idx])
    geods, err, line = [], [], []
    for i in idx:
        demo = paths.x[i]
        g = geodesic_bvp(metric, demo[0], demo[-1], n=n)
        geods.append(g)
        err.append(path_error_mm(g, demo))
        line.append(path_error_mm(np.stack([demo[0], demo[-1]]), demo))
    return {"geodesics": geods, "error_mm": np.array(err), "line_mm": np.array(line)}
