"""
Public checks for Programming Assignment 2. Run them as often as you like; they never change your objects.

    assignment2utils.checks.part1(Sphere, PoseManifold, frechet_mean)
    assignment2utils.checks.part2(Sphere, UnitQuaternions, WrappedGaussian, quat_body_basis)
    assignment2utils.checks.part3(ConformalMetric)
    assignment2utils.checks.part4(christoffel, airm_rgrad, airm_exp, rgd_step)
    assignment2utils.checks.part5(bimap, reeig, SPDMetricNet, airm_exp)

Each line reads  [PASS] / [FAIL] / [TODO] / [ERROR]  name  (largest error vs tolerance).
TODO means the method still raises NotImplementedError.
"""
import traceback

import numpy as np

from . import rotations as rot

_STATUSES = ("PASS", "FAIL", "TODO", "ERROR")


class _Report:
    def __init__(self, title):
        self.title = title
        self.rows = []

    def check(self, name, fn, tol=None):
        """
        fn() returns either a bool or a non-negative error. With a tolerance, the
        check passes if the error is <= tol.
        """
        try:
            out = fn()
            if tol is None:
                status, info = ("PASS" if bool(out) else "FAIL"), ""
            else:
                err = float(out)
                ok = np.isfinite(err) and err <= tol
                status, info = ("PASS" if ok else "FAIL"), f"max err {err:.1e} (tol {tol:.0e})"
        except NotImplementedError:
            status, info = "TODO", "not implemented yet"
        except Exception as e:  # noqa: BLE001
            tb = traceback.extract_tb(e.__traceback__)[-1]
            status, info = "ERROR", f"{type(e).__name__}: {e} (line {tb.lineno} in {tb.name})"
        self.rows.append((status, name, info))

    def show(self):
        print(f"── {self.title} " + "─" * max(0, 60 - len(self.title)))
        for status, name, info in self.rows:
            print(f"  [{status:5s}] {name:55s} {info}")
        counts = {s: sum(r[0] == s for r in self.rows) for s in _STATUSES}
        summary = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        print(f"  => {summary}")


def _err(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _rng(seed=0):
    return np.random.default_rng(seed)


def _sphere_pairs(S, n, rng, max_angle=None, min_angle=0.0):
    """n pairs (p, q) on S^d with angle(p, q) in [min_angle, max_angle]."""
    d = S.n + 1
    p = rng.normal(size=(n, d))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    u = rng.normal(size=(n, d))
    u -= np.sum(u * p, -1, keepdims=True) * p
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    max_angle = np.pi if max_angle is None else max_angle
    th = rng.uniform(min_angle, max_angle, size=(n, 1))
    q = np.cos(th) * p + np.sin(th) * u
    return p, q, u, th[:, 0]


# ======================================================================= Part 1
def part1(Sphere, PoseManifold, frechet_mean, seed=0):
    from . import oracle

    rng = _rng(seed)
    r = _Report("Part 1: frechet_mean")
    S = Sphere(2)

    def hand():
        a = np.deg2rad(120)
        X = np.array([[1, 0, 0]] * 3 + [[np.cos(a), np.sin(a), 0]], float)
        m, h = frechet_mean(S, X)
        assert "objective" in h and "grad_norm" in h, "history must have keys 'objective' and 'grad_norm'"
        return abs(np.degrees(np.arctan2(m[1], m[0])) - 30.0)
    r.check("hand-built case: mean at 30 deg", hand, 1e-6)

    def gs():
        mu = S.projection(np.array([0.2, 0.4, 0.9]))
        V = S.to_tangent(np.broadcast_to(mu, (200, 3)), rng.normal(scale=0.4, size=(200, 3)))
        X = S.exp(np.broadcast_to(mu, V.shape), V)
        m, _ = frechet_mean(S, X)
        # geomstats' default stopping rule (epsilon = 1e-4) is loose; tighten it.
        ref = oracle.FrechetMean(oracle.Hypersphere(dim=2)).set(epsilon=1e-20, max_iter=500).fit(X).estimate_
        return _err(m, ref)
    r.check("matches geomstats FrechetMean on S^2", gs, 1e-6)

    def history():
        X = S.projection(rng.normal(size=(50, 3)) * 0.2 + np.array([0, 0, 1]))
        _, h = frechet_mean(S, X, tol=1e-12)
        g = np.asarray(h["grad_norm"])
        return g[-1] < 1e-10 and len(h["objective"]) == len(g)
    r.check("history: grad_norm decreases below tol", history)

    def pose():
        M = PoseManifold(lam=0.1)
        P = M.random_point(30, rng=rng)
        P[:, 3:] = rot.quat_from_matrix(rot.matrix_from_rotvec(rng.normal(scale=0.3, size=(30, 3))))
        m, _ = frechet_mean(M, P)
        return _err(m[:3], P[:, :3].mean(0))
    r.check("works on PoseManifold; position part = arithmetic mean", pose, 1e-8)
    r.show()


# ======================================================================= Part 2
def part2(Sphere, UnitQuaternions, WrappedGaussian, quat_body_basis, seed=0):
    rng = _rng(seed)
    r = _Report("Part 2: WrappedGaussian")
    S = Sphere(2)
    mu = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    cov = np.array([[0.02, 0.005], [0.005, 0.01]])

    def basis():
        B = WrappedGaussian(S, mu, cov).tangent_basis()
        assert B.shape == (3, 2), f"tangent_basis shape {B.shape}, expected (3, 2)"
        return max(_err(B.T @ B, np.eye(2)), np.abs(mu @ B).max())
    r.check("tangent_basis: orthonormal and orthogonal to the mean", basis, 1e-12)

    def tangent():
        V = WrappedGaussian(S, mu, cov).tangent_samples(1000, rng=rng)
        assert V.shape == (1000, 3), f"tangent_samples shape {V.shape}, expected (1000, 3)"
        return np.abs(V @ mu).max()
    r.check("tangent_samples: shape (N, 3) and tangent at the mean", tangent, 1e-12)

    def stats():
        W = WrappedGaussian(S, mu, cov)
        X = W.sample(20000, rng=rng)
        Z = S.log(np.broadcast_to(mu, X.shape), X) @ W.tangent_basis()
        return max(np.abs(Z.mean(0)).max(), np.abs(np.cov(Z.T) - cov).max())
    r.check("sample: Log_mu(x) has mean ~0 and covariance ~cov (small sigma)", stats, 3e-3)

    def on_sphere():
        X = WrappedGaussian(S, mu, 4.0 * np.eye(2)).sample(500, rng=rng)
        return np.all(S.belongs(X))
    r.check("sample: points lie on the sphere (large sigma too)", on_sphere)

    def s3():
        Q = UnitQuaternions()
        q0 = Q.random_point(rng=rng)
        X = WrappedGaussian(Q, q0, 0.01 * np.eye(3)).sample(100, rng=rng)
        return X.shape == (100, 4) and np.all(Q.belongs(X))
    r.check("works on S^3 (UnitQuaternions) with a 3x3 cov", s3)

    def body():
        Q = UnitQuaternions()
        q0 = Q.random_point(rng=rng)
        B = quat_body_basis(q0)
        w = rng.normal(size=3) * 0.7
        R_expected = rot.quat_to_matrix(q0) @ rot.matrix_from_rotvec(w)
        R_got = rot.quat_to_matrix(Q.exp(q0, B @ (w / 2)))
        return max(_err(B.T @ B, np.eye(3)), np.abs(q0 @ B).max(), _err(R_got, R_expected))
    r.check("quat_body_basis: Exp_q(B w/2) is the rotation R_q exp([w]x)", body, 1e-10)
    r.show()


# ======================================================================= Parts 3 to 5 (PyTorch)
def _torch():
    import torch
    torch.set_default_dtype(torch.float64)
    return torch


def _test_metric(theta=None):
    """A smooth, anisotropic metric field for testing; theta (6,) are parameters."""
    torch = _torch()
    from .metrics import vec_to_sym
    theta = torch.tensor([0.3, -0.2, 0.5, 0.4, 0.1, -0.3]) if theta is None else theta

    def metric(x):
        S = vec_to_sym(torch.sin(x @ torch.ones(3, 6) * theta + torch.arange(6.0)), 3)
        return torch.linalg.matrix_exp(0.5 * S) + 0.2 * torch.eye(3)
    return metric


def _random_spd(torch, n, batch, g, scale=1.0):
    A = torch.randn(batch, n, n, generator=g)
    return torch.linalg.matrix_exp(scale * 0.5 * (A + A.transpose(-1, -2)))


def part3(ConformalMetric, seed=0):
    torch = _torch()
    rng = _rng(seed)
    r = _Report("Part 3: ConformalMetric")
    data = rng.normal(size=(40, 3)) * 0.1
    x = rng.normal(size=(7, 3)) * 0.1
    sigma, eps = 0.07, 0.02
    p_ref = np.exp(-((x[:, None] - data[None]) ** 2).sum(-1) / (2 * sigma ** 2)).mean(1)

    def density():
        p = ConformalMetric(data, sigma=sigma, eps=eps).density(torch.as_tensor(x))
        assert tuple(p.shape) == (7,), f"density shape {tuple(p.shape)}, expected (7,)"
        return _err(p.detach().numpy(), p_ref)
    r.check("density(x): the kernel density estimate, shape (N,)", density, 1e-12)

    def metric():
        M = ConformalMetric(data, sigma=sigma, eps=eps)(torch.as_tensor(x))
        assert tuple(M.shape) == (7, 3, 3), f"metric shape {tuple(M.shape)}, expected (7, 3, 3)"
        return _err(M.detach().numpy(), np.eye(3) / (p_ref + eps)[:, None, None])
    r.check("metric(x) = I / (p(x) + eps), shape (N, 3, 3)", metric, 1e-12)

    def differentiable():
        xt = torch.as_tensor(x).requires_grad_(True)
        M = ConformalMetric(data, sigma=sigma, eps=eps)(xt)
        g, = torch.autograd.grad(M[:, 0, 0].sum(), xt)
        h = 1e-6
        fd = [(ConformalMetric(data, sigma=sigma, eps=eps)(torch.as_tensor(x + h * e))[:, 0, 0]
               - ConformalMetric(data, sigma=sigma, eps=eps)(torch.as_tensor(x - h * e))[:, 0, 0]).detach().numpy() / (2 * h)
              for e in np.eye(3)]
        return _err(g.numpy(), np.stack(fd, 1)) / max(1.0, np.abs(g.numpy()).max())
    r.check("autograd through metric(x) matches finite differences", differentiable, 1e-6)
    r.show()


def part4(christoffel, airm_rgrad, airm_exp, rgd_step, seed=0):
    torch = _torch()
    from . import metrics as mt
    g = torch.Generator().manual_seed(seed)
    r = _Report("Part 4: Christoffel symbols and SPD gradient descent")
    x = 0.5 * torch.randn(6, 3, generator=g)

    def shape_zero():
        G = christoffel(mt.euclidean, x)
        assert tuple(G.shape) == (6, 3, 3, 3), f"shape {tuple(G.shape)}, expected (6, 3, 3, 3)"
        return float(G.detach().abs().max())
    r.check("christoffel: shape (N, 3, 3, 3); zero for M = I", shape_zero, 1e-14)

    def hyperbolic():
        metric = lambda y: (1.0 / y[:, 2:3, None] ** 2) * torch.eye(3)        # noqa: E731
        y = x.clone()
        y[:, 2] = y[:, 2].abs() + 0.5
        v = torch.randn(6, 3, generator=g)
        G = christoffel(metric, y)
        acc = torch.einsum("nkij,ni,nj->nk", G, v, v)
        z = y[:, 2]
        ref = torch.stack([-2 * v[:, 0] * v[:, 2] / z, -2 * v[:, 1] * v[:, 2] / z,
                           (v[:, 0] ** 2 + v[:, 1] ** 2 - v[:, 2] ** 2) / z], 1)
        return float((acc.detach() - ref).abs().max())
    r.check("christoffel: M = I / z^2 gives the known Gamma(v, v)", hyperbolic, 1e-10)

    def versus_fd():
        metric = _test_metric()
        G = christoffel(metric, x).detach()
        ref = mt.christoffel_fd(metric, x, h=1e-5)
        return float((G - ref).abs().max() / ref.abs().max())
    r.check("christoffel: matches finite differences (anisotropic metric)", versus_fd, 1e-7)

    def second_order():
        theta = torch.tensor([0.3, -0.2, 0.5, 0.4, 0.1, -0.3], requires_grad=True)
        G = christoffel(_test_metric(theta), x)
        dtheta, = torch.autograd.grad((G ** 2).sum(), theta)
        h = 1e-6
        fd = []
        for e in torch.eye(6):
            fp = (christoffel(_test_metric(theta.detach() + h * e), x).detach() ** 2).sum()
            fm = (christoffel(_test_metric(theta.detach() - h * e), x).detach() ** 2).sum()
            fd.append((fp - fm) / (2 * h))
        fd = torch.stack(fd)
        return float((dtheta - fd).abs().max() / fd.abs().max())
    r.check("christoffel: differentiable w.r.t. the metric's parameters", second_order, 1e-6)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import geoopt
    spd = geoopt.SymmetricPositiveDefinite()
    P = _random_spd(torch, 4, 5, g)
    G = torch.randn(5, 4, 4, generator=g)
    xi = 0.5 * (G + G.transpose(-1, -2))

    def rgrad():
        return float((airm_rgrad(P, G) - spd.egrad2rgrad(P, G)).abs().max())
    r.check("airm_rgrad(P, G) = P sym(G) P (geoopt)", rgrad, 1e-12)

    def expmap():
        E = airm_exp(P, xi)
        return float((E - spd.expmap(P, xi)).abs().max() / E.abs().max())
    r.check("airm_exp(P, xi) matches geoopt's expmap", expmap, 1e-10)

    def spd_out():
        E = airm_exp(P, 3.0 * xi)
        sym_err = float((E - E.transpose(-1, -2)).abs().max())
        return sym_err < 1e-10 and bool((torch.linalg.eigvalsh(E) > 0).all())
    r.check("airm_exp: symmetric positive definite, also for large steps", spd_out)

    def invariance():
        A = torch.randn(4, 4, generator=g) + 2 * torch.eye(4)
        lhs = airm_exp(A.T @ P @ A, A.T @ xi @ A)
        rhs = A.T @ airm_exp(P, xi) @ A
        return float((lhs - rhs).abs().max() / rhs.abs().max())
    r.check("airm_exp is affine invariant: Exp_{A'PA}(A'xA) = A' Exp_P(x) A", invariance, 1e-10)

    def step():
        S = rgd_step(P, G, 0.1)
        return float((S - spd.expmap(P, -0.1 * spd.egrad2rgrad(P, G))).abs().max() / S.abs().max())
    r.check("rgd_step(P, G, lr) = Exp_P(-lr grad)", step, 1e-10)

    def at_identity():
        I = torch.eye(3).repeat(2, 1, 1).requires_grad_(True)
        loss = (mt.spd_dist(torch.eye(3), rgd_step(I, torch.ones(2, 3, 3), 0.1)) ** 2).sum()
        gI, = torch.autograd.grad(loss, I)
        return bool(torch.isfinite(gI).all())
    r.check("rgd_step at P = I: finite gradients (use mt.spd_* functions)", at_identity)
    r.show()


def part5(bimap, reeig, SPDMetricNet, airm_exp, seed=0):
    torch = _torch()
    from . import metrics as mt
    g = torch.Generator().manual_seed(seed)
    r = _Report("Part 5: SPD layers")
    Y = _random_spd(torch, 6, 5, g)
    W = torch.linalg.qr(torch.randn(6, 3, generator=g))[0]

    def bm():
        Z = bimap(Y, W)
        assert tuple(Z.shape) == (5, 3, 3), f"shape {tuple(Z.shape)}, expected (5, 3, 3)"
        return float((Z - W.T @ Y @ W).abs().max())
    r.check("bimap(Y, W) = W^T Y W, shape (N, m, m)", bm, 1e-12)

    def re():
        S = torch.randn(5, 6, 6, generator=g)
        S = 0.5 * (S + S.transpose(-1, -2))
        Z = reeig(S, eps=0.1)
        lam, U = torch.linalg.eigh(S)
        ref = U @ torch.diag_embed(0.1 + torch.nn.functional.softplus(lam - 0.1)) @ U.transpose(-1, -2)
        return float((Z - ref).abs().max())
    r.check("reeig(Y, eps): eigenvalues eps + softplus(lambda - eps)", re, 1e-10)

    def forward(use_bimap, use_reeig):
        def fn():
            net = SPDMetricNet(use_bimap=use_bimap, use_reeig=use_reeig, seed=1)
            with torch.no_grad():
                for p in net.mlp.parameters():
                    p.copy_(0.3 * torch.randn(p.shape, generator=g))
            x = torch.randn(4, 3, generator=g)
            M = net.shape(x)
            assert tuple(M.shape) == (4, 3, 3), f"shape(x) has shape {tuple(M.shape)}, expected (4, 3, 3)"
            with torch.no_grad():
                P, W1, W2 = net.weights()
                Z = airm_exp(P, mt.vec_to_sym(net.mlp((x - net.centre) / net.scale), net.n))
                if use_bimap:
                    Z = W1.T @ Z @ W1
                    if use_reeig:
                        lam, U = torch.linalg.eigh(Z)
                        Z = U @ torch.diag_embed(net.eps + torch.nn.functional.softplus(lam - net.eps)) @ U.transpose(-1, -2)
                    Z = W2.T @ Z @ W2
            return float((M.detach() - Z).abs().max() / Z.abs().max())
        return fn

    def manifold_params():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import geoopt
        net = SPDMetricNet(seed=1)
        ok = isinstance(net.P, geoopt.ManifoldParameter) and isinstance(net.P.manifold, geoopt.SymmetricPositiveDefinite)
        for W, shape in [(net.W1, (net.n, 6)), (net.W2, (6, 3))]:
            ok &= isinstance(W, geoopt.ManifoldParameter) and isinstance(W.manifold, geoopt.Stiefel)
            ok &= tuple(W.shape) == shape and float((W.T @ W - torch.eye(shape[1])).detach().abs().max()) < 1e-12
        return ok
    r.check("riemannian variant: P on the SPD manifold, W1, W2 on Stiefel (geoopt)", manifold_params)
    r.check("SPDMetricNet.shape: MLP, Exp_P, BiMap, ReEig, BiMap", forward(True, True), 1e-10)
    r.check("SPDMetricNet.shape with use_reeig=False", forward(True, False), 1e-10)
    r.check("SPDMetricNet.shape with use_bimap=False (M = Exp_P(xi), n = 3)", forward(False, False), 1e-10)

    def with_prior():
        prior = lambda y: (1.0 + (y ** 2).sum(-1))[:, None, None] * torch.eye(3)      # noqa: E731
        net = SPDMetricNet(seed=1, prior=prior)
        x = torch.randn(4, 3, generator=g)
        with torch.no_grad():
            return float((net(x) - prior(x) @ net.shape(x)).abs().max())
    r.check("SPDMetricNet.forward = prior(x) @ shape(x)", with_prior, 1e-12)
    r.show()
