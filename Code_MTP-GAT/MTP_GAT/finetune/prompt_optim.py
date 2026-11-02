
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F

try:
    from botorch.acquisition import qExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _HAS_BOTORCH = True
except Exception:
    _HAS_BOTORCH = False

def get_probs(logits, temperature=1.0):
    logits = torch.as_tensor(logits)
    return F.softmax(logits / max(float(temperature), 1e-8), dim=-1)


def _as_pair_tensor(pairs, device):
    if pairs is None or len(pairs) == 0:
        return None
    if torch.is_tensor(pairs):
        return pairs.to(device=device, dtype=torch.long).view(-1, 2)
    return torch.as_tensor(np.asarray(pairs), device=device, dtype=torch.long).view(-1, 2)


def _as_weight_tensor(w, n, device):
    if w is None:
        return torch.ones(n, device=device, dtype=torch.float32)
    if torch.is_tensor(w):
        w = w.to(device=device, dtype=torch.float32).view(-1)
    else:
        w = torch.as_tensor(np.asarray(w), device=device, dtype=torch.float32).view(-1)
    if w.numel() != n:
        raise ValueError(f"Weight count ({w.numel()}) does not match pair count ({n})")
    return w


def _mean_distance(emb, pairs, weights=None):
    if pairs is None or pairs.size(0) == 0:
        return None
    diff = emb[pairs[:, 0]] - emb[pairs[:, 1]]
    d = torch.linalg.norm(diff, dim=-1)
    w = _as_weight_tensor(weights, d.numel(), emb.device)
    return (w * d).sum()


def _warn_once(msg):
    warnings.warn(msg, RuntimeWarning, stacklevel=3)

def compute_aw_ncr(graph_reps,
                   logits,
                   noncliff_pairs,
                   ac_pairs,
                   noncliff_weights=None,
                   ac_weights=None,
                   temperature=0.5,
                   epsilon=1e-6):

    if graph_reps.dim() != 3:
        raise ValueError("graph_reps must have shape [K, B, D]")

    device = graph_reps.device
    logits = torch.as_tensor(logits, device=device, dtype=graph_reps.dtype)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    if logits.size(1) != graph_reps.size(0):
        raise ValueError(
            f"Logit dimension ({logits.size(1)}) "
            f"does not match expert count ({graph_reps.size(0)})"
        )

    probs = get_probs(logits, temperature)
    nc_pairs = _as_pair_tensor(noncliff_pairs, device)
    ac_pairs_t = _as_pair_tensor(ac_pairs, device)

    values = []

    for n in range(probs.size(0)):
        emb = torch.einsum(
            'kbd,k->bd',
            graph_reps,
            probs[n].to(graph_reps.dtype)
        )

        num = _mean_distance(emb, nc_pairs, noncliff_weights)
        den = _mean_distance(emb, ac_pairs_t, ac_weights)

        if num is None or den is None:
            _warn_once("AW-NCR requires valid noncliff and AC pairs. Returning inf.")
            values.append(torch.tensor(float('inf'), device=device))
            continue

        if float(den) <= 0:
            _warn_once("AW-NCR denominator is zero. Returning inf.")
            values.append(torch.tensor(float('inf'), device=device))
            continue

        values.append(num / (den + epsilon))

    return torch.stack(values)


def build_pairs_from_labels(labels,
                            cliff_delta=1.0,
                            noncliff_delta=0.2,
                            max_pairs=20000,
                            weight_by_delta=True,
                            seed=0):

    y = torch.as_tensor(labels).detach().float().view(-1)
    valid = torch.isfinite(y)
    idx = torch.nonzero(valid, as_tuple=False).view(-1)
    yv = y[idx]

    n = yv.numel()

    if n < 2:
        return None, None, None, None

    iu = torch.triu_indices(n, n, offset=1)
    dy = (yv[iu[0]] - yv[iu[1]]).abs()

    ac_mask = dy >= float(cliff_delta)
    nc_mask = dy <= float(noncliff_delta)

    rng = np.random.default_rng(seed)

    def _pack(mask):
        sub = iu[:, mask]

        if sub.size(1) == 0:
            return None, None

        pairs = idx[sub.t().reshape(-1)].view(-1, 2)
        w = dy[mask] if weight_by_delta else torch.ones(int(mask.sum()))

        if pairs.size(0) > max_pairs:
            keep = rng.choice(pairs.size(0), size=max_pairs, replace=False)
            keep = torch.as_tensor(np.sort(keep))
            pairs, w = pairs[keep], w[keep]

        return pairs, w

    nc_pairs, nc_w = _pack(nc_mask)
    ac_pairs, ac_w = _pack(ac_mask)

    if weight_by_delta:
        all_w = torch.cat([
            w for w in (nc_w, ac_w)
            if w is not None and w.numel() > 0
        ])

        scale = float(all_w.mean()) if all_w.numel() > 0 else 1.0
        scale = scale if scale > 0 else 1.0

        nc_w = nc_w / scale if nc_w is not None else None
        ac_w = ac_w / scale if ac_w is not None else None

    return nc_pairs, nc_w, ac_pairs, ac_w

def _to_logits(x):
    x = torch.as_tensor(x).double()
    return torch.cat([torch.zeros_like(x), x], dim=1)


def generate_data(graph_reps, noncliff_pairs, ac_pairs,
                  noncliff_weights, ac_weights,
                  n_inits, temperature, epsilon, logit_bounds):

    lower, upper = logit_bounds
    n_inits = max(int(n_inits), 3)

    xs = torch.linspace(
        lower, upper, n_inits,
        dtype=torch.double,
        device=graph_reps.device
    ).view(-1, 1)

    if not torch.any(torch.isclose(xs, torch.zeros_like(xs))):
        xs = torch.cat([
            xs,
            torch.zeros(1, 1, dtype=torch.double, device=graph_reps.device)
        ], dim=0)

    with torch.no_grad():
        ncr = compute_aw_ncr(
            graph_reps,
            _to_logits(xs),
            noncliff_pairs,
            ac_pairs,
            noncliff_weights,
            ac_weights,
            temperature,
            epsilon
        )

    ys = -ncr.double()
    ys = torch.nan_to_num(ys, nan=-1e6, posinf=-1e6, neginf=-1e6)

    return xs, ys


def get_next_points(init_x, init_y, bounds, n_points=1,
                    n_restarts=10, n_samples=128, mc_samples=128):

    lower, upper = float(bounds[0, 0]), float(bounds[1, 0])

    if _HAS_BOTORCH:
        train_x = init_x.double()
        train_y = init_y.double().view(-1, 1)

        model = SingleTaskGP(train_x, train_y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([mc_samples])
        )

        acq = qExpectedImprovement(
            model=model,
            best_f=train_y.max(),
            sampler=sampler
        )

        candidates, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor([[lower], [upper]], dtype=torch.double),
            q=n_points,
            num_restarts=max(int(n_restarts), 1),
            raw_samples=max(int(n_samples), 16),
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    x = init_x.double().cpu().numpy().reshape(-1, 1)
    y = init_y.double().cpu().numpy().reshape(-1)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        nu=2.5
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(x, y)

    grid = np.linspace(
        lower, upper,
        max(int(n_samples) * 8, 256)
    ).reshape(-1, 1)

    mu, sd = gp.predict(grid, return_std=True)
    sd = np.maximum(sd, 1e-9)

    best = float(np.max(y))
    xi = 0.01 * (float(np.std(y)) + 1e-6)

    z = (mu - best - xi) / sd
    ei = (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)

    order = np.argsort(-ei)
    picks, seen = [], set()

    for k in order:
        key = round(float(grid[k, 0]), 6)

        if key in seen:
            continue

        seen.add(key)
        picks.append(k)

        if len(picks) >= n_points:
            break

    return torch.as_tensor(grid[picks], dtype=torch.double)


def optimize_moe_init_weights(graph_reps,
                              noncliff_pairs,
                              ac_pairs,
                              noncliff_weights=None,
                              ac_weights=None,
                              n_runs=20,
                              n_inits=10,
                              n_points=1,
                              n_restarts=10,
                              n_samples=128,
                              mc_samples=128,
                              temperature=0.5,
                              epsilon=1e-6,
                              logit_bounds=(-2.0, 2.0),
                              skip_bo=False,
                              verbose=True):

    start_time = time.time()

    if graph_reps.dim() != 3 or graph_reps.size(0) != 2:
        raise ValueError(
            "graph_reps must have shape [2, B, D], "
            "with order [MolSim, ActCliff]."
        )

    lower, upper = logit_bounds

    if lower >= upper:
        raise ValueError("logit_bounds must satisfy lower < upper.")

    if noncliff_pairs is None or ac_pairs is None:
        _warn_once("Missing noncliff or AC pairs. Using equal-weight initialization.")
        zeros = torch.zeros(2, device=graph_reps.device)
        return zeros, get_probs(zeros, temperature)

    init_x, init_y = generate_data(
        graph_reps,
        noncliff_pairs,
        ac_pairs,
        noncliff_weights,
        ac_weights,
        n_inits,
        temperature,
        epsilon,
        logit_bounds,
    )

    if verbose:
        idx = int(torch.argmax(init_y).item())

        print(f"[AW-NCR] Initial best AW-NCR = {-init_y[idx].item():.6f}")

        print(
            "[AW-NCR] Initial gating probabilities [MolSim, ActCliff]:",
            get_probs(
                _to_logits(init_x[idx:idx + 1]),
                temperature
            )[0].detach().cpu().numpy()
        )

    if not skip_bo:
        bounds = torch.tensor(
            [[lower], [upper]],
            device=graph_reps.device,
            dtype=torch.double
        )

        for it in range(int(n_runs)):
            try:
                candidates = get_next_points(
                    init_x,
                    init_y,
                    bounds,
                    n_points,
                    n_restarts,
                    n_samples,
                    mc_samples
                )

            except Exception as exc:
                print(
                    f"[AW-NCR] Bayesian optimization failed at "
                    f"iteration {it + 1} ({exc}). Stopping search."
                )
                break

            with torch.no_grad():
                cand_ncr = compute_aw_ncr(
                    graph_reps,
                    _to_logits(candidates),
                    noncliff_pairs,
                    ac_pairs,
                    noncliff_weights,
                    ac_weights,
                    temperature,
                    epsilon
                )

            cand_y = torch.nan_to_num(
                -cand_ncr.double(),
                nan=-1e6,
                posinf=-1e6,
                neginf=-1e6
            )

            init_x = torch.cat([init_x, candidates.double()], dim=0)
            init_y = torch.cat([init_y, cand_y], dim=0)

            if verbose:
                print(
                    f"[AW-NCR] Iteration {it + 1}/{n_runs} | "
                    f"Current best AW-NCR = {-init_y.max().item():.6f}"
                )

    best_idx = int(torch.argmax(init_y).item())
    best_x = init_x[best_idx]

    best_logits = torch.cat([
        torch.zeros(1, device=best_x.device, dtype=best_x.dtype),
        best_x.reshape(1),
    ])

    best_probs = get_probs(best_logits, temperature)

    if verbose:
        print("[AW-NCR] Best initial logits:", best_logits.detach().cpu().numpy())
        print(
            "[AW-NCR] Best gating probabilities [MolSim, ActCliff]:",
            best_probs.detach().cpu().numpy()
        )
        print(f"[AW-NCR] Minimum AW-NCR = {-init_y.max().item():.6f}")
        print(f"[AW-NCR] Elapsed time: {time.time() - start_time:.2f} s")

    return best_logits.detach(), best_probs.detach()

def optimize_prompt_weight_ri(graph_reps,
                              labels,
                              n_runs=50,
                              n_inits=50,
                              n_points=5,
                              n_restarts=512,
                              n_samples=512,
                              temperature=0.5,
                              metric='euclidean',
                              skip_bo=False,
                              verbose=True,
                              cliff_delta=1.0,
                              noncliff_delta=0.2,
                              max_pairs=20000,
                              **kwargs):

    nc_pairs, nc_w, ac_pairs, ac_w = build_pairs_from_labels(
        labels,
        cliff_delta=cliff_delta,
        noncliff_delta=noncliff_delta,
        max_pairs=max_pairs,
    )

    if verbose:
        n_nc = 0 if nc_pairs is None else nc_pairs.size(0)
        n_ac = 0 if ac_pairs is None else ac_pairs.size(0)

        print(
            f"[AW-NCR] Training pairs: noncliff {n_nc}, AC {n_ac} "
            f"(cliff_delta={cliff_delta}, noncliff_delta={noncliff_delta})"
        )

    logits, probs = optimize_moe_init_weights(
        graph_reps,
        nc_pairs,
        ac_pairs,
        nc_w,
        ac_w,
        n_runs=n_runs,
        n_inits=n_inits,
        n_points=n_points,
        n_restarts=n_restarts,
        n_samples=n_samples,
        temperature=temperature,
        skip_bo=skip_bo,
        verbose=verbose,
    )

    return logits.to(graph_reps.device)
