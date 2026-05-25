"""
verify_weight_recovery.py
=========================
Verification script for eigenmode weight (V) recovery.

Three paradigms:
  Paradigm 1: Know x_bar  → estimate V only            (oracle E-step)
  Paradigm 2: Know A      → estimate x_bar and V       (EM, fixed A)
  Paradigm 3: Know nothing → estimate A, x_bar, and V  (full EM)

Metrics (Frobenius norm based):
  - rel_err_V  = ||V_est - V_true||_F / ||V_true||_F
  - rel_err_xbar = ||x_est - x_true||_F / ||x_true||_F
  - rel_err_A  = ||A_est - A_true||_F / ||A_true||_F

Why Frobenius?
  - Captures total reconstruction error across all elements
  - Scale-invariant when normalised by true norm
  - Standard metric for matrix recovery problems

Results saved to: results/weight_recovery.pkl

Run:
    cd /Users/as12123/NLGC
    python verify_weight_recovery.py
"""

import numpy as np
import pickle
import os
from scipy import linalg

# ===================================================================
# Helpers
# ===================================================================
def rel_err(est, true):
    """Relative error ||est - true|| / ||true||.
    Works for both vectors (L2) and matrices (Frobenius).
    """
    est = np.asarray(est).ravel()
    true = np.asarray(true).ravel()
    return float(linalg.norm(est - true) / max(linalg.norm(true), 1e-12))


def build_toy_problem(n_channels=10, n_dipoles=15, n_rois=3, K=2,
                      p=2, T=1000, snr_db=10.0, seed=42):
    """Build synthetic problem with known ground truth.

    Returns
    -------
    dict with all arrays needed for verification.
    """
    rng = np.random.default_rng(seed)
    m = n_rois * K

    # --- Per-ROI forward via SVD ---
    G_T_list_true = []
    V_true_list = []
    f_blocks = []

    for j in range(n_rois):
        G_j = rng.standard_normal((n_channels, n_dipoles))
        G_j_T = G_j.T
        u, s, vh = linalg.svd(G_j_T, full_matrices=False)
        # True weights: random perturbation of SVD init
        V_j_true = u[:, :K] + 0.3 * rng.standard_normal((n_dipoles, K))
        # Re-normalise columns
        V_j_true /= (linalg.norm(V_j_true, axis=0, keepdims=True) + 1e-12)
        F_j = G_j @ V_j_true
        G_T_list_true.append(G_j_T.copy())
        V_true_list.append(V_j_true.copy())
        f_blocks.append(F_j)

    f_true = np.concatenate(f_blocks, axis=1)  # (n_channels, m)
    r = np.eye(n_channels) * 0.5

    # --- True VAR parameters (scaled to m) ---
    a_true = np.zeros((p, m, m))
    # Self-history for each source
    for i in range(m):
        a_true[0, i, i] = 0.6 + 0.1 * (i % 3)
    # One cross-ROI link (always valid: ROI 0 mode 0 -> ROI 1 mode 0)
    if m > K:
        a_true[1, 0, K] = -0.15

    q_true = np.eye(m)

    # --- Simulate VAR sources ---
    x_true = np.zeros((T, m))
    for t in range(p, T):
        for k in range(p):
            x_true[t] += a_true[k] @ x_true[t - 1 - k]
        x_true[t] += rng.standard_normal(m)

    # --- Simulate observations ---
    y_clean = x_true @ f_true.T
    sig_pow = float(np.mean(y_clean ** 2))
    snr_lin = 10.0 ** (snr_db / 10.0)
    sigma = np.sqrt(sig_pow / snr_lin) if sig_pow > 0 else 1.0
    noise = rng.standard_normal(y_clean.shape) * sigma
    y = (y_clean + noise).T  # (n_channels, T)

    # Initial V from SVD (starting point for estimation)
    V_init_list = []
    for j in range(n_rois):
        u, s, vh = linalg.svd(G_T_list_true[j], full_matrices=False)
        V_init_list.append(u[:, :K].copy())

    # Initial forward from SVD init
    f_init = np.concatenate(
        [G_T_list_true[j].T @ V_init_list[j] for j in range(n_rois)], axis=1)

    return {
        'y': y, 'f_true': f_true, 'f_init': f_init,
        'r': r, 'a_true': a_true, 'q_true': q_true,
        'G_T_list': G_T_list_true,
        'V_true_list': V_true_list,
        'V_init_list': V_init_list,
        'x_true': x_true,
        'n_channels': n_channels, 'n_dipoles': n_dipoles,
        'n_rois': n_rois, 'K': K, 'p': p, 'T': T, 'm': m,
        'sigma': sigma,
    }


# ===================================================================
# Paradigm 1: Know x_bar — estimate V only
# (Oracle E-step: uses true x as x_bar)
# ===================================================================
def paradigm1_known_xbar(d, n_iters=20, init_prior_var=1.0):
    """Estimate V only, given true x_bar.

    This is the easiest case: x_bar is known (oracle), and we only
    solve the coordinate-descent V update.

    Reconstruction error:
        rel_err_V only (x and A are known/given)
    """
    from nlgc.opt.m_step import update_eigenmode_weights

    x_bar = d['x_true'][:, :d['m']]      # (T, m)  — true sources
    y_t = d['y'].T                         # (T, n_channels)
    K = d['K']
    V_list = [v.copy() for v in d['V_init_list']]
    prior_var = float(init_prior_var)

    for it in range(n_iters):
        V_list, f_weighted, prior_var = update_eigenmode_weights(
            y_t, x_bar, d['f_init'], d['G_T_list'], V_list,
            d['r'], n_eigenmodes=K, prior_var=prior_var)

    # Stack for comparison
    V_est = np.concatenate([V_list[j] for j in range(d['n_rois'])], axis=0)
    V_tru = np.concatenate([d['V_true_list'][j] for j in range(d['n_rois'])], axis=0)

    err_V = rel_err(V_est, V_tru)
    print(f"  [P1] Known x_bar  | rel_err_V = {err_V:.4f} | prior_var = {prior_var:.4f}")
    return {
        'paradigm': 1,
        'description': 'Known x_bar: estimate V only',
        'rel_err_V': err_V,
        'V_est': V_list,
        'V_true': d['V_true_list'],
        'learned_prior_var': prior_var,
    }


# ===================================================================
# Paradigm 2: Know A — estimate x_bar and V jointly
# (EM with fixed A, updating x_bar via Kalman + V via weight update)
# ===================================================================
def paradigm2_known_A(d, max_iter=100, init_prior_var=1.0):
    """Estimate x_bar and V jointly, given true A.

    A is fixed at its true value. EM alternates between:
      E-step: Kalman smoother (updates x_bar given current V)
      M-step: V update (updates V given current x_bar)
              Q update only (A is fixed)

    Reconstruction errors:
        rel_err_V, rel_err_xbar
    """
    from nlgc.opt import NeuraLVAR

    model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model.set_roi_data(d['G_T_list'], [v.copy() for v in d['V_init_list']])

    model.fit(
        d['y'], d['f_init'].copy(), d['r'],
        lambda2=0.1, max_iter=max_iter, max_cyclic_iter=2,
        a_init=d['a_true'].copy(),   # fix A at true value
        q_init=0.1 * d['q_true'],
        rel_tol=1e-5,
        update_weights=True, weight_prior_var=init_prior_var,
    )

    a_est, f_est, q_est, r_est, x_est = model._parameters
    V_est_list = model.eigenmode_weights

    # x_est shape (T, m) from _parameters
    err_xbar = rel_err(x_est[:, :d['m']], d['x_true'][:, :d['m']])
    V_est = np.concatenate(V_est_list, axis=0)
    V_tru = np.concatenate(d['V_true_list'], axis=0)
    err_V = rel_err(V_est, V_tru)

    print(f"  [P2] Known A      | rel_err_V = {err_V:.4f} | "
          f"rel_err_xbar = {err_xbar:.4f} | "
          f"prior_var = {model.prior_var:.4f}")
    return {
        'paradigm': 2,
        'description': 'Known A: estimate x_bar and V jointly',
        'rel_err_V': err_V,
        'rel_err_xbar': err_xbar,
        'V_est': V_est_list,
        'V_true': d['V_true_list'],
        'x_est': x_est,
        'x_true': d['x_true'],
        'learned_prior_var': model.prior_var,
    }


# ===================================================================
# Paradigm 3: Know nothing — estimate A, x_bar, and V jointly
# (Full EM)
# ===================================================================
def paradigm3_full_em(d, max_iter=200, init_prior_var=1.0):
    """Estimate A, x_bar, and V jointly (full algorithm).

    Nothing is assumed known. EM iterates:
      E-step: Kalman smoother
      M-step: update A, Q, V

    Reconstruction errors:
        rel_err_A, rel_err_V, rel_err_xbar
    """
    from nlgc.opt import NeuraLVAR

    model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model.set_roi_data(d['G_T_list'], [v.copy() for v in d['V_init_list']])

    model.fit(
        d['y'], d['f_init'].copy(), d['r'],
        lambda2=0.1, max_iter=max_iter, max_cyclic_iter=2,
        a_init=None,
        q_init=0.1 * d['q_true'],
        rel_tol=1e-5,
        update_weights=True, weight_prior_var=init_prior_var,
    )

    a_est, f_est, q_est, r_est, x_est = model._parameters
    V_est_list = model.eigenmode_weights

    # flatten both A matrices the same way for comparison
    err_A = rel_err(np.array(a_est).ravel(), d['a_true'].ravel())
    err_xbar = rel_err(x_est[:, :d['m']], d['x_true'][:, :d['m']])
    V_est = np.concatenate(V_est_list, axis=0)
    V_tru = np.concatenate(d['V_true_list'], axis=0)
    err_V = rel_err(V_est, V_tru)

    print(f"  [P3] Full EM      | rel_err_A = {err_A:.4f} | "
          f"rel_err_V = {err_V:.4f} | rel_err_xbar = {err_xbar:.4f} | "
          f"prior_var = {model.prior_var:.4f}")
    return {
        'paradigm': 3,
        'description': 'Full EM: estimate A, x_bar, and V jointly',
        'rel_err_A': err_A,
        'rel_err_V': err_V,
        'rel_err_xbar': err_xbar,
        'V_est': V_est_list,
        'V_true': d['V_true_list'],
        'a_est': a_est,
        'a_true': d['a_true'],
        'x_est': x_est,
        'x_true': d['x_true'],
        'learned_prior_var': model.prior_var,
    }


# ===================================================================
# Run across different n_eigenmodes
# ===================================================================
def run_all(n_eigenmodes_list=(1, 2, 4), snr_db=10.0, seed=42):
    all_results = {}

    for K in n_eigenmodes_list:
        print(f"\n{'='*55}")
        print(f"n_eigenmodes = {K}  |  SNR = {snr_db} dB")
        print('='*55)

        d = build_toy_problem(
            n_channels=12, n_dipoles=20, n_rois=3, K=K,
            p=2, T=1000, snr_db=snr_db, seed=seed)

        results_K = {}

        print("Paradigm 1: know x_bar, estimate V only")
        results_K['P1'] = paradigm1_known_xbar(d, n_iters=30)

        print("Paradigm 2: know A, estimate x_bar + V")
        results_K['P2'] = paradigm2_known_A(d, max_iter=100)

        print("Paradigm 3: full EM — estimate A + x_bar + V")
        results_K['P3'] = paradigm3_full_em(d, max_iter=200)

        results_K['data'] = d
        all_results[K] = results_K

    return all_results


# ===================================================================
# Summary table
# ===================================================================
def print_summary(all_results):
    print(f"\n{'='*65}")
    print("SUMMARY TABLE — Relative Frobenius Reconstruction Errors")
    print(f"{'='*65}")
    print(f"{'K':>4}  {'Paradigm':<40}  {'err_A':>7}  {'err_V':>7}  {'err_xbar':>8}")
    print('-'*65)
    for K, results_K in all_results.items():
        for pkey in ['P1', 'P2', 'P3']:
            r = results_K[pkey]
            err_A = f"{r.get('rel_err_A', float('nan')):.4f}"
            err_V = f"{r.get('rel_err_V', float('nan')):.4f}"
            err_x = f"{r.get('rel_err_xbar', float('nan')):.4f}"
            print(f"{K:>4}  {r['description']:<40}  {err_A:>7}  {err_V:>7}  {err_x:>8}")
    print('='*65)

    print("\nMetric justification:")
    print("  Frobenius relative error  ||est - true||_F / ||true||_F")
    print("  - Captures total element-wise reconstruction error")
    print("  - Normalised => scale-invariant across ROIs and parameters")
    print("  - Standard for matrix recovery; directly comparable across K")


# ===================================================================
# Main
# ===================================================================
if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    all_results = run_all(
        n_eigenmodes_list=[1, 2, 4],
        snr_db=10.0,
        seed=42,
    )

    print_summary(all_results)

    # Save reconstructed values
    save_path = 'results/weight_recovery.pkl'
    # Remove raw data arrays to keep file size small but keep estimates
    save_data = {}
    for K, results_K in all_results.items():
        save_data[K] = {
            pkey: {k: v for k, v in r.items() if k != 'data'}
            for pkey, r in results_K.items()
            if pkey != 'data'
        }

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"\nResults saved to {save_path}")
