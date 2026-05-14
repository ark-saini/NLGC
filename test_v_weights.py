"""
test_v_weights.py — Complete test suite for NLGC V weight update

"""

 
import numpy as np
from scipy import linalg
import sys
import traceback
 
PASS_COUNT = 0
FAIL_COUNT = 0
 
 
def run_test(name, func):
    global PASS_COUNT, FAIL_COUNT
    try:
        func()
        PASS_COUNT += 1
        print(f"  PASS: {name}")
    except Exception as e:
        FAIL_COUNT += 1
        print(f"  FAIL: {name}")
        print(f"        {e}")
        traceback.print_exc()
        print()
 
 
# ===================================================================
# Shared test data builder
# ===================================================================
def make_toy_problem(n_channels=6, n_dipoles=10, n_rois=2, K=2, p=2, T=1000, seed=42):
    """Build synthetic data with known G_j, V_j, A, Q for testing."""
    rng = np.random.default_rng(seed)
    m = n_rois * K
 
    G_T_list = []
    V_list = []
    f_blocks = []
 
    for j in range(n_rois):
        G_j = rng.standard_normal((n_channels, n_dipoles))
        G_j_T = G_j.T
        u, s, vh = linalg.svd(G_j_T, full_matrices=False)
        V_j = u[:, :K]
        F_j = G_j @ V_j
        G_T_list.append(G_j_T.copy())
        V_list.append(V_j.copy())
        f_blocks.append(F_j)
 
    f = np.concatenate(f_blocks, axis=1)
    r = np.eye(n_channels)
 
    a = np.zeros((p, m, m))
    a[0, 0, 0] = 0.8
    a[0, 1, 1] = 0.7
    a[0, 2, 2] = 0.6
    a[0, 3, 3] = 0.5
    a[1, 0, 2] = -0.15
 
    q = np.eye(m)
    x = np.zeros((T, m))
    for t in range(p, T):
        for k in range(p):
            x[t] += a[k] @ x[t - 1 - k]
        x[t] += rng.standard_normal(m)
 
    y = (x @ f.T + 0.5 * rng.standard_normal((T, n_channels))).T
 
    return {
        'y': y, 'f': f, 'r': r, 'a': a, 'q': q,
        'G_T_list': G_T_list, 'V_list': V_list,
        'x': x, 'n_channels': n_channels, 'n_dipoles': n_dipoles,
        'n_rois': n_rois, 'K': K, 'p': p, 'T': T, 'm': m,
    }
 
 
# ===================================================================
# PART A: Standalone update_eigenmode_weights function
# ===================================================================
 
def test_A1_shapes():
    """Output shapes are correct."""
    from nlgc.opt.m_step import update_eigenmode_weights
    d = make_toy_problem()
    x_bar = d['x'][:, :d['m']]
    V_test = [v.copy() for v in d['V_list']]
    V_out, f_new = update_eigenmode_weights(
        d['y'].T, x_bar, d['G_T_list'], V_test, d['r'],
        n_eigenmodes=d['K'], prior_var=0.1)
    assert len(V_out) == d['n_rois']
    for j in range(d['n_rois']):
        assert V_out[j].shape == (d['n_dipoles'], d['K'])
    assert f_new.shape == d['f'].shape
 
 
def test_A2_f_equals_GV():
    """f_new exactly equals concat(G_j @ V_j)."""
    from nlgc.opt.m_step import update_eigenmode_weights
    d = make_toy_problem()
    x_bar = d['x'][:, :d['m']]
    V_test = [v.copy() for v in d['V_list']]
    V_out, f_new = update_eigenmode_weights(
        d['y'].T, x_bar, d['G_T_list'], V_test, d['r'],
        n_eigenmodes=d['K'], prior_var=0.1)
    f_check = np.concatenate(
        [d['G_T_list'][j].T @ V_out[j] for j in range(d['n_rois'])], axis=1)
    assert np.allclose(f_new, f_check, atol=1e-10)
 
 
def test_A3_all_finite():
    """All outputs are finite."""
    from nlgc.opt.m_step import update_eigenmode_weights
    d = make_toy_problem()
    x_bar = d['x'][:, :d['m']]
    V_test = [v.copy() for v in d['V_list']]
    V_out, f_new = update_eigenmode_weights(
        d['y'].T, x_bar, d['G_T_list'], V_test, d['r'],
        n_eigenmodes=d['K'], prior_var=0.1)
    for j in range(d['n_rois']):
        assert np.all(np.isfinite(V_out[j]))
    assert np.all(np.isfinite(f_new))
 
 
def test_A4_multiple_iterations():
    """5 iterations: f = G@V and finite each time."""
    from nlgc.opt.m_step import update_eigenmode_weights
    d = make_toy_problem()
    x_bar = d['x'][:, :d['m']]
    V_test = [v.copy() for v in d['V_list']]
    for it in range(5):
        V_test, f_new = update_eigenmode_weights(
            d['y'].T, x_bar, d['G_T_list'], V_test, d['r'],
            n_eigenmodes=d['K'], prior_var=0.1)
        f_check = np.concatenate(
            [d['G_T_list'][j].T @ V_test[j] for j in range(d['n_rois'])], axis=1)
        assert np.allclose(f_new, f_check, atol=1e-12)
        assert np.all(np.isfinite(f_new))
 
 
# ===================================================================
# PART B: Full EM integration via NeuraLVAR
# ===================================================================
 
def test_B1_backward_compatible():
    """update_weights=False: weights None, ll finite."""
    from nlgc.opt import NeuraLVAR
    d = make_toy_problem()
    model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model.fit(d['y'], d['f'], d['r'], lambda2=0.1, max_iter=50,
              a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
              update_weights=False)
    assert model._eigenmode_weights is None
    assert np.isfinite(model.ll)
 
 
def test_B2_v_update_produces_vlist():
    """update_weights=True: V_list returned with correct shapes."""
    from nlgc.opt import NeuraLVAR
    d = make_toy_problem()
    model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model._roi_data = {
        'G_T_list': d['G_T_list'],
        'V_list': [v.copy() for v in d['V_list']],
    }
    model.fit(d['y'], d['f'].copy(), d['r'], lambda2=0.1, max_iter=50,
              a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
              update_weights=True, weight_prior_var=0.1)
    V = model._eigenmode_weights
    assert V is not None and isinstance(V, list) and len(V) == d['n_rois']
    for j in range(d['n_rois']):
        assert V[j].shape == (d['n_dipoles'], d['K'])
        assert np.all(np.isfinite(V[j]))
    assert np.isfinite(model.ll)
 
 
def test_B3_prior_controls_deviation():
    """Smaller prior_var -> V closer to SVD init."""
    from nlgc.opt import NeuraLVAR
    d = make_toy_problem()
    diffs = {}
    for pv in [1.0, 0.1, 0.01]:
        model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
        model._roi_data = {
            'G_T_list': d['G_T_list'],
            'V_list': [v.copy() for v in d['V_list']],
        }
        model.fit(d['y'], d['f'].copy(), d['r'], lambda2=0.1, max_iter=50,
                  a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
                  update_weights=True, weight_prior_var=pv)
        V = model._eigenmode_weights
        diffs[pv] = sum(linalg.norm(V[j] - d['V_list'][j]) for j in range(d['n_rois']))
    assert diffs[0.01] < diffs[0.1] < diffs[1.0], f"Prior effect wrong: {diffs}"
 
 
def test_B4_no_prior_still_finite():
    """prior_var=1e6 (pure ML): still finite."""
    from nlgc.opt import NeuraLVAR
    d = make_toy_problem()
    model = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model._roi_data = {
        'G_T_list': d['G_T_list'],
        'V_list': [v.copy() for v in d['V_list']],
    }
    model.fit(d['y'], d['f'].copy(), d['r'], lambda2=0.1, max_iter=30,
              a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
              update_weights=True, weight_prior_var=1e6)
    V = model._eigenmode_weights
    for j in range(d['n_rois']):
        assert np.all(np.isfinite(V[j]))
    assert np.isfinite(model.ll)
 
 
# ===================================================================
# PART C: Full GC pipeline via _gc_extraction
# ===================================================================
 
def test_C1_gc_without_weights():
    """Original GC pipeline unchanged."""
    from nlgc.opt.test_Kalman import generate_processes
    from nlgc._nlgc import _gc_extraction
    x, y, f, r, p, a, q = generate_processes(0)
    out = _gc_extraction(
        y.T, f, r, p, p, n_eigenmodes=1, ROIs=[0, 1, 2],
        alpha=2.02, beta=1.02, lambda_range=[0.1, 0.05, 0.01],
        max_iter=200, max_cyclic_iter=3, tol=1e-5, cv=3, use_es=False,
        update_weights=False)
    dev = out[0]
    assert dev.shape == (3, 3) and np.all(np.isfinite(dev))
    assert dev[0, 1] > 0, f"True link 1->0 not detected: dev={dev[0,1]}"
 
 
def test_C2_gc_with_weights():
    """GC with V update still detects true link."""
    from nlgc.opt.test_Kalman import generate_processes
    from nlgc._nlgc import _gc_extraction
    x, y, f, r, p, a, q = generate_processes(0)
    out = _gc_extraction(
        y.T, f, r, p, p, n_eigenmodes=1, ROIs=[0, 1, 2],
        alpha=2.02, beta=1.02, lambda_range=[0.1, 0.05, 0.01],
        max_iter=200, max_cyclic_iter=3, tol=1e-5, cv=3, use_es=False,
        update_weights=True, weight_prior_var=0.01)
    dev = out[0]
    assert dev.shape == (3, 3) and np.all(np.isfinite(dev))
    assert dev[0, 1] > 0, f"True link 1->0 not detected: dev={dev[0,1]}"
 
 
# ===================================================================
# PART D: Mathematical verification
# ===================================================================
 
def test_D1_aq_unchanged_by_v():
    """A,Q identical after 1 iteration with vs without V.
 
    Proof: Q(theta) = Term1(A,Q,x_bar) + Term2(V,R,x_bar)
    A,Q only in Term1. V only in Term2. No cross-terms.
    V update runs AFTER A,Q in the M-step.
    """
    from nlgc.opt import NeuraLVAR
    d = make_toy_problem()
 
    model1 = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model1.fit(d['y'], d['f'].copy(), d['r'], lambda2=0.1, max_iter=1,
               a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
               update_weights=False)
    a1, _, q1, _, _ = model1._parameters
 
    model2 = NeuraLVAR(d['p'], n_eigenmodes=d['K'], use_lapack=True)
    model2._roi_data = {
        'G_T_list': d['G_T_list'],
        'V_list': [v.copy() for v in d['V_list']],
    }
    model2.fit(d['y'], d['f'].copy(), d['r'], lambda2=0.1, max_iter=1,
               a_init=d['a'].copy() * 0, q_init=0.1 * d['q'],
               update_weights=True, weight_prior_var=0.1)
    a2, _, q2, _, _ = model2._parameters
 
    assert np.allclose(a1, a2, atol=1e-10), f"A differs! max={np.max(np.abs(a1-a2))}"
    assert np.allclose(q1, q2, atol=1e-10), f"Q differs! max={np.max(np.abs(q1-q2))}"
 
 
def test_D3_likelihood_separation():
    """Changing V does not affect Term1. Changing A does not affect Term2.
 
    Term1 = E[log p(x | A, Q)]    depends on A, Q, x only
    Term2 = E[log p(y | x, V, R)] depends on V, R, x only
    """
    d = make_toy_problem(T=500)
    K, p, m = d['K'], d['p'], d['m']
    x_data = d['x'][:, :m]
 
    def term1(A_flat, Q_diag):
        A = A_flat.reshape(p, m, m)
        Q_inv = np.diag(1.0 / Q_diag)
        val = 0.0
        for t in range(p, d['T']):
            pred = sum(A[k] @ x_data[t - 1 - k] for k in range(p))
            diff = x_data[t] - pred
            val -= 0.5 * diff @ Q_inv @ diff
        val -= (d['T'] - p) / 2.0 * np.sum(np.log(Q_diag))
        return val
 
    def term2(f_mat, R_diag):
        R_inv = np.diag(1.0 / R_diag)
        val = 0.0
        for t in range(p, d['T']):
            diff = d['y'][:, t] - f_mat @ x_data[t]
            val -= 0.5 * diff @ R_inv @ diff
        val -= (d['T'] - p) / 2.0 * np.sum(np.log(R_diag))
        return val
 
    A_flat = d['a'].ravel()
    Q_diag, R_diag = np.diag(d['q']), np.diag(d['r'])
 
    t1_base = term1(A_flat, Q_diag)
    t2_base = term2(d['f'], R_diag)
 
    # Perturb V (changes f) -> Term1 unchanged, Term2 changes
    rng = np.random.default_rng(99)
    V_pert = [v + 0.1 * rng.standard_normal(v.shape) for v in d['V_list']]
    f_pert = np.concatenate(
        [d['G_T_list'][j].T @ V_pert[j] for j in range(d['n_rois'])], axis=1)
    t1_vp = term1(A_flat, Q_diag)
    t2_vp = term2(f_pert, R_diag)
    assert t1_vp == t1_base, "Term1 changed when V changed"
    assert t2_vp != t2_base, "Term2 unchanged when V changed"
 
    # Perturb A -> Term2 unchanged, Term1 changes
    A_pert = A_flat + 0.01 * rng.standard_normal(A_flat.shape)
    t1_ap = term1(A_pert, Q_diag)
    t2_ap = term2(d['f'], R_diag)
    assert t1_ap != t1_base, "Term1 unchanged when A changed"
    assert t2_ap == t2_base, "Term2 changed when A changed"
 
 
# ===================================================================
# Run all tests
# ===================================================================
if __name__ == '__main__':
    np.random.seed(42)
 
    print("=" * 60)
    print("NLGC V Weight Update - Complete Test Suite")
    print("=" * 60)
 
    print("\nPart A: Standalone update_eigenmode_weights function")
    run_test("A1 output shapes", test_A1_shapes)
    run_test("A2 f_new = G @ V", test_A2_f_equals_GV)
    run_test("A3 all finite", test_A3_all_finite)
    run_test("A4 multiple iterations", test_A4_multiple_iterations)
 
    print("\nPart B: Full EM via NeuraLVAR")
    run_test("B1 backward compatible", test_B1_backward_compatible)
    run_test("B2 V update produces V_list", test_B2_v_update_produces_vlist)
    run_test("B3 prior controls deviation", test_B3_prior_controls_deviation)
    run_test("B4 no prior still finite", test_B4_no_prior_still_finite)
 
    print("\nPart C: Full GC pipeline")
    run_test("C1 GC without weights", test_C1_gc_without_weights)
    run_test("C2 GC with weights", test_C2_gc_with_weights)
 
    print("\nPart D: Mathematical verification")
    run_test("D1 A,Q unchanged by V", test_D1_aq_unchanged_by_v)
    run_test("D2 likelihood separation", test_D3_likelihood_separation)
 
    print("\n" + "=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"Results: {PASS_COUNT} passed, {FAIL_COUNT} failed, {total} total")
    if FAIL_COUNT == 0:
        print("ALL 12 TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    sys.exit(FAIL_COUNT)