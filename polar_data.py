"""POLAR simulated DTR data generator.

Source: https://github.com/Papillon-Xiang/POLAR-blind (simulation/POLAR.py)
The functions below (phi, r_bar, get_transition, compute_Q_values,
compute_V_Pi_star, generate_offline_data) are copied verbatim or with light
adaptation from the POLAR repository, which accompanies the paper
"POLAR: A Pessimistic Model-based Policy Learning Algorithm for Dynamic
Treatment Regimes."

Setup recap (3-stage DTR):
    Trajectory: (s_1, a_1, s_2, a_2, s_3, a_3, s_4)
    s_k in [0, 1]^2 (continuous patient indicators)
    a_k in {0, 1}   (binary treatment decision)
    Reward only at final stage (k = k_max = 3).
    Behavior policy: optimal action with prob p, otherwise flipped.
"""

from __future__ import annotations

import numpy as np


DEFAULT_W_STARS = {
    1: np.array(
        [
            [0.4, 0.2, 0.0, 0.6, 0.0, -0.2],
            [0.4, 0.0, 0.2, 0.4, 0.2, 0.0],
        ]
    ),
    2: np.array(
        [
            [0.5, 0.1, -0.1, 0.5, -0.1, 0.1],
            [0.5, -0.1, 0.1, 0.5, 0.1, -0.1],
        ]
    ),
    3: np.array(
        [
            [0.6, -0.12, -0.08, 0.4, 0.08, 0.12],
            [0.6, -0.08, -0.12, 0.4, 0.12, 0.08],
        ]
    ),
}

DIM_S = np.array([2, 2, 2, 2])
DIM_S_HISTORY = np.cumsum(DIM_S)
DIM_H = DIM_S_HISTORY + np.arange(4)
K_MAX = 3


def r_bar(k, k_max, h, a, s_new, dim_h):
    if k < k_max:
        return np.zeros(h.shape[0])
    if k == k_max:
        s_k_1, s_k_2 = h[:, dim_h[k - 2] + 1], h[:, dim_h[k - 2] + 2]
        s_next_1, s_next_2 = s_new[:, 0], s_new[:, 1]
        reward = (
            np.cos(-np.pi * s_k_1)
            + 2 * np.cos(np.pi * s_k_2)
            + s_next_1
            + 2 * s_next_2
        )
        return (reward - 1.37) * 3.8
    raise ValueError("Invalid value for k.")


def phi(k, h, a, dim_h):
    m = h.shape[0]
    if k == 1:
        s_k = h
    elif k > 1:
        s_k = h[:, dim_h[k - 2] + 1 :]
    else:
        raise ValueError("Invalid value for k.")
    feature_matrix = np.concatenate(
        [
            (1 - a).reshape(m, 1),
            s_k * (1 - a)[:, np.newaxis],
            a.reshape(m, 1),
            s_k * a[:, np.newaxis],
        ],
        axis=1,
    )
    return feature_matrix


def r_true_expected(k, k_max, h, a, W_stars, dim_h):
    W_star = W_stars[k]
    expected_state = phi(k, h, a, dim_h) @ W_star.T
    return r_bar(k, k_max, h, a, expected_state, dim_h)


def get_transition(k, W, h, a, dim_h, dim_s, rng=None):
    rng = rng if rng is not None else np.random
    m = h.shape[0]
    mean = phi(k, h, a, dim_h) @ W.T
    noise = 0.8 * (rng.beta(2, 2, size=(m, dim_s[k])) - 0.5)
    return np.minimum(np.maximum(mean + noise, 0), 1)


def compute_Q_values(k, h_tuple, k_max, W_stars, dim_h, dim_s):
    h = np.array(h_tuple)
    m = h.shape[0]
    if k == k_max:
        Q0 = r_true_expected(k, k_max, h, np.zeros(m), W_stars, dim_h)
        Q1 = r_true_expected(k, k_max, h, np.ones(m), W_stars, dim_h)
        return Q0, Q1

    def compute_Q(a):
        s_next = get_transition(k, W_stars[k], h, a, dim_h, dim_s)
        r = r_bar(k, k_max, h, a, s_next, dim_h)
        h_next = np.concatenate((h, a.reshape(m, 1), s_next), axis=1)
        V_next, _ = compute_V_Pi_star(k + 1, tuple(map(tuple, h_next)), k_max, W_stars, dim_h, dim_s)
        return r + V_next

    Q0 = compute_Q(np.zeros(m))
    Q1 = compute_Q(np.ones(m))
    return Q0, Q1


def compute_V_Pi_star(k, h_tuple, k_max, W_stars, dim_h, dim_s):
    Q0, Q1 = compute_Q_values(k, h_tuple, k_max, W_stars, dim_h, dim_s)
    V = np.maximum(Q0, Q1)
    Pi = (Q0 < Q1).astype(int)
    return V, Pi


def optimal_action_stage1(s1, W_stars=None, dim_h=None, dim_s=None, k_max=K_MAX, n_mc=2000, seed=0):
    """Compute the (stochastic-MC-estimated) optimal action at stage 1 for each row of s1.

    The exact optimal policy at stage 1 requires integrating over downstream
    transition randomness. We Monte-Carlo this by averaging Q-values over
    n_mc resamples per input. Cached evaluation per unique s1 row keeps it cheap.
    """
    W_stars = W_stars or DEFAULT_W_STARS
    dim_h = DIM_H if dim_h is None else dim_h
    dim_s = DIM_S if dim_s is None else dim_s
    rng = np.random.default_rng(seed)
    m = s1.shape[0]
    Q0_acc = np.zeros(m)
    Q1_acc = np.zeros(m)
    for _ in range(n_mc):
        # Re-seed the global RNG so get_transition draws are different each MC pass.
        np.random.seed(rng.integers(0, 2**31 - 1))
        Q0, Q1 = compute_Q_values(1, tuple(map(tuple, s1)), k_max, W_stars, dim_h, dim_s)
        Q0_acc += Q0
        Q1_acc += Q1
    Q0_acc /= n_mc
    Q1_acc /= n_mc
    return (Q0_acc < Q1_acc).astype(int), Q0_acc, Q1_acc


def generate_offline_data(num_samples, p, k_max=K_MAX, W_stars=None, dim_h=None, dim_s=None, seed=None):
    """Generate offline DTR trajectories (n, 11): s1 a1 s2 a2 s3 a3 s4."""
    W_stars = W_stars or DEFAULT_W_STARS
    dim_h = DIM_H if dim_h is None else dim_h
    dim_s = DIM_S if dim_s is None else dim_s
    if seed is not None:
        np.random.seed(seed)

    histories = [np.random.uniform(0, 1, size=(num_samples, dim_s[0]))]
    actions = []

    for k in range(1, k_max + 1):
        h = histories[-1]
        h_tuple = tuple(map(tuple, h))
        _, Pi_star = compute_V_Pi_star(k, h_tuple, k_max, W_stars, dim_h, dim_s)
        u = np.random.binomial(1, p, size=num_samples)
        a = ((2 * Pi_star - 1) * (2 * u - 1) + 1) / 2
        actions.append(a)
        if k < k_max:
            s_next = get_transition(k, W_stars[k], h, a, dim_h, dim_s)
            h_next = np.concatenate((h, a.reshape(num_samples, 1), s_next), axis=1)
            histories.append(h_next)

    final_states = get_transition(k_max, W_stars[k_max], histories[-1], actions[-1], dim_h, dim_s)
    final_data = np.concatenate(
        (histories[-1], actions[-1].reshape(num_samples, 1), final_states), axis=1
    )
    return final_data


def generate_stage1_dataset(n=5000, p=0.95, n_mc=100, seed=0):
    """Convenience wrapper for the MVP: emits (s1, a1_optimal) pairs.

    We deliberately use the *true MC-estimated* optimal action as the
    label, ignoring p (per the user's request). p is kept in the API so
    you can later pivot to the behavior-policy label if desired.
    """
    rng = np.random.default_rng(seed)
    s1 = rng.uniform(0, 1, size=(n, DIM_S[0])).astype(np.float32)
    a_opt, Q0, Q1 = optimal_action_stage1(s1, n_mc=n_mc, seed=seed)
    return s1, a_opt.astype(np.int64), Q0.astype(np.float32), Q1.astype(np.float32)
