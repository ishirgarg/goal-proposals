import jax
import jax.numpy as jnp
import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

from jaxgcrl.goal_proposers import MEGAProposer, ProposalContext, ValueCutoff
from jaxgcrl.goal_proposers.mega import chunked_kde_log_density, kde_log_density


def make_ctx(start_obs, ended, reached, value_fn, kde_samples):
    return ProposalContext(
        start_obs=start_obs,
        env_goals=jnp.zeros((start_obs.shape[0], 2)),
        episodes_ended=jnp.asarray(ended, jnp.float32),
        episodes_reached=jnp.asarray(reached, jnp.float32),
        sample_buffer_goals=lambda key, n: kde_samples[:n],
        value_fn=value_fn,
        policy_fn=None,
    )


@pytest.mark.parametrize("bandwidth", [0.1, 0.5])
def test_kde_log_density_matches_sklearn(bandwidth):
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((500, 3)).astype(np.float32)
    queries = rng.standard_normal((40, 3)).astype(np.float32)
    expected = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(samples).score_samples(queries)
    np.testing.assert_allclose(kde_log_density(queries, samples, bandwidth), expected, rtol=1e-4, atol=1e-4)


def test_chunked_kde_matches_unchunked(monkeypatch):
    from jaxgcrl.goal_proposers import mega

    rng = np.random.default_rng(1)
    samples = jnp.asarray(rng.standard_normal((300, 2)), jnp.float32)
    queries = jnp.asarray(rng.standard_normal((6, 7, 2)), jnp.float32)
    # force several chunks (and a chunk size that must shrink to divide num_envs)
    monkeypatch.setattr(mega, "_KDE_MAX_CHUNK_ELEMENTS", 4 * 7 * 300 * 2)
    chunked = chunked_kde_log_density(queries, samples, 0.1)
    full = kde_log_density(queries.reshape(-1, 2), samples, 0.1).reshape(6, 7)
    np.testing.assert_allclose(chunked, full, rtol=1e-5, atol=1e-5)


def test_selection_lowest_density_above_cutoff_else_highest_q():
    # 2 envs x 4 candidates on a line; the KDE samples are dense near 0
    proposer = MEGAProposer(num_candidates=4, kde_num_samples=64, cutoff=ValueCutoff(initial_cutoff=-5.0))
    candidates = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
    )
    kde_samples = jnp.concatenate([jnp.zeros((60, 2)), jnp.array([[3.0, 0.0]] * 4)])
    kde_samples = kde_samples + 0.01 * jax.random.normal(jax.random.PRNGKey(0), kde_samples.shape)
    # density order: 0 (highest) > 3 > 1, 2 (lowest, far from all samples)
    q = jnp.array(
        [
            [0.0, -1.0, -9.0, 0.0],  # env 0: candidate 2 (lowest density) is below cutoff -> pick 1
            [-9.0, -8.0, -7.0, -6.0],  # env 1: nothing viable -> highest q, candidate 3
        ]
    )
    ctx = make_ctx(jnp.zeros((2, 4)), [0, 0], [0, 0], lambda obs, goals: q, kde_samples)
    state = proposer.init(jax.random.PRNGKey(0), 2, 4, 2)
    goals, new_state, metrics = proposer.propose(state, candidates, ctx, jax.random.PRNGKey(1))
    np.testing.assert_allclose(goals, [[1.0, 0.0], [3.0, 0.0]])
    # no episode ended -> cutoff untouched
    assert float(new_state.cutoff) == -5.0
    np.testing.assert_allclose(metrics["goals/selected_value"], (-1.0 + -6.0) / 2)
    np.testing.assert_allclose(metrics["goals/frac_candidates_below_cutoff"], 5 / 8)

    # with the cutoff disabled, MEGA picks each env's lowest-density candidate, 2
    proposer = MEGAProposer(num_candidates=4, kde_num_samples=64, cutoff=ValueCutoff(initial_cutoff=-np.inf))
    goals, _, metrics = proposer.propose(proposer.init(None, 2, 4, 2), candidates, ctx, jax.random.PRNGKey(1))
    np.testing.assert_allclose(goals, [[2.0, 0.0], [2.0, 0.0]])
    assert "goals/cutoff" not in metrics and "goals/frac_candidates_below_cutoff" not in metrics


def test_propose_jits():
    proposer = MEGAProposer(num_candidates=5, kde_num_samples=32)
    num_envs = 3
    kde_samples = jax.random.normal(jax.random.PRNGKey(0), (32, 2))

    @jax.jit
    def run(state, candidates, key):
        ctx = make_ctx(
            jnp.zeros((num_envs, 4)),
            jnp.ones(num_envs),
            jnp.ones(num_envs),
            lambda obs, goals: -jnp.linalg.norm(goals, axis=-1),
            kde_samples,
        )
        return proposer.propose(state, candidates, ctx, key)

    candidates = jax.random.normal(jax.random.PRNGKey(1), (num_envs * 5, 2))
    goals, _, _ = run(proposer.init(None, 2, 4, 2), candidates, jax.random.PRNGKey(2))
    assert goals.shape == (num_envs, 2)
    # every chosen goal is one of that env's own candidates
    per_env = np.asarray(candidates).reshape(num_envs, 5, 2)
    for e in range(num_envs):
        assert any(np.allclose(goals[e], c) for c in per_env[e])
