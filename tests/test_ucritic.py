from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.goal_proposers import MEGAProposer, ProposalContext, UCriticProposer, UpdateContext, ValueCutoff

STATE_DIM, GOAL_DIM, ACTION_DIM = 4, 2, 2


def zero_policy(states, goals, key):
    return jnp.zeros(goals.shape[:-1] + (ACTION_DIM,))


@dataclass(frozen=True)
class FixedUProposer(UCriticProposer):
    """U ignores its parameters: every member's U is the goal's x coordinate, plus a
    per-candidate spread (+spread for half the ensemble, -spread for the other half)."""

    spread_at_x3: float = 0.0

    def u_values(self, params, states, actions, goals):
        x = goals[..., 0]
        spread = jnp.where(x == 3.0, self.spread_at_x3, 0.0)
        return jnp.stack([x + spread, x - spread])


def proposal_setup():
    """2 envs x 4 candidates at x = 0..3, as in test_mega: density order 0 > 3 > 1 > 2.
    Env 0: candidate 2 is below the cutoff (-5), so MEGA picks 1. Env 1: nothing is
    viable, so every rule falls back to the highest-q candidate, 0."""
    candidates = jnp.array([[x, 0.0] for x in range(4)] * 2)
    kde_samples = jnp.concatenate([jnp.zeros((60, 2)), jnp.array([[3.0, 0.0]] * 4)])
    kde_samples = kde_samples + 0.01 * jax.random.normal(jax.random.PRNGKey(0), kde_samples.shape)
    q = jnp.array([[0.0, -1.0, -9.0, 0.0], [-6.0, -9.0, -9.0, -9.0]])
    ctx = ProposalContext(
        start_obs=jnp.zeros((2, STATE_DIM + GOAL_DIM)),
        env_goals=jnp.zeros((2, GOAL_DIM)),
        episodes_ended=jnp.zeros(2),
        episodes_reached=jnp.zeros(2),
        sample_buffer_goals=lambda key, n: kde_samples[:n],
        value_fn=lambda obs, goals: q,
        policy_fn=zero_policy,
    )
    return candidates, ctx


def propose(proposer, warm):
    candidates, ctx = proposal_setup()
    state = proposer.init(jax.random.PRNGKey(0), GOAL_DIM, STATE_DIM, ACTION_DIM)
    if warm:
        state = state.replace(num_updates=jnp.int32(proposer.warmup_updates))
    return jax.jit(proposer.propose)(state, candidates, ctx, jax.random.PRNGKey(1))


PROPOSER_KWARGS = dict(
    num_candidates=4, kde_num_samples=64, cutoff=ValueCutoff(initial_cutoff=-5.0), u_ensemble_size=2
)


def test_warmup_uses_mega_choice():
    goals, _, metrics = propose(UCriticProposer(**PROPOSER_KWARGS), warm=False)
    candidates, ctx = proposal_setup()
    mega = MEGAProposer(num_candidates=4, kde_num_samples=64, cutoff=ValueCutoff(initial_cutoff=-5.0))
    mega_goals, _, _ = mega.propose(
        mega.init(None, GOAL_DIM, STATE_DIM, ACTION_DIM), candidates, ctx, jax.random.PRNGKey(1)
    )
    np.testing.assert_allclose(goals, mega_goals)
    np.testing.assert_allclose(goals, [[1.0, 0.0], [0.0, 0.0]])
    assert float(metrics["ucritic/warm"]) == 0.0


def test_ranks_viable_candidates_by_u():
    goals, _, metrics = propose(FixedUProposer(**PROPOSER_KWARGS), warm=True)
    # env 0: highest U among viable {0, 1, 3} is x = 3; env 1: highest-q fallback
    np.testing.assert_allclose(goals, [[3.0, 0.0], [0.0, 0.0]])
    assert float(metrics["ucritic/warm"]) == 1.0
    np.testing.assert_allclose(metrics["ucritic/agrees_with_mega"], 0.5)
    np.testing.assert_allclose(metrics["goals/selected_value"], (0.0 + -6.0) / 2)


def test_pessimism_penalizes_ensemble_disagreement():
    # x = 3 has mean U 3 but std 3, so with pessimism 1 it scores 0 < x = 1's score of 1
    goals, _, _ = propose(FixedUProposer(**PROPOSER_KWARGS, spread_at_x3=3.0, pessimism=1.0), warm=True)
    np.testing.assert_allclose(goals, [[1.0, 0.0], [0.0, 0.0]])
    goals, _, _ = propose(FixedUProposer(**PROPOSER_KWARGS, spread_at_x3=3.0, pessimism=0.0), warm=True)
    np.testing.assert_allclose(goals, [[3.0, 0.0], [0.0, 0.0]])


def test_disabled_cutoff_ranks_all_candidates_by_u():
    kwargs = {**PROPOSER_KWARGS, "cutoff": ValueCutoff(initial_cutoff=-np.inf)}
    goals, state, metrics = propose(FixedUProposer(**kwargs), warm=True)
    # every candidate is viable: both envs pick the highest-U one, x = 3
    np.testing.assert_allclose(goals, [[3.0, 0.0], [3.0, 0.0]])
    assert float(state.cutoff.cutoff) == -np.inf
    assert "goals/cutoff" not in metrics


def test_zero_shortlist_keeps_only_megas_choice():
    goals, _, metrics = propose(FixedUProposer(**PROPOSER_KWARGS, shortlist_frac=0.0), warm=True)
    np.testing.assert_allclose(goals, [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(metrics["ucritic/agrees_with_mega"], 1.0)


def update_context(traj_ids):
    """Windows moving along a line: state t = (t / T, w / W, 0, 0), achieving its first 2 dims."""
    num_windows, window_length = traj_ids.shape
    t = jnp.arange(window_length)[None, :] / window_length
    w = jnp.arange(num_windows)[:, None] / num_windows
    states = jnp.stack(jnp.broadcast_arrays(t, w, 0 * t, 0 * t), axis=-1)
    return UpdateContext(
        states=states,
        actions=jnp.zeros(traj_ids.shape + (ACTION_DIM,)),
        achieved_goals=states[..., :GOAL_DIM],
        commanded_goals=jnp.ones(traj_ids.shape + (GOAL_DIM,)),
        traj_ids=traj_ids,
        # deterministic "buffer" so the reward is a fixed function of the next state
        sample_buffer_goals=lambda key, n: jax.random.uniform(jax.random.PRNGKey(0), (n, GOAL_DIM)),
        policy_fn=zero_policy,
    )


UPDATE_KWARGS = dict(kde_num_samples=64, u_batch_size=32, u_updates_per_step=16, u_ensemble_size=2)


def test_update_learns_novelty_reward():
    # discount 0: U regresses onto the (smooth, with a wide kernel) novelty of the next state
    proposer = UCriticProposer(**UPDATE_KWARGS, u_discount=0.0, u_lr=1e-3, kde_bandwidth=0.5)
    ctx = update_context(jnp.zeros((4, 16)))
    state = proposer.init(jax.random.PRNGKey(0), GOAL_DIM, STATE_DIM, ACTION_DIM)
    update = jax.jit(proposer.update)
    losses = []
    for i in range(30):
        state, metrics = update(state, ctx, jax.random.PRNGKey(i))
        losses.append(float(metrics["ucritic/td_loss"]))
    assert int(state.num_updates) == 30 * 16
    assert float(metrics["ucritic/valid_frac"]) == 1.0
    assert losses[-1] < 0.25 * losses[0]


def test_update_ignores_transitions_across_episodes():
    proposer = UCriticProposer(**UPDATE_KWARGS)
    # every step is its own episode: no valid transition, so U must not move
    ctx = update_context(jnp.arange(4 * 16, dtype=jnp.float32).reshape(4, 16))
    state = proposer.init(jax.random.PRNGKey(0), GOAL_DIM, STATE_DIM, ACTION_DIM)
    new_state, metrics = jax.jit(proposer.update)(state, ctx, jax.random.PRNGKey(0))
    assert float(metrics["ucritic/valid_frac"]) == 0.0
    for old, new in zip(jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(new_state.params)):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("ensemble_size", [1, 3])
def test_init_shapes(ensemble_size):
    proposer = UCriticProposer(u_ensemble_size=ensemble_size)
    state = proposer.init(jax.random.PRNGKey(0), GOAL_DIM, STATE_DIM, ACTION_DIM)
    u = proposer.u_values(
        state.params, jnp.zeros((5, STATE_DIM)), jnp.zeros((5, ACTION_DIM)), jnp.zeros((5, GOAL_DIM))
    )
    assert u.shape == (ensemble_size, 5)
    assert int(state.num_updates) == 0
