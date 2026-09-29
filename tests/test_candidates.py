import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgcrl.goal_proposers import CutoffState, ProposalContext, ValueCutoff


def make_ctx(ended, reached):
    return ProposalContext(
        start_obs=jnp.zeros((len(ended), 4)),
        env_goals=jnp.zeros((len(ended), 2)),
        episodes_ended=jnp.asarray(ended, jnp.float32),
        episodes_reached=jnp.asarray(reached, jnp.float32),
        sample_buffer_goals=None,
        value_fn=None,
        policy_fn=None,
    )


def _update(cutoff, state, q, ended, reached):
    return cutoff.update(state, jnp.asarray(q), make_ctx(ended, reached))


def test_initial_state():
    state = ValueCutoff().init()
    assert float(state.cutoff) == -6.0 and float(state.min_cutoff) == -6.0
    state = ValueCutoff(initial_cutoff=-7.0).init()
    assert float(state.cutoff) == -7.0 and float(state.min_cutoff) == -7.0
    state = ValueCutoff(initial_cutoff=-7.0, cutoff_floor=-3.0).init()
    assert float(state.cutoff) == -3.0 and float(state.min_cutoff) == -3.0


def test_initial_cutoff_must_not_exceed_max():
    with pytest.raises(ValueError, match="max_cutoff"):
        ValueCutoff(initial_cutoff=1.0, max_cutoff=0.0)


def test_cutoff_update_cases():
    cutoff = ValueCutoff(
        initial_cutoff=-7.0, max_cutoff=-4.5, cutoff_step=1.0, success_lo=0.3, success_hi=0.7
    )
    state = CutoffState(cutoff=jnp.float32(-7.0), min_cutoff=jnp.float32(-7.0))

    # no episode ended: skipped entirely (min_cutoff does not track q either)
    new = _update(cutoff, state, [[-20.0]], [0, 0], [0, 0])
    assert float(new.cutoff) == -7.0 and float(new.min_cutoff) == -7.0

    # high success: cutoff drops by a step, but not below min_cutoff = min(q) (no floor)
    new = _update(cutoff, state, [[-20.0]], [2, 2], [2, 1])
    assert float(new.min_cutoff) == -20.0 and float(new.cutoff) == -8.0
    new = _update(cutoff, state, [[-7.5]], [1, 0], [1, 0])
    assert float(new.min_cutoff) == -7.5 and float(new.cutoff) == -7.5

    # low success: cutoff rises by a step, above initial_cutoff but capped at max_cutoff
    low = CutoffState(cutoff=jnp.float32(-10.0), min_cutoff=jnp.float32(-20.0))
    new = _update(cutoff, low, [[-5.0]], [5, 5], [1, 1])
    assert float(new.cutoff) == -9.0 and float(new.min_cutoff) == -20.0
    new = _update(cutoff, state, [[-5.0]], [5, 5], [0, 0])
    assert float(new.cutoff) == -6.0
    near_max = CutoffState(cutoff=jnp.float32(-5.0), min_cutoff=jnp.float32(-20.0))
    new = _update(cutoff, near_max, [[-5.0]], [5, 5], [0, 0])
    assert float(new.cutoff) == -4.5

    # in between: unchanged
    new = _update(cutoff, low, [[-5.0]], [2, 2], [1, 1])
    assert float(new.cutoff) == -10.0

    # a floor bounds min_cutoff (and hence the cutoff) from below
    floored = ValueCutoff(initial_cutoff=-7.0, cutoff_floor=-7.2)
    new = _update(floored, state, [[-20.0]], [1, 0], [1, 0])
    assert float(new.min_cutoff) == pytest.approx(-7.2) and float(new.cutoff) == pytest.approx(-7.2)


def test_apply_filters_below_updated_cutoff():
    cutoff = ValueCutoff(initial_cutoff=-7.0, cutoff_step=1.0, success_lo=0.3)
    q = jnp.array([[-8.0, -6.5, -5.0], [-6.0, -9.0, 0.0]])
    # low success raises the cutoff to -6 before filtering
    viable, state = jax.jit(cutoff.apply)(cutoff.init(), q, make_ctx([2, 2], [0, 0]))
    assert float(state.cutoff) == -6.0
    np.testing.assert_array_equal(viable, [[False, False, True], [True, False, True]])


@pytest.mark.parametrize("cutoff_floor", [None, -3.0])
def test_negative_infinite_initial_cutoff_disables_filter(cutoff_floor):
    cutoff = ValueCutoff(initial_cutoff=-np.inf, cutoff_floor=cutoff_floor)
    assert not cutoff.enabled
    state = cutoff.init()
    q = jnp.array([[-1e9, -5.0, 0.0]])
    # neither extreme success rate moves the cutoff, and nothing is ever filtered
    for reached in ([0], [4]):
        viable, new_state = jax.jit(cutoff.apply)(state, q, make_ctx([4], reached))
        assert bool(jnp.all(viable))
        assert float(new_state.cutoff) == -np.inf and float(new_state.min_cutoff) == -np.inf
