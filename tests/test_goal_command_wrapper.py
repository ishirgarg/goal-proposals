"""Tests for GoalCommandWrapper, on a tiny point-mass env so they run fast."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.envs.base import Env, State
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

from jaxgcrl.envs.wrappers import GoalCommandWrapper, TrajectoryIdWrapper
from jaxgcrl.goal_proposers import EnvGoalProposer

EPISODE_LENGTH = 10
NUM_ENVS = 4


class PointEnvWithoutSetGoal(Env):
    """2D point: obs = [pos, target]. Moves by `action`; success within 0.5 of the
    target; unhealthy (done) when pos[0] > 50."""

    state_dim = 2
    goal_indices = jnp.array([0, 1])

    def reset(self, rng):
        pos_key, target_key = jax.random.split(rng)
        pipeline_state = {
            "pos": jax.random.uniform(pos_key, (2,), minval=-1, maxval=1),
            "target": jax.random.uniform(target_key, (2,), minval=5, maxval=10),
        }
        zero = jnp.zeros(())
        return State(pipeline_state, self._obs(pipeline_state), zero, zero, {"success": zero})

    def step(self, state, action):
        pipeline_state = {**state.pipeline_state, "pos": state.pipeline_state["pos"] + action}
        dist = jnp.linalg.norm(pipeline_state["pos"] - pipeline_state["target"])
        success = (dist < 0.5).astype(jnp.float32)
        done = (pipeline_state["pos"][0] > 50).astype(jnp.float32)
        state.metrics.update(success=success)
        return state.replace(
            pipeline_state=pipeline_state, obs=self._obs(pipeline_state), reward=success, done=done
        )

    def _obs(self, pipeline_state):
        return jnp.concatenate([pipeline_state["pos"], pipeline_state["target"]])

    @property
    def observation_size(self):
        return 4

    @property
    def action_size(self):
        return 2

    @property
    def backend(self):
        return "none"


class PointEnv(PointEnvWithoutSetGoal):
    def set_goal(self, state, goal):
        pipeline_state = {**state.pipeline_state, "target": goal}
        return state.replace(pipeline_state=pipeline_state, obs=self._obs(pipeline_state))


@dataclass(frozen=True)
class GoExploreProposer(EnvGoalProposer):
    """Stand-in for a proposing proposer (only its rollout settings matter here)."""

    proposes: ClassVar[bool] = True
    on_goal_reached: ClassVar[str] = "go_explore"


def make_env(proposer, base=None):
    env = TrajectoryIdWrapper(base or PointEnv())
    env = VmapWrapper(env)
    env = EpisodeWrapper(env, EPISODE_LENGTH, 1)
    return GoalCommandWrapper(env, proposer)


def zero_action():
    return jnp.zeros((NUM_ENVS, 2))


def command(env, state, key, goals):
    """Force-reset every env into (fresh start, goals), as CRL does at a boundary."""
    start = env.env.reset(jax.random.split(key, NUM_ENVS))
    spec = jax.vmap(env.unwrapped.set_goal)(start, goals)
    return env.command_goals(state, spec.pipeline_state, spec.obs, goals), spec


def action_towards(state, goal):
    return goal - state.pipeline_state["pos"]


def test_requires_set_goal_when_proposing():
    with pytest.raises(ValueError, match="set_goal"):
        make_env(GoExploreProposer(), base=PointEnvWithoutSetGoal())
    # EnvGoals needs no set_goal
    make_env(EnvGoalProposer(), base=PointEnvWithoutSetGoal())


def test_time_limit_fresh_reset_bumps_traj_id():
    env = make_env(EnvGoalProposer())
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), NUM_ENVS))
    obs0 = state.obs
    step = jax.jit(env.step)
    for _ in range(EPISODE_LENGTH):
        state = step(state, zero_action())
    assert np.all(state.done == 1)
    # fresh start and fresh env goal
    assert not np.allclose(state.obs, obs0)
    np.testing.assert_allclose(state.info["commanded_goal"], state.obs[:, 2:])
    state = step(state, zero_action())
    # the first step after env.reset bumped traj_id to 1; this reset bumps it to 2
    np.testing.assert_array_equal(state.info["traj_id"], 2)
    np.testing.assert_array_equal(state.info["steps"], 1)


def test_reset_on_reach_fires():
    env = make_env(EnvGoalProposer())
    state = env.reset(jax.random.split(jax.random.PRNGKey(1), NUM_ENVS))
    target = state.obs[:, 2:]
    # env 0 jumps onto its goal, the others stay put
    action = zero_action().at[0].set(action_towards(state, target)[0])
    state = env.step(state, action)
    np.testing.assert_array_equal(state.done, [1, 0, 0, 0])
    np.testing.assert_array_equal(state.info["ended_now"], [True, False, False, False])
    np.testing.assert_array_equal(state.info["ended_reached_now"], [True, False, False, False])
    np.testing.assert_array_equal(state.info["truncation"], 0)
    # env 0 was reset to a fresh start with a new goal
    assert not np.allclose(state.obs[0, 2:], target[0])
    np.testing.assert_allclose(state.info["commanded_goal"][0], state.obs[0, 2:])
    np.testing.assert_array_equal(state.info["goal_hits"], 0)
    state = env.step(state, zero_action())
    np.testing.assert_array_equal(state.info["traj_id"], [2, 1, 1, 1])
    # EnvGoals episodes never count towards proposer statistics
    np.testing.assert_array_equal(state.info["episodes_ended"], 0)


def test_force_reset_installs_spec_and_bumps_traj_id():
    env = make_env(GoExploreProposer())
    state = env.reset(jax.random.split(jax.random.PRNGKey(2), NUM_ENVS))
    state = env.step(state, zero_action())
    goals = jnp.arange(NUM_ENVS * 2, dtype=jnp.float32).reshape(NUM_ENVS, 2) + 20
    state, spec = command(env, state, jax.random.PRNGKey(3), goals)

    np.testing.assert_allclose(state.obs, spec.obs)
    np.testing.assert_allclose(state.obs[:, 2:], goals)
    np.testing.assert_allclose(state.info["commanded_goal"], goals)
    np.testing.assert_allclose(state.info["reset_obs"], spec.obs)
    np.testing.assert_array_equal(state.info["use_stored_reset"], True)
    np.testing.assert_array_equal(state.done, 0)
    # force-reset episodes are not counted as ended
    np.testing.assert_array_equal(state.info["episodes_ended"], 0)

    state = env.step(state, zero_action())
    np.testing.assert_array_equal(state.info["traj_id"], 2)
    np.testing.assert_array_equal(state.info["ended_now"], False)


@pytest.mark.parametrize("reason", ["time_limit", "unhealthy"])
def test_mid_period_resets_restore_stored_spec(reason):
    env = make_env(GoExploreProposer())
    state = env.reset(jax.random.split(jax.random.PRNGKey(4), NUM_ENVS))
    goals = jnp.full((NUM_ENVS, 2), 30.0)
    state, spec = command(env, state, jax.random.PRNGKey(5), goals)
    step = jax.jit(env.step)

    num_steps = EPISODE_LENGTH if reason == "time_limit" else 1
    action = zero_action() if reason == "time_limit" else jnp.full((NUM_ENVS, 2), 100.0)
    for _ in range(num_steps):
        state = step(state, action)
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_allclose(state.obs, spec.obs)
    for k in ("pos", "target"):
        np.testing.assert_allclose(state.pipeline_state[k], spec.pipeline_state[k])
    np.testing.assert_allclose(state.info["commanded_goal"], goals)
    np.testing.assert_array_equal(state.info["episodes_ended"], 1)
    np.testing.assert_array_equal(state.info["episodes_reached"], 0)

    state = step(state, zero_action())
    np.testing.assert_array_equal(state.info["traj_id"], 2)  # force-reset + this reset


def test_go_explore_counter_and_stats():
    env = make_env(GoExploreProposer())
    state = env.reset(jax.random.split(jax.random.PRNGKey(6), NUM_ENVS))
    goals = jnp.full((NUM_ENVS, 2), 3.0)
    state, _ = command(env, state, jax.random.PRNGKey(7), goals)

    # env 0 reaches its goal and stays there: no reset, counter keeps growing
    state = env.step(state, zero_action().at[0].set(action_towards(state, goals)[0]))
    for _ in range(3):
        state = env.step(state, zero_action())
    np.testing.assert_array_equal(state.done, 0)
    np.testing.assert_array_equal(state.info["goal_hits"], [4, 0, 0, 0])
    np.testing.assert_array_equal(state.info["reached_this_episode"], [True, False, False, False])

    # it leaves the goal: the counter holds until the episode ends
    state = env.step(state, zero_action().at[0].set(jnp.array([-5.0, -5.0])))
    np.testing.assert_array_equal(state.info["goal_hits"], [4, 0, 0, 0])
    for _ in range(EPISODE_LENGTH - 5):
        state = env.step(state, zero_action())
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_array_equal(state.info["goal_hits"], 0)
    np.testing.assert_array_equal(state.info["reached_this_episode"], False)
    np.testing.assert_array_equal(state.info["episodes_ended"], 1)
    np.testing.assert_array_equal(state.info["episodes_reached"], [1, 0, 0, 0])
    np.testing.assert_array_equal(state.info["ended_reached_now"], [True, False, False, False])

    # a boundary zeroes the per-period statistics
    state, _ = command(env, state, jax.random.PRNGKey(8), goals)
    np.testing.assert_array_equal(state.info["episodes_ended"], 0)
    np.testing.assert_array_equal(state.info["episodes_reached"], 0)
