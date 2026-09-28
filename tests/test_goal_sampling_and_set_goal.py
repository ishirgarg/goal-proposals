import jax
import jax.numpy as jnp
import numpy as np

import pytest
from brax.envs.wrappers.training import EpisodeWrapper, VmapWrapper

from jaxgcrl.agents.crl.crl import Transition
from jaxgcrl.envs.ant_ball import AntBall
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.wrappers import GoalCommandWrapper, TrajectoryIdWrapper
from jaxgcrl.goal_proposers import MEGAProposer
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue


def test_sample_goals_only_returns_valid_slots():
    num_envs, obs_size, max_size, unroll = 3, 5, 20, 4
    dummy = Transition(
        observation=jnp.zeros((obs_size,)),
        action=jnp.zeros((2,)),
        reward=0.0,
        discount=0.0,
        extras={"state_extras": {"truncation": 0.0, "traj_id": 0.0}},
    )
    buffer = TrajectoryUniformSamplingQueue(
        max_replay_size=max_size,
        dummy_data_sample=dummy,
        sample_batch_size=4,
        num_envs=num_envs,
        episode_length=unroll,
        goal_indices=jnp.array([1, 3]),
    )
    buffer_state = buffer.init(jax.random.PRNGKey(0))

    def insert(buffer_state, start):
        # observation[1] = 1000 * env + time and observation[3] = -that, so each slot is identifiable
        t = jnp.arange(start, start + unroll, dtype=jnp.float32)[:, None]
        e = jnp.arange(num_envs, dtype=jnp.float32)[None, :]
        tag = 1000 * e + t
        obs = jnp.zeros((unroll, num_envs, obs_size)).at[..., 1].set(tag).at[..., 3].set(-tag)
        data = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x, (unroll, num_envs) + jnp.shape(x)), dummy
        )._replace(observation=obs)
        return buffer.insert(buffer_state, data)

    buffer_state = insert(buffer_state, 1)
    before = jax.tree_util.tree_map(np.asarray, buffer_state)
    goals = np.asarray(buffer.sample_goals(buffer_state, jax.random.PRNGKey(1), 2000))
    # buffer_state is not modified
    jax.tree_util.tree_map(np.testing.assert_array_equal, before, buffer_state)

    assert goals.shape == (2000, 2)
    np.testing.assert_array_equal(goals[:, 1], -goals[:, 0])
    time, env = goals[:, 0] % 1000, goals[:, 0] // 1000
    # never an empty (all-zero) slot; every valid (time, env) slot is hit
    assert set(time.astype(int)) == set(range(1, 1 + unroll))
    assert set(env.astype(int)) == set(range(num_envs))
    assert len(set(goals[:, 0].astype(int))) == unroll * num_envs


def test_ant_maze_set_goal():
    env = AntMaze(maze_layout_name="u_maze", backend="spring")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    torso = state.obs[:2]
    goal = torso + jnp.array([0.2, -0.1])  # within goal_reach_thresh of the torso

    new_state = jax.jit(env.set_goal)(state, goal)
    np.testing.assert_allclose(new_state.obs[-2:], goal, atol=1e-5)
    # the rest of the observation (the ant itself) is unchanged
    np.testing.assert_allclose(new_state.obs[:-2], state.obs[:-2], atol=1e-5)
    np.testing.assert_allclose(new_state.obs[env.goal_indices], torso)

    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size)
    assert float(step(state, zero).metrics["success"]) == 0.0
    assert float(step(new_state, zero).metrics["success"]) == 1.0

    far = jax.jit(env.set_goal)(state, torso + jnp.array([3.0, 0.0]))
    assert float(step(far, zero).metrics["success"]) == 0.0
    np.testing.assert_allclose(step(far, zero).metrics["dist"], 3.0, atol=0.1)


def test_ant_ball_set_goal():
    env = AntBall(backend="spring")
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    ball = state.obs[env.goal_indices]
    goal = ball + jnp.array([0.2, -0.1])  # within goal_reach_thresh of the ball

    new_state = jax.jit(env.set_goal)(state, goal)
    np.testing.assert_allclose(new_state.obs[-2:], goal, atol=1e-5)
    # the ant and the ball are unchanged
    np.testing.assert_allclose(new_state.obs[:-2], state.obs[:-2], atol=1e-5)
    np.testing.assert_allclose(new_state.obs[env.goal_indices], ball, atol=1e-5)

    step = jax.jit(env.step)
    zero = jnp.zeros(env.action_size)
    assert float(step(new_state, zero).metrics["success"]) == 1.0

    far = jax.jit(env.set_goal)(state, ball + jnp.array([3.0, 0.0]))
    assert float(step(far, zero).metrics["success"]) == 0.0
    np.testing.assert_allclose(step(far, zero).metrics["dist"], 3.0, atol=0.1)


@pytest.mark.parametrize("env_cls", [AntMaze, AntBall])
def test_commanded_goal_persists_across_steps_and_resets(env_cls):
    """The goal written by set_goal is the one used at every step, including after
    mid-period resets; env.step never restores the env's own goal."""
    num_envs, episode_length = 4, 20
    base = env_cls(backend="spring")
    env = TrajectoryIdWrapper(base)
    env = EpisodeWrapper(VmapWrapper(env), episode_length, 1)
    env = GoalCommandWrapper(env, MEGAProposer())

    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), num_envs))
    start = env.env.reset(jax.random.split(jax.random.PRNGKey(1), num_envs))
    goals = start.obs[:, base.goal_indices] + jnp.array([1.5, -2.5])
    assert np.all(np.abs(np.asarray(goals - start.obs[:, base.state_dim :])) > 1e-3)
    spec = jax.vmap(base.set_goal)(start, goals)
    state = env.command_goals(state, spec.pipeline_state, spec.obs, goals)

    step = jax.jit(env.step)
    key = jax.random.PRNGKey(2)
    for _ in range(2 * episode_length + 5):  # crosses two time-limit resets
        key, action_key = jax.random.split(key)
        np.testing.assert_allclose(state.obs[:, base.state_dim :], goals, atol=1e-5)
        action = jax.random.uniform(action_key, (num_envs, base.action_size), minval=-1, maxval=1)
        state = step(state, action)
        np.testing.assert_allclose(state.obs[:, base.state_dim :], goals, atol=1e-5)
        np.testing.assert_allclose(state.pipeline_state.q[:, -2:], goals, atol=1e-5)
        np.testing.assert_allclose(state.info["commanded_goal"], goals)
