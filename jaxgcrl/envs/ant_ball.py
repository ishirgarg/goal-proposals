import os
import xml.etree.ElementTree as ET
from typing import Tuple

import jax
import mujoco
from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from jax import numpy as jnp

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py


class AntBall(PipelineEnv):
    """Ant pushing a ball to a target.

    With `ant_goal=True` (registered as the `ant_ball_4d` env) the goal is 4D, [ant xy, ball xy]: a
    second, non-colliding target body marks the ant's goal, and success requires the ant AND the
    ball to each be within `goal_reach_thresh` of their targets. The env's own goals put both
    targets on the same square.
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="generalized",
        dense_reward: bool = False,
        ant_goal: bool = False,
        **kwargs,
    ):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "assets", "ant_ball.xml")
        if ant_goal:
            sys = mjcf.loads(_add_ant_target(path))
        else:
            sys = mjcf.load(path)

        n_frames = 5

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(actuator=sys.actuator.replace(gear=200 * jnp.ones_like(sys.actuator.gear)))

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._object_idx = self.sys.link_names.index("object")
        self.dense_reward = dense_reward
        self.ant_goal = ant_goal
        # trailing non-ant q/qd entries: ball (2), [ant target (2)], ball target (2)
        self._num_extra_q = 6 if ant_goal else 4

        self.state_dim = 31
        # obs[0:2] is the ant (torso) xy, obs[29:31] the ball xy
        self.goal_indices = jnp.array([0, 1, 29, 30]) if ant_goal else jnp.array([29, 30])
        self.goal_reach_thresh = 0.5

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(rng1, (self.sys.q_size(),), minval=low, maxval=hi)
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        _, target, obj = self._random_target(rng)

        if self.ant_goal:
            # the env's own goal puts the ant target and the ball target on the same square
            q = q.at[-6:].set(jnp.concatenate([obj, target, target]))
        else:
            q = q.at[-4:].set(jnp.concatenate([obj, target]))

        qd = qd.at[-self._num_extra_q :].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jnp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
        }
        if self.ant_goal:
            metrics["ant_dist"] = zero
        state = State(pipeline_state, obs, reward, done, metrics)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        old_obs = self._get_obs(pipeline_state0)
        # Distance between goal and object
        old_dist = jnp.linalg.norm(old_obs[-2:] - old_obs[29:31])
        obs = self._get_obs(pipeline_state)
        dist = jnp.linalg.norm(obs[-2:] - obs[29:31])
        vel_to_target = (old_dist - dist) / self.dt
        success = jnp.array(dist < self.goal_reach_thresh, dtype=float)
        success_easy = jnp.array(dist < 2.0, dtype=float)
        if self.ant_goal:
            # the ant must also be at its own target
            ant_dist = jnp.linalg.norm(obs[-4:-2] - obs[0:2])
            success = success * (ant_dist < self.goal_reach_thresh)
            success_easy = success_easy * (ant_dist < 2.0)
            state.metrics.update(ant_dist=ant_dist)

        if self.dense_reward:
            reward = 10 * vel_to_target + healthy_reward - ctrl_cost - contact_cost
        else:
            reward = success

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        state.metrics.update(
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
        )
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def set_goal(self, state: State, goal: jax.Array) -> State:
        """Command a new goal (obs[goal_indices]: ball xy, or [ant xy, ball xy] with ant_goal) by
        moving the target(s) there."""
        q = state.pipeline_state.q.at[-len(self.goal_indices) :].set(goal)
        pipeline_state = self.pipeline_init(q, state.pipeline_state.qd)
        obs = self._get_obs(pipeline_state)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        # remove target and object q, qd
        qpos = pipeline_state.q[: -self._num_extra_q]
        qvel = pipeline_state.qd[: -self._num_extra_q]

        target_pos = pipeline_state.x.pos[-1][:2]
        if self.ant_goal:
            # the ant target body sits right before the ball target
            target_pos = jnp.concatenate([pipeline_state.x.pos[-2][:2], target_pos])

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        object_position = pipeline_state.x.pos[self._object_idx][:2]

        return jnp.concatenate([qpos] + [qvel] + [object_position] + [target_pos])

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a target and object location. Target is in a random position on a circle around ant.
        Object is in the middle between ant and target with small deviation."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        dist = 5
        ang = jnp.pi * 2.0 * jax.random.uniform(rng1)
        target_x = dist * jnp.cos(ang)
        target_y = dist * jnp.sin(ang)

        ang_obj = jnp.pi * 2.0 * jax.random.uniform(rng2)
        obj_x_offset = jnp.cos(ang_obj)
        obj_y_offset = jnp.sin(ang)

        target_pos = jnp.array([target_x, target_y])
        obj_pos = target_pos * 0.2 + jnp.array([obj_x_offset, obj_y_offset])
        return rng, target_pos, obj_pos


def _add_ant_target(path: str) -> str:
    """ant_ball.xml with a second, non-colliding target body for the ant goal, inserted right before
    the ball target (so its joints come right before the ball target's in q)."""
    tree = ET.parse(path)
    worldbody = tree.find(".//worldbody")
    target_idx = [i for i, body in enumerate(worldbody) if body.get("name") == "target"][0]
    ant_target = ET.Element("body", name="ant_target", pos="0 0 0.01")
    for axis, name in (("1 0 0", "ant_target_x"), ("0 1 0", "ant_target_y")):
        ET.SubElement(
            ant_target,
            "joint",
            armature="0",
            axis=axis,
            damping="0",
            limited="true",
            name=name,
            pos="0 0 0",
            range="-100 100",
            stiffness="0",
            type="slide",
        )
    ET.SubElement(
        ant_target,
        "geom",
        conaffinity="0",
        contype="0",
        name="ant_target",
        pos="0 0 0",
        size=".5",
        type="sphere",
        mass="1.0",
        rgba="0.2 0.4 1 0.5",
    )
    worldbody.insert(target_idx, ant_target)

    # brax's init_qpos (ant 15, ball 2, ball target 2) needs the ant target's 2 entries too
    init_qpos = tree.find(".//custom/numeric[@name='init_qpos']")
    data = init_qpos.get("data").split()
    init_qpos.set("data", " ".join(data[:-2] + ["0.0", "0.0"] + data[-2:]))
    return ET.tostring(tree.getroot())
