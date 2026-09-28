import jax
from brax.envs import PipelineEnv, State, Wrapper
from jax import numpy as jnp


class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


def _where_env(mask: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Per-env jnp.where for batched leaves: x where mask[env] else y."""
    mask = jnp.reshape(mask, mask.shape + (1,) * (x.ndim - mask.ndim))
    return jnp.where(mask, x, y)


class GoalCommandWrapper(Wrapper):
    """Auto-reset wrapper that lets a goal proposer command per-env goals.

    Replaces brax's AutoResetWrapper, which resets every env to the state cached
    at its first reset. Wraps a batched env: TrajectoryId -> Vmap -> Episode.

    Until `command_goals` is first called for an env, done envs are reset with a
    fresh `env.reset` (new start and env goal). After it, done envs are restored
    to the stored (start, goal) episode spec.

    Per-env info:
        commanded_goal: the env's current goal
        use_stored_reset: false until the first proposal
        reset_pipeline_state, reset_obs: the stored episode spec
        goal_hits: steps spent within the goal threshold this episode (go-explore counter)
        reached_this_episode: whether the goal was reached this episode
        episodes_ended, episodes_reached: episodes that ended (and reached their goal) under
            the stored spec since the last `command_goals`. Force-resets are not counted.
        ended_now, ended_reached_now: whether an episode ended on this step (and had reached
            its goal); counts every env, for logging
        reset_key: key for fresh resets
    """

    def __init__(self, env: Wrapper, goal_proposer):
        super().__init__(env)
        self.goal_proposer = goal_proposer
        if goal_proposer.on_goal_reached not in ("reset", "go_explore"):
            raise ValueError(f"Unknown on_goal_reached: {goal_proposer.on_goal_reached}")
        if goal_proposer.proposes and not hasattr(env.unwrapped, "set_goal"):
            raise ValueError(
                f"{type(goal_proposer).__name__} proposes goals, but {type(env.unwrapped).__name__} "
                "has no set_goal method."
            )

    def _env_goal(self, obs: jax.Array) -> jax.Array:
        return obs[..., self.env.state_dim :]

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        num_envs = rng.shape[0]
        zeros = jnp.zeros((num_envs,))
        false = jnp.zeros((num_envs,), dtype=bool)
        state.info.update(
            commanded_goal=self._env_goal(state.obs),
            use_stored_reset=false,
            reset_pipeline_state=state.pipeline_state,
            reset_obs=state.obs,
            goal_hits=zeros,
            reached_this_episode=false,
            episodes_ended=zeros,
            episodes_reached=zeros,
            ended_now=false,
            ended_reached_now=false,
            reset_key=jax.vmap(lambda k: jax.random.fold_in(k, 1))(rng),
        )
        return state

    def step(self, state: State, action: jax.Array) -> State:
        info = state.info
        # zero steps for envs that were done (so TrajectoryIdWrapper bumps traj_id)
        info["steps"] = jnp.where(state.done, jnp.zeros_like(info["steps"]), info["steps"])
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        info = state.info

        # every env reports success; after set_goal it is measured against the commanded goal
        reached = state.metrics["success"] > 0
        goal_hits = info["goal_hits"] + reached
        reached_this_episode = info["reached_this_episode"] | reached

        done = state.done
        if self.goal_proposer.on_goal_reached == "reset":
            done = jnp.where(reached, jnp.ones_like(done), done)
            info["truncation"] = jnp.where(reached, jnp.zeros_like(info["truncation"]), info["truncation"])
        ended = done > 0
        ended_reached = ended & reached_this_episode

        use_stored = info["use_stored_reset"]
        info["episodes_ended"] = info["episodes_ended"] + (ended & use_stored)
        info["episodes_reached"] = info["episodes_reached"] + (ended_reached & use_stored)
        info["ended_now"] = ended
        info["ended_reached_now"] = ended_reached

        # fresh resets for done envs without a stored spec; one batched env.reset, only if any need it
        def fresh_reset(reset_key):
            keys = jax.vmap(jax.random.split)(reset_key)
            fresh = self.env.reset(keys[:, 1])
            return fresh.pipeline_state, fresh.obs, keys[:, 0]

        fresh_pipeline_state, fresh_obs, info["reset_key"] = jax.lax.cond(
            jnp.any(ended & ~use_stored),
            fresh_reset,
            lambda reset_key: (state.pipeline_state, state.obs, reset_key),
            info["reset_key"],
        )
        reset_pipeline_state = jax.tree_util.tree_map(
            lambda stored, fresh: _where_env(use_stored, stored, fresh),
            info["reset_pipeline_state"],
            fresh_pipeline_state,
        )
        reset_obs = _where_env(use_stored, info["reset_obs"], fresh_obs)

        pipeline_state = jax.tree_util.tree_map(
            lambda r, s: _where_env(ended, r, s), reset_pipeline_state, state.pipeline_state
        )
        obs = _where_env(ended, reset_obs, state.obs)
        # a stored spec keeps its commanded goal; a fresh reset commands the new env goal
        info["commanded_goal"] = _where_env(
            ended & ~use_stored, self._env_goal(fresh_obs), info["commanded_goal"]
        )
        info["goal_hits"] = jnp.where(ended, jnp.zeros_like(goal_hits), goal_hits)
        info["reached_this_episode"] = reached_this_episode & ~ended

        return state.replace(pipeline_state=pipeline_state, obs=obs, done=done)

    def command_goals(
        self,
        state: State,
        spec_pipeline_state,
        spec_obs: jax.Array,
        goals: jax.Array,
    ) -> State:
        """Boundary force-reset: store each env's (start, goal) episode spec and
        reset every env into it. Episodes cut short here are not counted as ended."""
        info = dict(state.info)
        zeros = jnp.zeros_like(info["goal_hits"])
        false = jnp.zeros_like(info["reached_this_episode"])
        info.update(
            reset_pipeline_state=spec_pipeline_state,
            reset_obs=spec_obs,
            commanded_goal=goals,
            use_stored_reset=jnp.ones_like(info["use_stored_reset"]),
            # the next step starts a new trajectory (TrajectoryIdWrapper bumps traj_id)
            steps=jnp.zeros_like(info["steps"]),
            truncation=jnp.zeros_like(info["truncation"]),
            goal_hits=zeros,
            reached_this_episode=false,
            episodes_ended=zeros,
            episodes_reached=zeros,
        )
        return state.replace(
            pipeline_state=spec_pipeline_state, obs=spec_obs, done=jnp.zeros_like(state.done), info=info
        )
