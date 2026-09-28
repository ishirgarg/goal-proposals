"""Goal proposer interface.

A goal proposer is a frozen, hashable config dataclass (so it is static under
jit) with pure methods. Its state is a pytree carried through the training
loop by the agent.
"""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Literal, Tuple

import flax
import jax
import jax.numpy as jnp

ProposerState = Any
Metrics = Dict[str, jax.Array]


@flax.struct.dataclass
class ProposalContext:
    """Everything a proposer may look at when proposing goals.

    E = num_envs, N = candidates per env.
    """

    # [E, obs_dim] the start states the envs will be reset into
    start_obs: jax.Array
    # [E, goal_dim] the env's own goals for those starts
    env_goals: jax.Array
    # [E] episodes that ended (not force-reset) since the last proposal, under proposed goals
    episodes_ended: jax.Array
    # [E] ...of which reached their commanded goal at least once
    episodes_reached: jax.Array
    # (key, n) -> [n, goal_dim]: achieved goals sampled uniformly from the replay buffer
    sample_buffer_goals: Callable[[jax.Array, int], jax.Array] = flax.struct.field(pytree_node=False)
    # (obs [E, obs_dim], goals [E, N, goal_dim]) -> [E, N]: the agent's value of each goal from each start
    value_fn: Callable[[jax.Array, jax.Array], jax.Array] = flax.struct.field(pytree_node=False)


class GoalProposer:
    # False -> the agent compiles no proposal logic (the env's own goals are used)
    proposes: ClassVar[bool]
    # rollout behavior when the commanded goal is reached
    on_goal_reached: Literal["reset", "go_explore"]
    # random-action probability added per step spent within the goal threshold this episode
    go_explore_eps_increment: float
    # propose every `proposal_interval_episodes * episode_length` env steps
    proposal_interval_episodes: int

    def total_candidates(self, num_envs: int) -> int:
        """Total number M of buffer goals sampled as candidates per proposal (static)."""
        raise NotImplementedError

    def init(self, key: jax.Array, goal_dim: int) -> ProposerState:
        raise NotImplementedError

    def update(self, state: ProposerState, transitions: Any, key: jax.Array) -> Tuple[ProposerState, Metrics]:
        """Called every training step with the training batch (for learned proposers)."""
        return state, {}

    def propose(
        self, state: ProposerState, candidates: jax.Array, ctx: ProposalContext, key: jax.Array
    ) -> Tuple[jax.Array, ProposerState, Metrics]:
        """candidates: [M, goal_dim] -> goals: [num_envs, goal_dim]."""
        raise NotImplementedError


@dataclass(frozen=True)
class EnvGoalProposer(GoalProposer):
    """Use the env's own goals: every reset draws a fresh start and goal, and
    episodes end when the goal is reached."""

    proposes: ClassVar[bool] = False
    on_goal_reached: ClassVar[Literal["reset", "go_explore"]] = "reset"
    go_explore_eps_increment: ClassVar[float] = 0.0
    proposal_interval_episodes: ClassVar[int] = 1

    def total_candidates(self, num_envs: int) -> int:
        return 0

    def init(self, key, goal_dim):
        return ()

    def propose(self, state, candidates, ctx, key):
        return ctx.env_goals, state, {}


def zeros_like_metrics(metrics_shape: Metrics) -> Metrics:
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), metrics_shape)
