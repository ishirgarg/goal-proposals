"""Shared machinery for proposers that pick each env's goal among achieved-goal
candidates sampled from the replay buffer: an adaptive value cutoff that filters out
candidates the agent is unlikely to reach, and helpers for selecting among the rest.

The cutoff follows MEGA's official implementation (spitis/mrl, curiosity.py) but is
independent of how the proposer ranks the surviving candidates.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Optional

import flax
import jax
import jax.numpy as jnp

from .base import GoalProposer, ProposalContext


@flax.struct.dataclass
class CutoffState:
    cutoff: jax.Array
    min_cutoff: jax.Array


@dataclass(frozen=True)
class ValueCutoff:
    """Adaptive value cutoff: candidates whose value from the start state is below it are
    filtered out, and it moves with the intrinsic success rate since the last proposal.

    Args:
        initial_cutoff: starting value of the cutoff; -inf disables the filter entirely
            (on the command line: `--initial_cutoff=-inf`, with the `=`)
        max_cutoff: ceiling of the cutoff (the strictest it can get); must be >= initial_cutoff.
            Unlike the official MEGA code, where initial_cutoff is both start and ceiling.
        cutoff_step: how much the cutoff moves per proposal
        success_lo: intrinsic success rate at or below which the cutoff rises (easier goals)
        success_hi: intrinsic success rate at or above which the cutoff drops (harder goals)
        cutoff_floor: hard lower limit on the cutoff; None means no floor
    """

    initial_cutoff: float = -6.0
    max_cutoff: float = 0.0
    cutoff_step: float = 1.0
    success_lo: float = 0.3
    success_hi: float = 0.7
    cutoff_floor: Optional[float] = None

    def __post_init__(self):
        if self.initial_cutoff > self.max_cutoff:
            raise ValueError(
                f"initial_cutoff ({self.initial_cutoff}) must be <= max_cutoff ({self.max_cutoff})"
            )

    @property
    def enabled(self) -> bool:
        return self.initial_cutoff > -math.inf

    @property
    def _floor(self) -> float:
        return -jnp.inf if self.cutoff_floor is None else self.cutoff_floor

    def init(self) -> CutoffState:
        cutoff = jnp.float32(self.initial_cutoff)
        if self.enabled:
            cutoff = jnp.maximum(cutoff, self._floor)
        return CutoffState(cutoff=cutoff, min_cutoff=cutoff)

    def update(self, state: CutoffState, q: jax.Array, ctx: ProposalContext) -> CutoffState:
        """MEGA's adaptive cutoff, from the intrinsic success rate of episodes
        that ended under proposed goals since the last proposal. Skipped if no
        such episode ended."""
        ended = jnp.sum(ctx.episodes_ended)
        reached = jnp.sum(ctx.episodes_reached)

        def update(state):
            rate = reached / ended
            min_cutoff = jnp.maximum(self._floor, jnp.minimum(state.min_cutoff, jnp.min(q)))
            cutoff = state.cutoff
            cutoff = jnp.where(
                rate >= self.success_hi, jnp.maximum(min_cutoff, cutoff - self.cutoff_step), cutoff
            )
            cutoff = jnp.where(
                rate <= self.success_lo,
                jnp.maximum(jnp.minimum(self.max_cutoff, cutoff + self.cutoff_step), self._floor),
                cutoff,
            )
            return CutoffState(cutoff=cutoff, min_cutoff=min_cutoff)

        return jax.lax.cond(ended > 0, update, lambda s: s, state)

    def apply(self, state: CutoffState, q: jax.Array, ctx: ProposalContext):
        """Update the cutoff, then mark the candidates with values q >= cutoff viable.
        When disabled, every candidate is viable and the state never changes."""
        if not self.enabled:
            return jnp.ones(q.shape, dtype=bool), state
        state = self.update(state, q, ctx)
        return q >= state.cutoff, state


def argmax_viable(score: jax.Array, viable: jax.Array, fallback: jax.Array) -> jax.Array:
    """Per-env index [E] of the highest-`score` viable candidate ([E, N] inputs); for envs
    with no viable candidate, the highest-`fallback` one."""
    best = jnp.argmax(jnp.where(viable, score, -jnp.inf), axis=1)
    return jnp.where(jnp.any(viable, axis=1), best, jnp.argmax(fallback, axis=1))


def take_chosen(x: jax.Array, chosen: jax.Array) -> jax.Array:
    """x [E, N, ...] at per-env indices chosen [E] -> [E, ...]."""
    chosen = chosen.reshape(chosen.shape + (1,) * (x.ndim - 1))
    return jnp.take_along_axis(x, chosen, axis=1)[:, 0]


@dataclass(frozen=True)
class CandidateGoalProposer(GoalProposer):
    """Base for proposers that pick each env's goal among `num_candidates` achieved goals
    from the buffer, after the value cutoff (if enabled) filters them.

    Args:
        num_candidates: achieved-goal candidates sampled from the buffer per env
        cutoff: the adaptive value cutoff; set initial_cutoff to -inf to disable it
    """

    proposes: ClassVar[bool] = True

    num_candidates: int = 100
    cutoff: ValueCutoff = ValueCutoff()

    def total_candidates(self, num_envs: int) -> int:
        return num_envs * self.num_candidates

    def filter_candidates(self, state: CutoffState, candidates: jax.Array, ctx: ProposalContext):
        """Candidates [M, goal_dim] -> per-env candidates [E, N, goal_dim], their values
        q [E, N] from the start states, which pass the cutoff, and the updated cutoff state."""
        num_envs = ctx.start_obs.shape[0]
        candidates = candidates.reshape(num_envs, self.num_candidates, -1)
        q = ctx.value_fn(ctx.start_obs, candidates)
        viable, state = self.cutoff.apply(state, q, ctx)
        return candidates, q, viable, state

    def candidate_metrics(self, state: CutoffState, q, viable, chosen) -> dict:
        metrics = {"goals/selected_value": jnp.mean(take_chosen(q, chosen))}
        if self.cutoff.enabled:
            metrics["goals/cutoff"] = state.cutoff
            metrics["goals/frac_candidates_below_cutoff"] = jnp.mean(~viable)
        return metrics
