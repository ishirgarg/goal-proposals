"""MEGA: Maximum Entropy Gain Exploration (Pitis et al., ICML 2020).

Goal selection follows the official implementation (spitis/mrl, curiosity.py
and density.py): among achieved-goal candidates from the replay buffer, drop
those whose value from the start state is below an adaptive cutoff, then pick
the one with the lowest density under a KDE fit on (normalized) buffer goals.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, Literal, Optional

import flax
import jax
import jax.numpy as jnp

from .base import GoalProposer, ProposalContext

# Upper bound on elements of the [chunk * N, K, goal_dim] KDE difference tensor
# held in memory at once (2**26 float32 = 256 MB).
_KDE_MAX_CHUNK_ELEMENTS = 2**26


@flax.struct.dataclass
class MEGAState:
    cutoff: jax.Array
    min_cutoff: jax.Array


def kde_log_density(queries: jax.Array, samples: jax.Array, bandwidth: float) -> jax.Array:
    """Gaussian KDE log-density of queries [Q, d] under samples [K, d].

    Matches sklearn.neighbors.KernelDensity(kernel="gaussian").score_samples.
    """
    num_samples, dim = samples.shape
    sq_dist = jnp.sum((queries[:, None, :] - samples[None, :, :]) ** 2, axis=-1)
    log_norm = math.log(num_samples) + 0.5 * dim * math.log(2 * math.pi * bandwidth**2)
    return jax.nn.logsumexp(-0.5 * sq_dist / bandwidth**2, axis=-1) - log_norm


def chunked_kde_log_density(queries: jax.Array, samples: jax.Array, bandwidth: float) -> jax.Array:
    """kde_log_density for queries [E, N, d], computed in env chunks with lax.map
    so the full [E * N, K] distance matrix is never materialized."""
    num_envs, num_queries, dim = queries.shape
    per_env = num_queries * samples.shape[0] * dim
    chunk = max(1, min(num_envs, _KDE_MAX_CHUNK_ELEMENTS // per_env))
    while num_envs % chunk:
        chunk -= 1
    chunked = queries.reshape(num_envs // chunk, chunk * num_queries, dim)
    log_density = jax.lax.map(lambda q: kde_log_density(q, samples, bandwidth), chunked)
    return log_density.reshape(num_envs, num_queries)


@dataclass(frozen=True)
class MEGAProposer(GoalProposer):
    """MEGA goal proposer.

    Args:
        num_candidates: achieved-goal candidates sampled from the buffer per env
        kde_num_samples: buffer goals the KDE is fit on at each proposal
        kde_bandwidth: Gaussian kernel bandwidth (on mean/std-normalized goals)
        initial_cutoff: starting value of the value cutoff
        max_cutoff: ceiling of the value cutoff (the strictest it can get); must be >= initial_cutoff.
            Unlike the official MEGA code, where initial_cutoff is both start and ceiling.
        cutoff_step: how much the cutoff moves per proposal
        success_lo: intrinsic success rate at or below which the cutoff rises (easier goals)
        success_hi: intrinsic success rate at or above which the cutoff drops (harder goals)
        cutoff_floor: hard lower limit on the cutoff; None means no floor
        on_goal_reached: rollout behavior when the commanded goal is reached
        go_explore_eps_increment: random-action probability added per step spent at the goal
        proposal_interval_episodes: propose every this many episodes
    """

    proposes: ClassVar[bool] = True

    num_candidates: int = 100
    kde_num_samples: int = 10000
    kde_bandwidth: float = 0.1
    initial_cutoff: float = -6.0
    max_cutoff: float = 0.0
    cutoff_step: float = 1.0
    success_lo: float = 0.3
    success_hi: float = 0.7
    cutoff_floor: Optional[float] = None
    on_goal_reached: Literal["reset", "go_explore"] = "go_explore"
    go_explore_eps_increment: float = 0.1
    proposal_interval_episodes: int = 1

    def __post_init__(self):
        if self.initial_cutoff > self.max_cutoff:
            raise ValueError(
                f"initial_cutoff ({self.initial_cutoff}) must be <= max_cutoff ({self.max_cutoff})"
            )

    @property
    def _floor(self) -> float:
        return -jnp.inf if self.cutoff_floor is None else self.cutoff_floor

    def total_candidates(self, num_envs: int) -> int:
        return num_envs * self.num_candidates

    def init(self, key, goal_dim):
        del key, goal_dim
        cutoff = jnp.maximum(jnp.float32(self.initial_cutoff), self._floor)
        return MEGAState(cutoff=cutoff, min_cutoff=cutoff)

    def update_cutoff(self, state: MEGAState, q: jax.Array, ctx: ProposalContext) -> MEGAState:
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
            return MEGAState(cutoff=cutoff, min_cutoff=min_cutoff)

        return jax.lax.cond(ended > 0, update, lambda s: s, state)

    def propose(self, state, candidates, ctx, key):
        num_envs = ctx.start_obs.shape[0]
        candidates = candidates.reshape(num_envs, self.num_candidates, -1)

        # KDE on normalized buffer goals, refit at every proposal
        kde_samples = ctx.sample_buffer_goals(key, self.kde_num_samples)
        mean = jnp.mean(kde_samples, axis=0)
        std = jnp.std(kde_samples, axis=0) + 1e-4
        log_density = chunked_kde_log_density(
            (candidates - mean) / std, (kde_samples - mean) / std, self.kde_bandwidth
        )

        q = ctx.value_fn(ctx.start_obs, candidates)
        state = self.update_cutoff(state, q, ctx)

        # lowest-density candidate with q >= cutoff; if none qualify, the highest-q one
        viable = q >= state.cutoff
        lowest_density = jnp.argmin(jnp.where(viable, log_density, jnp.inf), axis=1)
        highest_q = jnp.argmax(q, axis=1)
        chosen = jnp.where(jnp.any(viable, axis=1), lowest_density, highest_q)

        take = lambda x: jnp.take_along_axis(x, chosen[:, None], axis=1)[:, 0]
        goals = jnp.take_along_axis(candidates, chosen[:, None, None], axis=1)[:, 0]

        metrics = {
            "mega/cutoff": state.cutoff,
            "mega/frac_candidates_below_cutoff": jnp.mean(~viable),
            "mega/selected_log_density": jnp.mean(take(log_density)),
            "mega/candidate_log_density": jnp.mean(log_density),
            "mega/selected_value": jnp.mean(take(q)),
        }
        return goals, state, metrics
