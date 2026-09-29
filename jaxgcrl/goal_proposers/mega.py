"""MEGA: Maximum Entropy Gain Exploration (Pitis et al., ICML 2020).

Goal selection follows the official implementation (spitis/mrl, curiosity.py
and density.py): among achieved-goal candidates from the replay buffer, drop
those whose value from the start state is below an adaptive cutoff (see
candidates.py), then pick the one with the lowest density under a KDE fit on
(normalized) buffer goals.
"""

import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .base import ProposalContext
from .candidates import CandidateGoalProposer, CutoffState, argmax_viable, take_chosen

# Upper bound on elements of the [chunk * N, K, goal_dim] KDE difference tensor
# held in memory at once (2**26 float32 = 256 MB).
_KDE_MAX_CHUNK_ELEMENTS = 2**26


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


def normalized_kde_log_density(queries: jax.Array, samples: jax.Array, bandwidth: float) -> jax.Array:
    """chunked_kde_log_density for queries [E, N, d], with queries and samples [K, d]
    normalized by the samples' mean and std."""
    mean = jnp.mean(samples, axis=0)
    std = jnp.std(samples, axis=0) + 1e-4
    return chunked_kde_log_density((queries - mean) / std, (samples - mean) / std, bandwidth)


@dataclass(frozen=True)
class MEGAProposer(CandidateGoalProposer):
    """MEGA goal proposer. Takes the candidate and cutoff arguments of
    CandidateGoalProposer, plus:

    Args:
        kde_num_samples: buffer goals the KDE is fit on at each proposal
        kde_bandwidth: Gaussian kernel bandwidth (on mean/std-normalized goals)
        on_goal_reached: rollout behavior when the commanded goal is reached
        go_explore_eps_increment: random-action probability added per step spent at the goal
        proposal_interval_episodes: propose every this many episodes
    """

    kde_num_samples: int = 10000
    kde_bandwidth: float = 0.1
    on_goal_reached: Literal["reset", "go_explore"] = "go_explore"
    go_explore_eps_increment: float = 0.1
    proposal_interval_episodes: int = 1

    def init(self, key, goal_dim, state_dim, action_dim):
        del key, goal_dim, state_dim, action_dim
        return self.cutoff.init()

    def log_density(self, candidates: jax.Array, ctx: ProposalContext, key: jax.Array) -> jax.Array:
        """KDE log-density [E, N] of per-env candidates [E, N, goal_dim], fit on
        (normalized) buffer goals at every call."""
        kde_samples = ctx.sample_buffer_goals(key, self.kde_num_samples)
        return normalized_kde_log_density(candidates, kde_samples, self.kde_bandwidth)

    def metrics(self, state: CutoffState, log_density, q, viable, chosen) -> dict:
        return {
            **self.candidate_metrics(state, q, viable, chosen),
            "mega/selected_log_density": jnp.mean(take_chosen(log_density, chosen)),
            "mega/candidate_log_density": jnp.mean(log_density),
        }

    def propose(self, state, candidates, ctx, key):
        candidates, q, viable, state = self.filter_candidates(state, candidates, ctx)
        log_density = self.log_density(candidates, ctx, key)

        # lowest-density candidate with q >= cutoff; if none qualify, the highest-q one
        chosen = argmax_viable(-log_density, viable, q)
        goals = take_chosen(candidates, chosen)
        return goals, state, self.metrics(state, log_density, q, viable, chosen)
