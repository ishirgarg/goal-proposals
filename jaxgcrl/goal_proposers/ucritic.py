"""U-critic: MEGA's candidates, ranked by a learned novelty critic instead of density.

MEGA commands the lowest-density viable goal, which is the entropy-maximizing choice
only if commanded goals are always reached. This proposer instead learns

    U(s, a, g) = E[ sum_t gamma^t r(s_{t+1}) | s, a, then the policy commanded g ],
    r(s) = -log rho(s) - H(rho)   (standardized),

the novelty of everything the agent actually visits when commanding g (failed attempts
and go-explore included), and commands the goal with the highest pessimistic U(s0, a0, g)
among those passing the value cutoff (if enabled). r is the first variation of the
buffer entropy H(rho), so this greedily maximizes the buffer's entropy growth;
U(s0, a0, g) := -log rho(g) recovers MEGA.

U is learned in one of two ways (u_target):
  td: 1-step TD. The reward does not depend on g, so any buffer transition trains U for
      any goal: each sample is paired with the goal commanded when it was collected (SARSA
      target with the stored next action, which evaluates the behavior as it actually ran,
      go-explore included) or with a random buffer goal (target with a fresh policy action).
  mc: regression onto the discounted novelty actually collected from each step to the end
      of its episode, under the goal commanded then. No bootstrapping, so errors do not
      compound over the horizon, but only commanded goals get targets.
U only ranks goals; it never trains the agent.
"""

from dataclasses import dataclass
from typing import Any, Literal

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from .base import UpdateContext
from .candidates import CutoffState, argmax_viable, take_chosen
from .mega import MEGAProposer, normalized_kde_log_density


def returns_to_episode_end(rewards: jax.Array, same_episode: jax.Array, discount: float):
    """Discounted returns within trajectory windows.

    rewards, same_episode: [W, T-1], for the transitions t -> t+1 of W windows of T steps;
    same_episode[w, t] is whether s_{t+1} continues the episode of s_t. Returns
    (returns, complete), both [W, T-1]: the discounted sum of rewards from transition t to the
    end of its episode, and whether that end lies inside the window (else the return is
    truncated). Values at transitions with same_episode false are meaningless.
    """

    def backward(carry, x):
        next_return, next_complete, next_same = carry
        reward, same = x
        # the episode ends at s_{t+1} unless transition t+1 stays in it
        ret = reward + discount * next_same * next_return
        complete = ~next_same | next_complete
        return (ret, complete, same), (ret, complete)

    num_windows = rewards.shape[0]
    # past the last transition, the episode's continuation is unknown: incomplete
    init = (jnp.zeros(num_windows), jnp.zeros(num_windows, bool), jnp.ones(num_windows, bool))
    _, (returns, complete) = jax.lax.scan(backward, init, (rewards.T, same_episode.T), reverse=True)
    return returns.T, complete.T


class UNetwork(nn.Module):
    width: int
    depth: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = nn.swish(nn.LayerNorm()(nn.Dense(self.width)(x)))
        return nn.Dense(1)(x)[..., 0]


@flax.struct.dataclass
class UCriticState:
    cutoff: CutoffState
    # ensemble-stacked U parameters (leading axis: ensemble member)
    params: Any
    target_params: Any
    opt_state: Any
    num_updates: jax.Array


@dataclass(frozen=True)
class UCriticProposer(MEGAProposer):
    """U-critic goal proposer. Takes all of MEGA's arguments, plus:

    Args:
        u_ensemble_size: number of U networks; their spread is the pessimism penalty
        u_hidden_dim: U network width
        u_num_hidden: U network depth
        u_lr: U learning rate
        u_discount: U discount factor
        u_target: how U is learned: "td" (1-step TD) or "mc" (regression onto the novelty
            collected until the end of the episode); see the module docstring
        u_tau: Polyak rate of the U target networks ("td" only)
        u_batch_size: transitions per U gradient step
        u_updates_per_step: U gradient steps per agent training step
        u_mc_windows: buffer windows whose per-step novelty is computed for "mc" returns
        commanded_goal_prob: probability that a sample is paired with its own commanded goal
            rather than a random buffer goal ("td" only)
        pessimism: goals are ranked by mean(U) - pessimism * std(U) over the ensemble
        shortlist_frac: re-rank only this lowest-density fraction of the viable candidates
            (1 re-ranks all of them)
        warmup_updates: use MEGA's choice until U has taken this many gradient steps
    """

    u_ensemble_size: int = 5
    u_hidden_dim: int = 256
    u_num_hidden: int = 2
    u_lr: float = 3e-4
    u_discount: float = 0.99
    u_target: Literal["td", "mc"] = "td"
    u_tau: float = 0.005
    u_batch_size: int = 256
    u_updates_per_step: int = 64
    u_mc_windows: int = 64
    commanded_goal_prob: float = 0.5
    pessimism: float = 0.5
    shortlist_frac: float = 1.0
    warmup_updates: int = 5_000

    @property
    def _network(self) -> UNetwork:
        return UNetwork(width=self.u_hidden_dim, depth=self.u_num_hidden)

    @property
    def _optimizer(self) -> optax.GradientTransformation:
        return optax.adam(self.u_lr)

    def init(self, key, goal_dim, state_dim, action_dim):
        dummy_input = jnp.zeros((1, state_dim + action_dim + goal_dim))
        params = jax.vmap(self._network.init, in_axes=(0, None))(
            jax.random.split(key, self.u_ensemble_size), dummy_input
        )
        return UCriticState(
            cutoff=self.cutoff.init(),
            params=params,
            target_params=params,
            opt_state=self._optimizer.init(params),
            num_updates=jnp.zeros((), jnp.int32),
        )

    def u_values(self, params, states, actions, goals) -> jax.Array:
        """U(s, a, g) of every ensemble member: [ensemble_size, *leading dims]."""
        inputs = jnp.concatenate([states, actions, goals], axis=-1)
        return jax.vmap(self._network.apply, in_axes=(0, None))(params, inputs)

    def _novelty_rewards(self, ctx: UpdateContext, goals: jax.Array, valid: jax.Array, key):
        """Novelty -log rho of achieved goals [E, N, goal_dim] under the current buffer, centered
        by its mean over the valid entries (a sample estimate of the buffer entropy) and scaled
        by their std; a positive rescaling leaves the ranking of goals unchanged.
        Returns (rewards [E, N], buffer_entropy, novelty_std)."""
        num_valid = jnp.maximum(jnp.sum(valid), 1)
        kde_samples = ctx.sample_buffer_goals(key, self.kde_num_samples)
        novelty = -normalized_kde_log_density(goals, kde_samples, self.kde_bandwidth)
        buffer_entropy = jnp.sum(novelty * valid) / num_valid
        novelty_std = jnp.sqrt(jnp.sum((novelty - buffer_entropy) ** 2 * valid) / num_valid)
        return (novelty - buffer_entropy) / (novelty_std + 1e-6), buffer_entropy, novelty_std

    def _td_samples(self, ctx: UpdateContext, key):
        """(states, actions, goals, rewards, next_states, next_actions, valid) for 1-step TD,
        and stats: buffer entropy, novelty std, and the fraction of transitions with a target."""
        num_windows, window_length = ctx.traj_ids.shape
        num_samples = self.u_updates_per_step * self.u_batch_size
        window_key, time_key, kde_key, pairing_key, goal_key, action_key = jax.random.split(key, 6)

        # transitions t -> t+1 from the windows, masked where t+1 starts a new episode
        w = jax.random.randint(window_key, (num_samples,), 0, num_windows)
        t = jax.random.randint(time_key, (num_samples,), 0, window_length - 1)
        now, nxt = (lambda x: x[w, t]), (lambda x: x[w, t + 1])
        valid = now(ctx.traj_ids) == nxt(ctx.traj_ids)

        shape = (self.u_updates_per_step, self.u_batch_size)
        rewards, buffer_entropy, novelty_std = self._novelty_rewards(
            ctx, nxt(ctx.achieved_goals).reshape(shape + (-1,)), valid.reshape(shape), kde_key
        )

        # pair each sample with its commanded goal (next action: the one taken) or a
        # random buffer goal (next action: a fresh policy sample)
        commanded = jax.random.bernoulli(pairing_key, self.commanded_goal_prob, (num_samples,))
        random_goals = ctx.sample_buffer_goals(goal_key, num_samples)
        goals = jnp.where(commanded[:, None], now(ctx.commanded_goals), random_goals)
        next_actions = jnp.where(
            commanded[:, None],
            nxt(ctx.actions),
            ctx.policy_fn(nxt(ctx.states), random_goals, action_key),
        )
        samples = (
            now(ctx.states),
            now(ctx.actions),
            goals,
            rewards.reshape(-1),
            nxt(ctx.states),
            next_actions,
            valid,
        )
        return samples, (buffer_entropy, novelty_std, jnp.mean(valid))

    def _mc_samples(self, ctx: UpdateContext, key):
        """(states, actions, goals, returns, None, None, valid) for regression onto the
        discounted novelty collected until the end of the episode, under its commanded goal,
        and stats as for _td_samples (the valid fraction is over the windows' transitions)."""
        num_windows = ctx.traj_ids.shape[0]
        num_samples = self.u_updates_per_step * self.u_batch_size
        window_key, kde_key, sample_key = jax.random.split(key, 3)

        # novelty of every step of a subset of windows (the KDE is the expensive part)
        w = jax.random.choice(window_key, num_windows, (min(self.u_mc_windows, num_windows),), replace=False)
        traj_ids = ctx.traj_ids[w]
        same_episode = traj_ids[:, :-1] == traj_ids[:, 1:]
        rewards, buffer_entropy, novelty_std = self._novelty_rewards(
            ctx, ctx.achieved_goals[w, 1:], same_episode, kde_key
        )

        # a transition gets a target if it stays in its episode and that episode ends in the window
        returns, complete = returns_to_episode_end(rewards, same_episode, self.u_discount)
        valid = (same_episode & complete).reshape(-1)
        p = jnp.where(jnp.any(valid), valid / jnp.maximum(jnp.sum(valid), 1), 1.0 / valid.size)
        i = jax.random.choice(sample_key, valid.size, (num_samples,), p=p)
        take = lambda x: x[w, :-1].reshape((valid.size,) + x.shape[2:])[i]
        samples = (
            take(ctx.states),
            take(ctx.actions),
            take(ctx.commanded_goals),
            returns.reshape(-1)[i],
            None,
            None,
            valid[i],
        )
        return samples, (buffer_entropy, novelty_std, jnp.mean(valid))

    def update(self, state: UCriticState, ctx: UpdateContext, key):
        get_samples = self._td_samples if self.u_target == "td" else self._mc_samples
        samples, (buffer_entropy, novelty_std, valid_frac) = get_samples(ctx, key)
        batches = jax.tree_util.tree_map(
            lambda x: x.reshape((self.u_updates_per_step, self.u_batch_size) + x.shape[1:]), samples
        )

        def gradient_step(carry, batch):
            params, target_params, opt_state = carry
            states, actions, goals, rewards, next_states, next_actions, valid = batch
            if self.u_target == "td":
                targets = rewards + self.u_discount * self.u_values(
                    target_params, next_states, next_actions, goals
                )
            else:
                targets = rewards  # already the return

            def loss_fn(params):
                errors = self.u_values(params, states, actions, goals) - targets
                return jnp.sum(jnp.mean(errors**2, axis=0) * valid) / jnp.maximum(jnp.sum(valid), 1)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = self._optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            if self.u_target == "td":
                target_params = optax.incremental_update(params, target_params, self.u_tau)
            return (params, target_params, opt_state), loss

        (params, target_params, opt_state), losses = jax.lax.scan(
            gradient_step, (state.params, state.target_params, state.opt_state), batches
        )
        state = state.replace(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            num_updates=state.num_updates + self.u_updates_per_step,
        )
        metrics = {
            f"ucritic/{self.u_target}_loss": jnp.mean(losses),
            "ucritic/buffer_entropy": buffer_entropy,
            "ucritic/novelty_std": novelty_std,
            "ucritic/valid_frac": valid_frac,
        }
        return state, metrics

    def propose(self, state: UCriticState, candidates, ctx, key):
        kde_key, action_key = jax.random.split(key)
        candidates, q, viable, cutoff_state = self.filter_candidates(state.cutoff, candidates, ctx)
        log_density = self.log_density(candidates, ctx, kde_key)

        # the lowest-density viable candidates (all of them if shortlist_frac = 1)
        threshold = jnp.nanquantile(
            jnp.where(viable, log_density, jnp.nan), self.shortlist_frac, axis=1, keepdims=True
        )
        shortlist = viable & (log_density <= threshold)

        # U(s0, a0, g) for each candidate; start_obs is [state, env goal]
        goal_dim = candidates.shape[-1]
        start_states = jnp.repeat(ctx.start_obs[:, None, :-goal_dim], self.num_candidates, axis=1)
        start_actions = ctx.policy_fn(start_states, candidates, action_key)
        u = self.u_values(state.params, start_states, start_actions, candidates)
        u_mean, u_std = jnp.mean(u, axis=0), jnp.std(u, axis=0)

        u_choice = argmax_viable(u_mean - self.pessimism * u_std, shortlist, q)
        mega_choice = argmax_viable(-log_density, viable, q)
        warm = state.num_updates >= self.warmup_updates
        chosen = jnp.where(warm, u_choice, mega_choice)
        goals = take_chosen(candidates, chosen)

        metrics = self.metrics(cutoff_state, log_density, q, viable, chosen)
        metrics.update(
            {
                "ucritic/warm": warm.astype(jnp.float32),
                "ucritic/selected_u": jnp.mean(take_chosen(u_mean, chosen)),
                "ucritic/selected_u_std": jnp.mean(take_chosen(u_std, chosen)),
                "ucritic/candidate_u": jnp.mean(u_mean),
                # how often U picks MEGA's goal anyway (tracked during warmup too)
                "ucritic/agrees_with_mega": jnp.mean(u_choice == mega_choice),
            }
        )
        return goals, state.replace(cutoff=cutoff_state), metrics
