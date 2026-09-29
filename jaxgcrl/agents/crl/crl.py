import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import GoalCommandWrapper, TrajectoryIdWrapper
from jaxgcrl.goal_proposers import EnvGoalProposer, GoalProposer, ProposalContext, UpdateContext
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

from .losses import apply_logit_scale, energy_fn, update_actor_and_alpha, update_critic
from .networks import Actor, Encoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    alpha_state: TrainState


@dataclass
class GoalState:
    """Goal-proposal state carried through the training loop next to env_state"""

    proposer_state: Any
    # actor steps until the next proposal boundary
    steps_until_proposal: jnp.ndarray
    # goal metrics summed since the start of the epoch
    metrics: Dict[str, jnp.ndarray]


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@dataclass
class CRL:
    """Contrastive Reinforcement Learning (CRL) agent."""

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1

    disable_entropy_actor: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    unroll_length: int = 62
    h_dim: int = 512
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "bwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # probability that a training (not eval) action is replaced by U[-1, 1]
    random_action_prob: float = 0.1

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
        goal_proposer: GoalProposer = EnvGoalProposer(),
    ):
        self.check_config(config)

        unwrapped_env = train_env
        train_env = TrajectoryIdWrapper(train_env)
        train_env = envs.training.VmapWrapper(train_env)
        train_env = envs.training.EpisodeWrapper(
            train_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )
        train_env = GoalCommandWrapper(train_env, goal_proposer)

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = envs.training.wrap(
            eval_env,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = np.ceil(self.min_replay_size / self.unroll_length)
        num_training_steps_per_epoch = (config.total_env_steps - num_prefill_env_steps) // (
            config.num_evals * env_steps_per_actor_step
        )

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, sa_key, g_key, proposer_key = jax.random.split(
            key, 8
        )

        env_keys = jax.random.split(env_key, config.num_envs)
        env_state = jax.jit(train_env.reset)(env_keys)
        train_env.step = jax.jit(train_env.step)

        # Dimensions definitions and sanity checks
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Network setup
        # Actor
        actor = Actor(
            action_size=action_size,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
        )
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, np.ones([1, obs_size])),
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Critic
        sa_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, state_size + action_size]))
        g_encoder = Encoder(
            repr_dim=self.repr_dim,
            network_width=self.h_dim,
            network_depth=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
        )
        g_encoder_params = g_encoder.init(g_key, np.ones([1, goal_size]))
        critic_params = {"sa_encoder": sa_encoder_params, "g_encoder": g_encoder_params}
        if self.energy_fn == "cosine":
            # Learned temperature rescaling the (necessarily bounded, [-1, 1])
            # cosine similarity into a useful InfoNCE logit range. Initialized
            # to CLIP's initial value (1 / 0.07 =~ 14.3).
            critic_params["log_logit_scale"] = jnp.log(jnp.asarray(1 / 0.07, dtype=jnp.float32))
        critic_state = TrainState.create(
            apply_fn=None,
            params=critic_params,
            tx=optax.adam(learning_rate=self.critic_lr),
        )

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            actor_state=actor_state,
            critic_state=critic_state,
            alpha_state=alpha_state,
        )

        # Replay Buffer
        dummy_obs = jnp.zeros((obs_size,))
        dummy_action = jnp.zeros((action_size,))

        dummy_transition = Transition(
            observation=dummy_obs,
            action=dummy_action,
            reward=0.0,
            discount=0.0,
            extras={
                "state_extras": {
                    "truncation": 0.0,
                    "traj_id": 0.0,
                }
            },
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
                goal_indices=train_env.goal_indices,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        # Goal proposal
        num_candidates = goal_proposer.total_candidates(config.num_envs)
        proposal_interval = goal_proposer.proposal_interval_episodes * config.episode_length
        proposer_state = goal_proposer.init(proposer_key, goal_size, state_size, action_size)
        if goal_proposer.proposes:
            dummy_ctx = ProposalContext(
                start_obs=jnp.zeros((config.num_envs, obs_size)),
                env_goals=jnp.zeros((config.num_envs, goal_size)),
                episodes_ended=jnp.zeros((config.num_envs,)),
                episodes_reached=jnp.zeros((config.num_envs,)),
                sample_buffer_goals=lambda key, n: jnp.zeros((n, goal_size)),
                value_fn=lambda obs, goals: jnp.zeros(goals.shape[:2]),
                policy_fn=lambda states, goals, key: jnp.zeros(goals.shape[:-1] + (action_size,)),
            )
            proposal_metrics_shape = jax.eval_shape(
                lambda state, key: goal_proposer.propose(
                    state, jnp.zeros((num_candidates, goal_size)), dummy_ctx, key
                )[2],
                proposer_state,
                proposer_key,
            )
        else:
            proposal_metrics_shape = {}

        def zero_goal_metrics():
            metrics = {
                "goals/ended": 0.0,
                "goals/reached": 0.0,
                "goals/random_actions": 0.0,
                "goals/actions": 0.0,
                "goals/num_proposals": 0.0,
            }
            metrics = {k: jnp.zeros((), jnp.float32) for k in metrics}
            metrics.update({k: jnp.zeros(v.shape, v.dtype) for k, v in proposal_metrics_shape.items()})
            return metrics

        goal_state = GoalState(
            proposer_state=proposer_state,
            steps_until_proposal=jnp.zeros((), jnp.int32),  # propose at the first training step
            metrics=zero_goal_metrics(),
        )

        def goal_value(training_state, obs, goals):
            """Q(s0, g, pi(s0, g)) as the critic energy, for obs [E, obs_dim] and goals [E, N, goal_dim]."""
            num_envs, num_goals, _ = goals.shape
            state = jnp.repeat(obs[:, None, :state_size], num_goals, axis=1)
            state = state.reshape(num_envs * num_goals, state_size)
            goals = goals.reshape(num_envs * num_goals, goal_size)
            means, _ = actor.apply(
                training_state.actor_state.params, jnp.concatenate([state, goals], axis=-1)
            )
            action = nn.tanh(means)
            critic_params = training_state.critic_state.params
            sa_repr = sa_encoder.apply(critic_params["sa_encoder"], jnp.concatenate([state, action], axis=-1))
            g_repr = g_encoder.apply(critic_params["g_encoder"], goals)
            value = energy_fn(self.energy_fn, sa_repr, g_repr)
            value = apply_logit_scale({"energy_fn": self.energy_fn}, critic_params, value)
            return value.reshape(num_envs, num_goals)

        def deterministic_actor_step(training_state, env, env_state, extra_fields):
            means, _ = actor.apply(training_state.actor_state.params, env_state.obs)
            actions = nn.tanh(means)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                extras={"state_extras": state_extras},
            )

        def sample_actions(actor_params, obs, key, random_action_prob):
            """Training actions for obs [..., obs_dim]: a policy sample, replaced by U[-1, 1]
            with probability random_action_prob [...]. Returns (actions, is_random)."""
            means, log_stds = actor.apply(actor_params, obs)
            stds = jnp.exp(log_stds)
            noise_key, random_key, uniform_key = jax.random.split(key, 3)
            actions = nn.tanh(
                means + stds * jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
            )
            is_random = jax.random.uniform(random_key, random_action_prob.shape) < random_action_prob
            random_actions = jax.random.uniform(uniform_key, actions.shape, minval=-1.0, maxval=1.0)
            return jnp.where(is_random[..., None], random_actions, actions), is_random

        def behavior_policy(actor_params, states, goals, key):
            """Training actions for states [..., state_dim] commanded goals [..., goal_dim], with
            the base random-action probability (no go-explore boost)."""
            eps = jnp.full(states.shape[:-1], self.random_action_prob)
            return sample_actions(actor_params, jnp.concatenate([states, goals], axis=-1), key, eps)[0]

        def actor_step(actor_state, env, env_state, key, extra_fields):
            # exploration: with probability eps_i the action is U[-1, 1]; go-explore raises
            # eps_i with every step spent within the goal threshold this episode
            eps = jnp.clip(
                self.random_action_prob
                + goal_proposer.go_explore_eps_increment * env_state.info["goal_hits"],
                0.0,
                1.0,
            )
            actions, is_random = sample_actions(actor_state.params, env_state.obs, key, eps)

            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}

            return (
                nstate,
                Transition(
                    observation=env_state.obs,
                    action=actions,
                    reward=nstate.reward,
                    discount=1 - nstate.done,
                    extras={"state_extras": state_extras},
                ),
                is_random,
            )

        @functools.partial(jax.jit, static_argnames=("propose",))
        def get_experience(training_state, env_state, goal_state, buffer_state, key, propose):
            def propose_and_reset(env_state, goal_state, key):
                reset_key, candidate_key, propose_key = jax.random.split(key, 3)
                # fresh start states (and the env's own goals for them), below the auto-reset wrapper
                start = train_env.env.reset(jax.random.split(reset_key, config.num_envs))
                candidates = replay_buffer.sample_goals(buffer_state, candidate_key, num_candidates)
                ctx = ProposalContext(
                    start_obs=start.obs,
                    env_goals=start.obs[:, state_size:],
                    episodes_ended=env_state.info["episodes_ended"],
                    episodes_reached=env_state.info["episodes_reached"],
                    sample_buffer_goals=functools.partial(replay_buffer.sample_goals, buffer_state),
                    value_fn=functools.partial(goal_value, training_state),
                    policy_fn=functools.partial(behavior_policy, training_state.actor_state.params),
                )
                goals, proposer_state, proposal_metrics = goal_proposer.propose(
                    goal_state.proposer_state, candidates, ctx, propose_key
                )
                spec = jax.vmap(unwrapped_env.set_goal)(start, goals)
                env_state = train_env.command_goals(env_state, spec.pipeline_state, spec.obs, goals)

                metrics = dict(goal_state.metrics)
                for k, v in proposal_metrics.items():
                    metrics[k] = metrics[k] + v
                metrics["goals/num_proposals"] = metrics["goals/num_proposals"] + 1
                goal_state = goal_state.replace(
                    proposer_state=proposer_state,
                    steps_until_proposal=jnp.asarray(proposal_interval, jnp.int32),
                    metrics=metrics,
                )
                return env_state, goal_state

            def f(carry, unused_t):
                env_state, goal_state, current_key = carry
                current_key, next_key, proposal_key = jax.random.split(current_key, 3)
                if propose:
                    env_state, goal_state = jax.lax.cond(
                        goal_state.steps_until_proposal == 0,
                        propose_and_reset,
                        lambda env_state, goal_state, key: (env_state, goal_state),
                        env_state,
                        goal_state,
                        proposal_key,
                    )
                    goal_state = goal_state.replace(steps_until_proposal=goal_state.steps_until_proposal - 1)
                env_state, transition, is_random = actor_step(
                    training_state.actor_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                )
                metrics = dict(goal_state.metrics)
                metrics["goals/ended"] += jnp.sum(env_state.info["ended_now"])
                metrics["goals/reached"] += jnp.sum(env_state.info["ended_reached_now"])
                metrics["goals/random_actions"] += jnp.sum(is_random)
                metrics["goals/actions"] += is_random.shape[0]
                goal_state = goal_state.replace(metrics=metrics)
                return (env_state, goal_state, next_key), transition

            # the buffer is only written after the scan, so proposal candidates come from
            # the buffer as of the start of the unroll
            (env_state, goal_state, _), data = jax.lax.scan(
                f, (env_state, goal_state, key), (), length=self.unroll_length
            )

            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, goal_state, buffer_state

        def prefill_replay_buffer(training_state, env_state, goal_state, buffer_state, key):
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, goal_state, buffer_state, key = carry
                key, new_key = jax.random.split(key)
                # prefill never proposes: fresh env starts and goals
                env_state, goal_state, buffer_state = get_experience(
                    training_state,
                    env_state,
                    goal_state,
                    buffer_state,
                    key,
                    propose=False,
                )
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + env_steps_per_actor_step,
                )
                return (training_state, env_state, goal_state, buffer_state, new_key), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, goal_state, buffer_state, key),
                (),
                length=num_prefill_actor_steps,
            )[0]

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )

            networks = dict(
                actor=actor,
                sa_encoder=sa_encoder,
                g_encoder=g_encoder,
            )

            training_state, actor_metrics = update_actor_and_alpha(
                context, networks, transitions, training_state, actor_key
            )
            training_state, critic_metrics = update_critic(
                context, networks, transitions, training_state, critic_key
            )
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            metrics = {}
            metrics.update(actor_metrics)
            metrics.update(critic_metrics)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, goal_state, buffer_state, key):
            experience_key1, experience_key2, sampling_key, training_key, proposer_key = jax.random.split(
                key, 5
            )

            # update buffer
            env_state, goal_state, buffer_state = get_experience(
                training_state,
                env_state,
                goal_state,
                buffer_state,
                experience_key1,
                propose=goal_proposer.proposes,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # sample actor-step worth of trajectory windows
            buffer_state, trajectories = replay_buffer.sample(buffer_state)

            # process transitions for training
            batch_keys = jax.random.split(sampling_key, trajectories.observation.shape[0])
            transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
                (self.discounting, state_size, tuple(train_env.goal_indices)),
                trajectories,
                batch_keys,
            )
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
            )

            # permute transitions
            permutation = jax.random.permutation(experience_key2, len(transitions.observation))
            transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]),
                transitions,
            )

            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)

            # learned proposers train on the same trajectory windows (no-op for MEGA)
            update_ctx = UpdateContext(
                states=trajectories.observation[..., :state_size],
                actions=trajectories.action,
                achieved_goals=trajectories.observation[..., train_env.goal_indices],
                commanded_goals=trajectories.observation[..., state_size:],
                traj_ids=trajectories.extras["state_extras"]["traj_id"],
                sample_buffer_goals=functools.partial(replay_buffer.sample_goals, buffer_state),
                policy_fn=functools.partial(behavior_policy, training_state.actor_state.params),
            )
            proposer_state, proposer_metrics = goal_proposer.update(
                goal_state.proposer_state, update_ctx, proposer_key
            )
            goal_state = goal_state.replace(proposer_state=proposer_state)
            metrics.update(proposer_metrics)

            return (
                training_state,
                env_state,
                goal_state,
                buffer_state,
            ), metrics

        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            goal_state,
            buffer_state,
            key,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, gs, bs, k = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (
                        ts,
                        es,
                        gs,
                        bs,
                    ),
                    metrics,
                ) = training_step(ts, es, gs, bs, train_key)
                return (ts, es, gs, bs, k), metrics

            (training_state, env_state, goal_state, buffer_state, key), metrics = jax.lax.scan(
                f,
                (training_state, env_state, goal_state, buffer_state, key),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, goal_state, buffer_state, metrics

        def goal_metrics(summed):
            summed = jax.device_get(summed)
            metrics = {
                "goals/episodes_per_env": summed["goals/ended"] / config.num_envs,
                "goals/random_action_frac": summed["goals/random_actions"] / summed["goals/actions"],
            }
            # only episodes that truly ended count (boundary force-resets do not)
            if summed["goals/ended"] > 0:
                metrics["goals/intrinsic_success"] = summed["goals/reached"] / summed["goals/ended"]
            if summed["goals/num_proposals"] > 0:
                for k in proposal_metrics_shape:
                    metrics[k] = summed[k] / summed["goals/num_proposals"]
            return {k: float(v) for k, v in metrics.items()}

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, goal_state, buffer_state, _ = prefill_replay_buffer(
            training_state, env_state, goal_state, buffer_state, prefill_key
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            deterministic_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            goal_state = goal_state.replace(metrics=zero_goal_metrics())
            training_state, env_state, goal_state, buffer_state, metrics = training_epoch(
                training_state, env_state, goal_state, buffer_state, epoch_key
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
                **goal_metrics(goal_state.metrics),
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            if config.checkpoint_logdir:
                # Save current policy and critic params.
                params = (
                    training_state.alpha_state.params,
                    training_state.actor_state.params,
                    training_state.critic_state.params,
                )
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step

        params = (
            training_state.alpha_state.params,
            training_state.actor_state.params,
            training_state.critic_state.params,
        )

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics
