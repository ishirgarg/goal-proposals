import logging
import os
import pickle
import pprint

import tyro
from brax.io import model

import wandb
from jaxgcrl.agents import CRL
from jaxgcrl.goal_proposers import (
    CandidateGoalProposer,
    EnvGoalProposer,
    MEGAProposer,
    UCriticProposer,
)
from jaxgcrl.utils.config import Config
from jaxgcrl.utils.env import MetricsRecorder, create_env


def main(config: Config):
    """Main function orchestrating the overall setup, initialization, and execution
    of training and evaluation processes. This function performs the following:
    1. Environment setup
    2. Directory creation for logging and checkpoints
    3. Training function creation
    4. Metrics recording
    5. Progress logging and monitoring
    6. Model saving and inference

    Creates the following directory structure:
        ./runs/
            run_{name}_s_{seed}/  # Run-specific directory
                args.pkl          # Saved command-line arguments
                ckpt/            # Model checkpoints
    Initializes wandb logging if enabled. Runs training with profiling and
    saves profiling results.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s|  %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    info = {**vars(config.run), **vars(config.agent)}

    utd_ratio = (
        config.run.num_envs
        * config.run.episode_length
        * config.agent.train_step_multiplier
        / config.agent.batch_size
    ) / (config.run.num_envs * config.agent.unroll_length)
    info["utd_ratio"] = utd_ratio
    info["agent"] = type(config.agent).__name__
    info["goal_proposer"] = type(config.goal_proposer).__name__
    info.update({f"goal_proposer/{k}": v for k, v in vars(config.goal_proposer).items()})

    if not isinstance(config.agent, CRL):
        assert isinstance(config.goal_proposer, EnvGoalProposer), (
            f"Goal proposers are only supported for CRL, not {type(config.agent).__name__}"
        )

    logging.info("Arguments:\n%s", pprint.pformat(info))

    wandb.init(
        project=config.run.wandb_project_name,
        group=config.run.wandb_group,
        name=config.run.exp_name,
        config=info,
        mode="online" if config.run.log_wandb else "disabled",
    )

    env = create_env(env_name=config.run.env, backend=config.run.backend)
    if config.run.eval_env:
        eval_env = create_env(env_name=config.run.eval_env, backend=config.run.backend)
    else:
        eval_env = env

    os.makedirs("./runs", exist_ok=True)
    run_dir = f"./runs/run_{config.run.exp_name}_s_{config.run.seed}"
    ckpt_dir = run_dir + "/ckpt"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(run_dir + "/args.pkl", "wb") as f:
        pickle.dump(vars(config), f)

    metrics_to_collect = [
        "eval/episode_dist",
        "eval/episode_reward",
        "eval/episode_reward_ctrl",
        "eval/episode_reward_dist",
        "eval/episode_reward_near",
        "eval/episode_reward_survive",
        "eval/episode_success",
        "eval/episode_success_any",
        "eval/episode_success_easy",
        "eval/episode_success_hard",
        "training/actor_loss",
        "training/log_alpha",
        "training/alpha_loss",
        "training/critic_loss",
        "training/entropy",
        "training/sps",
        "goals/intrinsic_success",
        "goals/episodes_per_env",
        "goals/random_action_frac",
    ]
    if isinstance(config.goal_proposer, CandidateGoalProposer):
        metrics_to_collect += ["goals/selected_value"]
        # missing metrics are recorded as 0, so only collect the cutoff's when it is enabled
        if config.goal_proposer.cutoff.enabled:
            metrics_to_collect += ["goals/cutoff", "goals/frac_candidates_below_cutoff"]
    if isinstance(config.goal_proposer, MEGAProposer):
        metrics_to_collect += [
            "mega/selected_log_density",
            "mega/candidate_log_density",
        ]
    if isinstance(config.goal_proposer, UCriticProposer):
        metrics_to_collect += [
            "ucritic/warm",
            "ucritic/selected_u",
            "ucritic/selected_u_std",
            "ucritic/candidate_u",
            "ucritic/agrees_with_mega",
            f"training/ucritic/{config.goal_proposer.u_target}_loss",
            "training/ucritic/buffer_entropy",
            "training/ucritic/novelty_std",
            "training/ucritic/valid_frac",
        ]

    metrics_recorder = MetricsRecorder(
        config.run.total_env_steps,
        metrics_to_collect,
        run_dir,
        config.run.exp_name,
        mode=config.run.wandb_mode,
    )

    train_kwargs = {}
    if isinstance(config.agent, CRL):
        train_kwargs["goal_proposer"] = config.goal_proposer
    _, params, _ = config.agent.train_fn(
        train_env=env,
        eval_env=eval_env,
        config=config.run,
        progress_fn=metrics_recorder.progress,
        **train_kwargs,
    )
    model.save_params(ckpt_dir + "/final", params)


def cli():
    tyro.cli(
        main,
        config=(
            tyro.conf.OmitArgPrefixes,
            tyro.conf.OmitSubcommandPrefixes,
            tyro.conf.ConsolidateSubcommandArgs,
        ),
    )


if __name__ == "__main__":
    cli()
