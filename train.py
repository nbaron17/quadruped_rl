"""
train.py — Train the quadruped policy using PPO (Stable-Baselines3).

Usage
-----
    python train.py                     # fresh training run
    python train.py --resume models/ppo_quadruped_500000_steps.zip
    python train.py --timesteps 5000000 --n_envs 8

The script will:
  1. Create N parallel training environments (vectorised).
  2. Instantiate a PPO agent with the hyperparameters from config.py.
  3. Log to TensorBoard — run `tensorboard --logdir logs/tensorboard` to watch.
  4. Save model checkpoints every SAVE_EVERY_STEPS steps.
  5. Run a periodic evaluation callback to track episode return.
"""

import os
import sys
import argparse

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from env.quadruped_env import QuadrupedEnv
from config import (
    MODEL_SAVE_DIR,
    TENSORBOARD_LOG_DIR,
    PPO_CONFIG,
    TOTAL_TIMESTEPS,
    SAVE_EVERY_STEPS,
    EVAL_EPISODES,
    N_ENVS,
)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int = 0):
    """
    Return a callable that creates a single training environment.

    Using a callable (not the env directly) is required by SubprocVecEnv so
    each subprocess can construct its own copy of the environment.
    """
    def _init():
        env = QuadrupedEnv(render_mode=None)   # no rendering during training
        env = Monitor(env)                      # wraps env to track episode stats
        env.reset(seed=seed + rank)
        return env
    return _init


def make_eval_env():
    """Single eval environment (no parallelism needed)."""
    env = QuadrupedEnv(render_mode=None)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    os.makedirs(MODEL_SAVE_DIR,      exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Vectorised training environments
    # ------------------------------------------------------------------ #
    # SubprocVecEnv runs each env in its own process — better CPU utilisation.
    # If you hit issues (Windows, debugger), swap to DummyVecEnv for debugging.
    print(f"Creating {args.n_envs} parallel training environments...")
    train_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])

    # VecNormalize tracks a running mean/std for observations AND rewards.
    # This greatly stabilises PPO on continuous-control tasks.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
    )

    # ------------------------------------------------------------------ #
    # Evaluation environment (un-normalised rewards for fair comparison)
    # ------------------------------------------------------------------ #
    eval_env = VecNormalize(
        SubprocVecEnv([make_eval_env]),
        norm_obs=True,
        norm_reward=False,   # keep raw rewards for evaluation
        clip_obs=10.0,
        training=False,      # don't update stats during eval
    )

    # ------------------------------------------------------------------ #
    # PPO model
    # ------------------------------------------------------------------ #
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=train_env,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            **PPO_CONFIG,
        )
        # Restore normalisation stats from alongside the checkpoint
        norm_path = args.resume.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            train_env = VecNormalize.load(norm_path, train_env.venv)
            print(f"Loaded VecNormalize stats from {norm_path}")
    else:
        print("Starting fresh training run...")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            tensorboard_log=TENSORBOARD_LOG_DIR,
            **PPO_CONFIG,
        )

    print(f"Policy network: {model.policy}")

    # ------------------------------------------------------------------ #
    # Callbacks
    # ------------------------------------------------------------------ #

    # Save a checkpoint (and VecNormalize stats) every N steps
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=MODEL_SAVE_DIR,
        name_prefix="ppo_quadruped",
        save_vecnormalize=True,
        verbose=1,
    )

    # Periodically evaluate the policy and save the best model
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best"),
        log_path=os.path.join(MODEL_SAVE_DIR, "eval_logs"),
        eval_freq=max(args.save_every // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    callbacks = [checkpoint_cb, eval_cb]

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    print(f"\nTraining for {args.timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}\n")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        progress_bar=True,
    )

    # ------------------------------------------------------------------ #
    # Save final model
    # ------------------------------------------------------------------ #
    final_path = os.path.join(MODEL_SAVE_DIR, "ppo_quadruped_final")
    model.save(final_path)
    train_env.save(final_path + "_vecnorm.pkl")
    print(f"\nFinal model saved to: {final_path}.zip")

    train_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train quadruped locomotion policy")
    parser.add_argument("--timesteps",    type=int,   default=TOTAL_TIMESTEPS)
    parser.add_argument("--n_envs",       type=int,   default=N_ENVS)
    parser.add_argument("--save_every",   type=int,   default=SAVE_EVERY_STEPS)
    parser.add_argument("--eval_episodes",type=int,   default=EVAL_EPISODES)
    parser.add_argument("--resume",       type=str,   default=None,
                        help="Path to .zip checkpoint to continue training")
    args = parser.parse_args()
    train(args)
