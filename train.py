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

import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
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
    CURRICULUM_PHASES,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class RewardTermCallback(BaseCallback):
    """Logs individual reward terms from info['reward_breakdown'] to TensorBoard."""

    def __init__(self, log_freq: int = 250, verbose: int = 0):
        super().__init__(verbose)
        self._log_freq = log_freq
        self._buffers: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            for k, v in info.get("reward_breakdown", {}).items():
                self._buffers.setdefault(k, []).append(float(v))

        if self.n_calls % self._log_freq == 0:
            for k, vals in self._buffers.items():
                self.logger.record(f"reward_terms/{k}", np.mean(vals))
            self._buffers.clear()

        return True


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int = 0, cmd_ranges: dict = None):
    def _init():
        env = QuadrupedEnv(render_mode=None, **(cmd_ranges or {}))
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_eval_env(cmd_ranges: dict = None):
    def _init():
        env = QuadrupedEnv(render_mode=None, **(cmd_ranges or {}))
        env = Monitor(env)
        return env
    return _init


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
    phase = CURRICULUM_PHASES.get(args.phase)
    if phase:
        cmd_ranges = dict(
            cmd_vx_range=phase["vx"],
            cmd_vy_range=phase["vy"],
            cmd_yaw_range=phase["yaw"],
        )
        print(f"Phase {args.phase}: vx={phase['vx']}  vy={phase['vy']}  yaw={phase['yaw']}")
    else:
        cmd_ranges = {}
        print("No phase selected — using config.py command ranges.")

    print(f"Creating {args.n_envs} parallel training environments...")
    train_env = SubprocVecEnv([make_env(i, cmd_ranges=cmd_ranges) for i in range(args.n_envs)])

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
        SubprocVecEnv([make_eval_env(cmd_ranges=cmd_ranges)]),
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
            device=args.device,
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

    class VecNormCheckpointCallback(BaseCallback):
        """Saves vecnorm stats alongside each checkpoint at the same frequency."""
        def __init__(self, save_freq: int, save_path: str, name_prefix: str):
            super().__init__()
            self._save_freq = save_freq
            self._save_path = save_path
            self._name_prefix = name_prefix

        def _on_step(self) -> bool:
            if self.n_calls % self._save_freq == 0:
                steps = self.num_timesteps
                path = os.path.join(
                    self._save_path, f"{self._name_prefix}_{steps}_steps_vecnorm.pkl"
                )
                self.training_env.save(path)
            return True

    vecnorm_checkpoint_cb = VecNormCheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1),
        save_path=MODEL_SAVE_DIR,
        name_prefix="ppo_quadruped",
    )

    # Periodically evaluate the policy and save the best model
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(MODEL_SAVE_DIR, "best"),
        log_path=os.path.join(MODEL_SAVE_DIR, "eval_logs"),
        eval_freq=max(args.eval_every // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # Save VecNormalize stats alongside best_model.zip whenever EvalCallback updates it
    class SaveVecNormCallback(BaseCallback):
        def __init__(self, eval_cb: EvalCallback, save_dir: str):
            super().__init__()
            self._eval_cb = eval_cb
            self._save_dir = save_dir
            self._last_best = None

        def _on_step(self) -> bool:
            if self._eval_cb.best_mean_reward != self._last_best:
                self._last_best = self._eval_cb.best_mean_reward
                path = os.path.join(self._save_dir, "best_model_vecnorm.pkl")
                self.training_env.save(path)
            return True

    save_vecnorm_cb = SaveVecNormCallback(eval_cb, os.path.join(MODEL_SAVE_DIR, "best"))
    reward_term_cb = RewardTermCallback()

    callbacks = [checkpoint_cb, vecnorm_checkpoint_cb, eval_cb, save_vecnorm_cb, reward_term_cb]

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
    parser.add_argument("--eval_episodes", type=int,  default=EVAL_EPISODES)
    parser.add_argument("--eval_every",    type=int,  default=100_000,
                        help="Evaluate every N global steps (default 100k)")
    parser.add_argument("--phase",          type=int,  default=None,
                        help="Curriculum phase 1-5 (overrides config command ranges)")
    parser.add_argument("--device",         type=str,  default="cpu",
                        help="Device to train on: cpu or cuda (default: cpu)")
    parser.add_argument("--resume",        type=str,  default=None,
                        help="Path to .zip checkpoint to continue training")
    args = parser.parse_args()
    train(args)
