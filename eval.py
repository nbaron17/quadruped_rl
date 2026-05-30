"""
eval.py — Run a trained policy in the MuJoCo viewer.

Usage
-----
    python eval.py --model models/best/best_model.zip
    python eval.py --model models/ppo_quadruped_final.zip --episodes 5

The script loads the policy, sets a fixed forward command, and renders the
robot walking in real-time using the MuJoCo passive viewer.
"""

import os
import sys
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from env.quadruped_env import QuadrupedEnv
from config import MODEL_SAVE_DIR, EPISODE_MAX_STEPS, CONTROL_DT


def evaluate(args):
    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")

    # Wrap in a DummyVecEnv so VecNormalize can optionally be loaded
    def make_env():
        return QuadrupedEnv(render_mode="human")

    env = DummyVecEnv([make_env])

    # Try to load VecNormalize stats — check both naming conventions
    norm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not os.path.exists(norm_path):
        norm_path = model_path.replace(".zip", "vecnorm.pkl")
    if os.path.exists(norm_path):
        print(f"Loading VecNormalize stats: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = False   # freeze running stats during eval
        env.norm_reward = False
    else:
        print("No VecNormalize stats found — running without obs normalisation.")

    model = PPO.load(model_path, env=env)

    # ------------------------------------------------------------------ #
    # Evaluation loop
    # ------------------------------------------------------------------ #
    for ep in range(args.episodes):
        obs = env.reset()
        ep_return = 0.0
        step = 0

        print(f"\n--- Episode {ep + 1} ---")

        while True:
            # Deterministic=True → take the mode action (no exploration noise)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_return += float(reward[0])
            step += 1

            # Slow down to real-time (the viewer syncs asynchronously)
            time.sleep(CONTROL_DT * 0.9)

            if done[0]:
                print(f"  Steps: {step}  |  Return: {ep_return:.2f}")
                break

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained quadruped policy")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_SAVE_DIR, "best", "best_model"),
        help="Path to .zip model file",
    )
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    evaluate(args)
