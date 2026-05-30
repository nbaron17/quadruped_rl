"""
Watch a trained checkpoint in the viewer.
Ctrl+C to quit.

Usage:
    python watch_training.py                                  # latest checkpoint
    python watch_training.py --model models/best/best_model.zip
"""

import os, sys, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.quadruped_env import QuadrupedEnv
from config import MODEL_SAVE_DIR, CONTROL_DT, CURRICULUM_PHASES


def latest_checkpoint():
    zips = [
        f for f in os.listdir(MODEL_SAVE_DIR)
        if f.startswith("ppo_quadruped") and f.endswith(".zip")
    ]
    if not zips:
        raise FileNotFoundError("No checkpoints found in models/")
    zips.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_SAVE_DIR, f)))
    return os.path.join(MODEL_SAVE_DIR, zips[-1])


def main(args):
    model_path = args.model or latest_checkpoint()
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    print(f"Loading: {model_path}")

    phase = CURRICULUM_PHASES.get(args.phase)
    cmd_ranges = {}
    if phase:
        cmd_ranges = dict(
            cmd_vx_range=phase["vx"],
            cmd_vy_range=phase["vy"],
            cmd_yaw_range=phase["yaw"],
        )
        print(f"Phase {args.phase}: vx={phase['vx']}  vy={phase['vy']}  yaw={phase['yaw']}")

    env = DummyVecEnv([lambda: QuadrupedEnv(render_mode="human", **cmd_ranges)])

    norm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if not os.path.exists(norm_path):
        norm_path = model_path.replace(".zip", "vecnorm.pkl")
    if os.path.exists(norm_path):
        print(f"Loading VecNormalize: {norm_path}")
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("No VecNormalize stats found — observations will be un-normalised.")

    model = PPO.load(model_path, env=env)

    obs = env.reset()
    ep, steps = 1, 0
    vel_accum  = np.zeros(2)
    cmd_accum  = np.zeros(2)
    yaw_cmd_accum = 0.0
    yaw_act_accum = 0.0
    deterministic = not args.stochastic
    print(f"\nEpisode {ep}  ({'stochastic' if args.stochastic else 'deterministic'})")
    print(f"  {'cmd_vx':>8} {'cmd_vy':>8} {'act_vx':>8} {'act_vy':>8} {'cmd_yaw':>9} {'act_yaw':>9}")
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = env.step(action)
        steps += 1

        cmd     = info[0].get("cmd", [0, 0, 0])
        act     = info[0].get("actual_vel", [0, 0])
        act_yaw = info[0].get("actual_yaw_rate", 0.0)
        cmd_accum     += np.array(cmd[:2])
        vel_accum     += np.array(act[:2])
        yaw_cmd_accum += cmd[2]
        yaw_act_accum += act_yaw

        if steps % 50 == 0:
            print(f"  {cmd[0]:>8.2f} {cmd[1]:>8.2f} {act[0]:>8.2f} {act[1]:>8.2f} {cmd[2]:>9.2f} {act_yaw:>9.2f}")

        time.sleep(CONTROL_DT)
        if done[0]:
            avg_cmd = cmd_accum / steps
            avg_act = vel_accum / steps
            avg_yaw_cmd = yaw_cmd_accum / steps
            avg_yaw_act = yaw_act_accum / steps
            print(f"  --- Episode {ep} done ({steps} steps) ---")
            print(f"  avg cmd:    vx={avg_cmd[0]:.2f}  vy={avg_cmd[1]:.2f}  yaw={avg_yaw_cmd:.2f}")
            print(f"  avg actual: vx={avg_act[0]:.2f}  vy={avg_act[1]:.2f}  yaw={avg_yaw_act:.2f}")
            ep += 1
            steps = 0
            vel_accum     = np.zeros(2)
            cmd_accum     = np.zeros(2)
            yaw_cmd_accum = 0.0
            yaw_act_accum = 0.0
            obs = env.reset()
            print(f"\nEpisode {ep}  ({'stochastic' if args.stochastic else 'deterministic'})")
            print(f"  {'cmd_vx':>8} {'cmd_vy':>8} {'act_vx':>8} {'act_vy':>8} {'cmd_yaw':>9} {'act_yaw':>9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to .zip checkpoint (default: latest in models/)")
    parser.add_argument("--phase", type=int, default=None,
                        help="Curriculum phase 1-5 (sets command ranges to match training)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic actions (matches training behaviour)")
    main(parser.parse_args())
