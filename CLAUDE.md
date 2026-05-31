# CLAUDE.md — Quadruped RL Project

This file tells Claude Code how to work with this project.

## Project Overview

Reinforcement learning locomotion for a 12-DOF quadruped robot using MuJoCo + PPO.
The robot learns to walk by tracking velocity commands (vx, vy, yaw rate).

## Tech Stack

- Python 3.14 (Ubuntu, primary training machine)
- MuJoCo 3.x — uses official Unitree A1 from mujoco_menagerie (not the generated model)
- Gymnasium 1.x
- Stable-Baselines3 2.x (PPO)
- PyTorch cu126 (CUDA 13.2, GTX 1660 Ti) — but training runs on CPU (MLP policy, GPU gives no benefit)
- TensorBoard

## Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Clone the real A1 model (already done — lives at ../mujoco_menagerie/unitree_a1/)
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git ../mujoco_menagerie
cd ../mujoco_menagerie && git sparse-checkout set unitree_a1
```

## Key Files — Where Things Live

| File | Purpose |
|------|---------|
| `config.py` | **Single source of truth** for ALL parameters — reward weights, obs scales, PPO hyperparams, command ranges, paths |
| `env/quadruped_env.py` | Gymnasium environment — observation, action, step, reset, termination |
| `env/rewards.py` | Each reward term as an independent function — edit here to shape behaviour |
| `env/controller.py` | Position controller — writes desired joint positions to `data.ctrl` (A1 uses built-in position actuators, Kp=100) |
| `env/commands.py` | Velocity command sampler — random during training, user-driven at inference |
| `train.py` | PPO training script — SubprocVecEnv, VecNormalize, checkpoints, TensorBoard |
| `eval.py` | Load checkpoint and visualise in MuJoCo passive viewer |
| `watch_training.py` | Watch a checkpoint mid-training — shows commanded vs actual velocity live |
| `view.py` | Simple viewer — `python view.py` (A1 random actions) or `python view.py --dog` (custom dog model) |
| `joystick_control.py` | Real-time keyboard/gamepad control of the **SB3 A1** policy |

### JAX/MJX Dog Pipeline (the faster, current direction)

| File | Purpose |
|------|---------|
| `assets/dog.xml` / `dog_scene.xml` | Custom 12-DOF dog quadruped (converted from `dog.urdf` + Fusion360 meshes), 2 kg, position actuators (Kp=20). Has MJX sensors + foot/floor contact sensors. |
| `assets/dog.urdf`, `assets/meshes/` | Original URDF source + STL meshes |
| `dog_locomotion.ipynb` | **Colab notebook** — trains the dog with JAX/MJX + Brax PPO (port of Playground's Go1 joystick task). ~minutes on a GPU vs hours for SB3. |
| `dog_joystick.py` | Real-time **keyboard** control of the trained dog policy (loads `dog_policy`) |
| `dog_policy` | Exported Brax policy params (gitignored — local only) |

## Running Things

```bash
# Train (always use --device cpu for MLP policy)
python train.py --n_envs 8 --device cpu

# Resume from checkpoint
python train.py --n_envs 8 --device cpu --resume models/best/best_model.zip

# Watch a checkpoint mid-training (shows cmd vs actual velocity)
python watch_training.py
python watch_training.py --model models/ppo_quadruped_5000000_steps.zip
python watch_training.py --stochastic   # matches training behaviour

# Simple viewer with random actions
python view.py

# Evaluate a checkpoint
python eval.py --model models/best/best_model.zip

# Keyboard control
python joystick_control.py --keyboard

# TensorBoard
tensorboard --logdir logs/tensorboard
```

## Robot Model

Uses the **official Unitree A1 from MuJoCo Menagerie** at `../mujoco_menagerie/unitree_a1/`.
- The env loads `scene.xml` (includes floor + lighting) automatically
- `a1.xml` uses built-in **position actuators** (Kp=100, joint damping for Kd)
- The controller writes desired joint positions to `data.ctrl` — NOT torques
- Actual torques are read from `data.actuator_force` (used for torque penalty)
- Foot slip detection uses calf body names (`FR_calf` etc.) since foot geoms are unnamed

Joint name order (must match `config.JOINT_NAMES`):
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf

Default standing pose (from A1 keyframe): `[0.0, 0.9, -1.8] × 4 legs`

## Observation Space (45 dims)

```
[0:3]   base angular velocity (body frame)   × 0.25
[3:6]   projected gravity vector             × 1.0
[6:9]   commanded velocity (vx, vy, yaw)     × 1.0
[9:21]  joint positions (Δ from default)     × 1.0
[21:33] joint velocities                     × 0.05
[33:45] previous action                      × 1.0
```

Note: linear velocity in `_compute_reward` is converted to **body frame** before
comparing against commands.

## Reward Function

Weights are in `config.py → REWARD_WEIGHTS`. Functions are in `env/rewards.py`.
Current weights reflect a curriculum training approach — see Training Strategy below.

To add a new reward term:
1. Write a function in `env/rewards.py`
2. Call it inside `QuadrupedEnv._compute_reward()` in `env/quadruped_env.py`
3. Add its weight to `REWARD_WEIGHTS` in `config.py`

Key insight: `lin_vel_tracking` uses a Gaussian with `sigma=0.5` (wider than default).
Tighter sigma → near-zero gradient when robot is stationary → policy learns nothing.

## Training Strategy (Curriculum)

Training proceeds in phases, resuming from the best checkpoint each time.
**Never jump command ranges too quickly — the policy will regress to standing still.**

| Phase | CMD_VX | CMD_VY | CMD_YAW | Notes |
|-------|--------|--------|---------|-------|
| 1 — forward only | (0.5, 1.0) | (0, 0) | (0, 0) | Establish basic gait |
| 2 — add slow lateral | (0.3, 1.0) | (-0.05, 0.05) | (0, 0) | Tiny vy nudge |
| 3 — expand lateral | (0.3, 1.0) | (-0.2, 0.2) | (0, 0) | Once phase 2 stable |
| 4 — add yaw | (0.3, 1.0) | (-0.2, 0.2) | (-0.5, 0.5) | Once lateral stable |
| 5 — full range | (-1.0, 1.0) | (-0.5, 0.5) | (-1.0, 1.0) | Final generalisation |

Watch `reward_terms/lin_vel_tracking` in TensorBoard. If it drops significantly after
a resume and doesn't recover within ~2M steps, the range jump was too large — stop
and use a more gradual transition.

## Architecture Note — Why Training is Slow

This project uses SB3 + CPU-based MuJoCo. The fundamental bottleneck is **8 parallel
environments** generating ~16K samples per PPO update. The JAX/MJX alternative
(MuJoCo Playground) runs **4096+ environments on GPU**, generating ~8M samples per
update — ~500× more experience per second. This is why Playground trains a full
vx/vy/yaw policy in 7 minutes while this project takes hours per curriculum phase.

If extending this project, consider porting to JAX/MJX. See `TRAINING_GUIDE.md` for
a detailed comparison.

## Custom Dog — JAX/MJX Pipeline

The custom dog quadruped (`assets/dog.xml`) is trained via `dog_locomotion.ipynb`
on Colab using JAX/MJX + Brax PPO — a self-contained port of Playground's
`Go1JoystickFlatTerrain`. Train on Colab, export the policy, then drive it locally
with `dog_joystick.py`.

**Hard-won gotchas (all cost real debugging time):**

- **MJX solver settings are everything for speed.** `dog.xml` must set
  `iterations="1" ls_iterations="5"` + `integrator="Euler"`. MJX runs a *fixed*
  iteration count every step, so MuJoCo's CPU defaults (iterations=100,
  ls_iterations=50) are ~100× too expensive. This was the single biggest speed bug.
- **Match Go1's buffer caps:** `impl="jax"`, `naconmax`, `njmax` capped small (the
  dog only has 4 foot-sphere contacts). Uncapped defaults are slower to compile/step.
- **Brax saves the observation normalizer separately from the network weights.**
  When reconstructing inference locally, rebuild the network with
  `preprocess_observations_fn=running_statistics.normalize` — otherwise the policy
  gets raw (unnormalized) observations, stays upright but tracks commands poorly /
  backwards. This is the #1 deployment trap for Brax policies.
- **Observation order must match `_get_obs` exactly:**
  `[local_linvel, gyro, gravity, qpos[7:]−default, qvel[6:], last_act, command]` = 48.
- Local inference needs `pip install "jax[cpu]" brax` (CPU is plenty for a tiny MLP
  at 50 Hz). Installing brax upgrades `mujoco` (3.8 → 3.9), which is compatible.

## Running the Dog

```bash
# View the dog model (static)
python view.py --dog

# Drive the trained policy with the keyboard (after exporting dog_policy from Colab)
python dog_joystick.py
#   Arrows = move, ,/. = turn, Space = stop, R = reset, Esc = quit
#   Holds standing pose at zero command by default (--no-hold-still to disable)
```

## Important Notes

- `models/` and `logs/` are gitignored — checkpoints stay local
- Always keep `_vecnorm.pkl` alongside a `.zip` checkpoint — VecNormalize stats are
  required for correct observations at inference time
- Training uses CPU (`--device cpu`) — the MLP policy on a 45-dim input is faster on
  CPU than GPU due to data transfer overhead
- `SubprocVecEnv` requires the `if __name__ == "__main__":` guard in `train.py`
- The position controller runs at 50 Hz (CONTROL_DT=0.02s); MuJoCo steps at 500 Hz (SIM_DT=0.002s)
- The A1's stiff actuators (Kp=100) produce large raw torques — keep `torque_penalty`
  weight small (≤ -0.00002) or it will overwhelm the locomotion reward
