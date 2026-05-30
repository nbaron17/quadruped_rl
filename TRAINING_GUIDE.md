# Quadruped RL — Training Guide & Lessons Learned

This document captures hard-won lessons from training the Unitree A1 in MuJoCo.
Read this before starting a new training run.

---

## Architecture Decision — SB3/CPU vs JAX/MJX

We built this project using **Stable-Baselines3 + standard MuJoCo on CPU**. A proven
alternative is **JAX + MJX (MuJoCo's JAX backend) + Brax PPO**, used by DeepMind's
MuJoCo Playground. The performance difference is significant:

| Metric | Our approach (SB3/CPU) | JAX/MJX (Playground) |
|--------|------------------------|----------------------|
| Parallel environments | 8 (CPU processes) | 4096+ (GPU tensors) |
| Samples per update | ~16K (8 envs × 2048 steps) | ~8M (4096 envs × 2048 steps) |
| Time for 10M steps | ~2–3 hours | ~1–2 minutes |
| Time for 100M steps | ~20–30 hours | ~7 minutes (RTX 4090) |
| Training all commands (vx/vy/yaw) | Requires curriculum | From step 1 |
| GPU used for physics | No | Yes (MJX) |

**Why JAX is faster:** JAX JIT-compiles the entire training loop — env step, reward,
observation, PPO update — into a single GPU kernel. Nothing leaves the GPU between
steps. Our approach serialises through Python and CPU memory on every step.

**Why we couldn't switch mid-project:** Requires a full rewrite — different physics
backend, different RL library, different environment API, JAX-native code throughout.

**Recommendation for a new project:** Start with JAX/MJX + Brax + MuJoCo Playground.
The Go1JoystickFlatTerrain environment in Playground is the closest reference
implementation. Requires a modern NVIDIA GPU (RTX 3080+ recommended, 8GB+ VRAM).

---

## Critical Bugs to Know About

### 1. Use `scene.xml`, not `a1.xml`
`a1.xml` has no floor. The robot will train in free-fall and learn nothing useful.
The environment now auto-loads `scene.xml` — but if you ever swap the robot model,
make sure it includes a ground plane.

### 2. The A1 uses position actuators — not torque
The menagerie A1 has `<position>` actuators with built-in Kp=100.
`data.ctrl` takes **desired joint positions**, not torques.
Actual torques are in `data.actuator_force` (used for torque penalty).
If you ever see the robot immediately collapse with huge oscillations, this is why.

### 3. Velocity reward must use body-frame velocity
`data.qvel[:3]` is world-frame. The commanded velocity (vx, vy) is in the robot's
body frame. If you compare world-frame velocity to body-frame commands, the robot
gets rewarded for moving in the +x world direction regardless of which way it faces —
it will spin in circles and drift sideways.
Always rotate velocity into body frame first (see `_world_to_body_vec` in env).

---

## Reward Shaping — What Works

### Torque penalty must be tiny
The A1's Kp=100 actuators produce raw torque sums of 1000–3000 per step.
At weight `-0.0002` this overwhelms the locomotion reward and the robot learns
to stand still. Use `-0.00002` or smaller.
**Rule:** if `reward_terms/torque_penalty` magnitude is larger than
`reward_terms/lin_vel_tracking`, reduce the torque penalty weight.

### Widen the Gaussian sigma for velocity tracking
Default sigma=0.25 means a stationary robot (v=0, cmd=0.75) gets reward ≈ 0.
No gradient = nothing to learn from. Use `sigma=0.5` in `linear_velocity_tracking`.
Only tighten sigma once the robot is already walking.

### Don't add all penalties at once
Start with just the core objective (lin_vel_tracking + upright + base_height).
Add smoothness and foot_slip only once the robot walks reliably.
Each new penalty is a new way for the robot to find a "standing still is safe" minimum.

### The standing-still trap
The robot will find that standing still scores reliably on upright + base_height.
Signs you're in this trap:
- `lin_vel_tracking` < 0.05 after 3M+ steps
- `watch_training.py` shows actual velocity ≈ 0.00 despite non-zero commands
- Episode return is positive but low and flat

Escape: reduce penalties, increase `lin_vel_tracking` weight, widen sigma.

---

## Curriculum Training — The Only Reliable Approach

**Never expand command ranges in one jump.** The robot forgets how to walk.

### Recommended phases

| Phase | CMD_VX | CMD_VY | CMD_YAW | Target metric |
|-------|--------|--------|---------|---------------|
| 1 | (0.5, 1.0) | (0.0, 0.0) | (0.0, 0.0) | `lin_vel_tracking` > 0.7 |
| 2 | (0.3, 1.0) | (-0.05, 0.05) | (0.0, 0.0) | No regression from phase 1 |
| 3 | (0.3, 1.0) | (-0.2, 0.2) | (0.0, 0.0) | `lin_vel_tracking` stays > 0.6 |
| 4 | (0.3, 1.0) | (-0.2, 0.2) | (-0.5, 0.5) | Stable |
| 5 | (-1.0, 1.0) | (-0.5, 0.5) | (-1.0, 1.0) | Full control |

Resume from `models/best/best_model.zip` at each phase transition.

### Detecting regression early
After resuming with new command ranges, watch `reward_terms/lin_vel_tracking`
in TensorBoard for the first 2M steps:
- **Dips then recovers** → normal adaptation, keep going
- **Keeps dropping** → range jump too large, stop and use a smaller step

If `lin_vel_tracking` was 0.95 and drops below 0.4 and keeps falling → abort immediately.

---

## Training Configuration

### Always train on CPU
```bash
python train.py --n_envs 8 --device cpu
```
The MLP policy (45-dim input, 64×64 network) is faster on CPU than GPU due to
data transfer overhead. The GPU warning from SB3 is correct.

### VecNormalize — never lose these files
Every checkpoint needs its `_vecnorm.pkl` file.
Without it, the policy receives un-normalised observations and behaves erratically.
The training script saves these automatically alongside each checkpoint.

### Eval frequency
Default eval every 100k steps gives ~50 data points over 5M steps — enough to
see trends. Don't rely on `rollout/ep_rew_mean` alone — it's normalised by
VecNormalize. Use `eval/mean_reward` and `reward_terms/lin_vel_tracking` instead.

---

## Diagnosing a Bad Training Run

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Robot stands still from step 0 | Torque penalty too large | Reduce `torque_penalty` weight 10× |
| Robot walks then stops after resume | Command range jump too large | Smaller range expansion |
| Robot walks in circles | Yaw rate reward too small | Increase `yaw_rate_tracking` weight |
| Joint vibration | No action smoothness penalty | Add `action_smoothness: -0.01` |
| Robot drifts sideways | World-frame velocity in reward | Ensure body-frame conversion in `_compute_reward` |
| `lin_vel_tracking` flat near 0 | Sigma too tight | Use `sigma=0.5` in rewards.py |
| Early termination (< 50 steps) | No floor / robot spawning wrong | Check `scene.xml` is loaded |
| Good training, bad eval | Missing `_vecnorm.pkl` | Find matching vecnorm file |

---

## Useful Commands

```bash
# Train fresh
python train.py --n_envs 8 --device cpu

# Resume from best checkpoint
python train.py --n_envs 8 --device cpu --resume models/best/best_model.zip

# Watch a checkpoint — see commanded vs actual velocity
python watch_training.py
python watch_training.py --stochastic   # matches training behaviour

# Verify environment works (random actions, joints should move)
python view.py

# Evaluate
python eval.py --model models/best/best_model.zip

# Keyboard control
python joystick_control.py --keyboard

# TensorBoard
tensorboard --logdir logs/tensorboard
```

---

## Key TensorBoard Metrics to Watch

| Metric | What it tells you |
|--------|------------------|
| `reward_terms/lin_vel_tracking` | Is the robot actually moving toward commanded velocity? |
| `reward_terms/yaw_rate_tracking` | Is it staying straight / turning correctly? |
| `reward_terms/torque_penalty` | Is torque penalty overwhelming the signal? |
| `eval/mean_reward` | Clean policy performance (not normalised) |
| `rollout/ep_rew_mean` | Overall trend — normalised, less interpretable |

A good run looks like: `lin_vel_tracking` climbs steadily from ~0.05 to 0.7+ over
5–10M steps. Flat or declining after 3M steps = something is wrong.
