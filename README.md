# Quadruped RL — Locomotion with MuJoCo + PPO

Train a 12-DOF quadruped robot to walk in simulation, then drive it live with a keyboard or gamepad.

---

## Project Structure

```
quadruped_rl/
├── env/
│   ├── quadruped_env.py   # Gymnasium environment (main file)
│   ├── rewards.py         # Individual reward term functions
│   ├── controller.py      # PD joint position controller
│   └── commands.py        # Velocity command sampler
├── assets/
│   ├── generate_model.py  # Generates the MJCF robot model
│   └── quadruped.xml      # Generated robot (created by generate_model.py)
├── models/                # Saved checkpoints go here
├── logs/                  # TensorBoard logs
├── config.py              # All tunable parameters
├── train.py               # PPO training script
├── eval.py                # Load & visualise a trained policy
├── joystick_control.py    # Real-time keyboard / gamepad control
└── requirements.txt
```

---

## Setup

### 1. Create and activate the virtual environment

```bash
# From the project root (Quarduped Control/)
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows CMD
venv\Scripts\activate.bat

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
cd quadruped_rl
pip install -r requirements.txt
```

### 3. Generate the robot MJCF model

```bash
python assets/generate_model.py
```

This writes `assets/quadruped.xml` — a simplified Unitree A1-style robot.

> **Tip:** You can replace this with the official Unitree A1 from
> [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
> Download the A1 folder and set `ROBOT_XML_PATH` in `config.py`.

---

## Training

```bash
# Basic run (uses config.py defaults)
python train.py

# More environments = faster training (needs more RAM/CPU)
python train.py --n_envs 8

# Resume from a checkpoint
python train.py --resume models/ppo_quadruped_500000_steps.zip

# Custom timestep budget
python train.py --timesteps 20000000
```

Monitor training live:

```bash
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006 in a browser
```

Checkpoints are saved to `models/` every 500k steps.
The best policy (by eval return) is saved to `models/best/best_model.zip`.

---

## Evaluation

View the trained robot walking in the MuJoCo viewer:

```bash
python eval.py
# or point to a specific checkpoint:
python eval.py --model models/ppo_quadruped_1000000_steps.zip --episodes 5
```

---

## Joystick / Keyboard Control

After training, drive the robot yourself:

```bash
# Gamepad (auto-detected via pygame)
python joystick_control.py

# Force keyboard input
python joystick_control.py --keyboard
```

**Keyboard controls:**

| Key     | Command          |
|---------|------------------|
| W / S   | Forward / Back   |
| A / D   | Strafe L / R     |
| Q / E   | Rotate L / R     |
| Space   | Stop             |
| Esc     | Quit             |

**Gamepad (Xbox / PlayStation):**

| Input          | Command     |
|----------------|-------------|
| Left stick Y   | Forward/Back|
| Left stick X   | Strafe      |
| Right stick X  | Yaw         |
| B / Circle     | Quit        |

---

## Tuning the Reward Function

All reward weights are in `config.py → REWARD_WEIGHTS`.

```python
REWARD_WEIGHTS = {
    "lin_vel_tracking":   1.0,    # ← increase to prioritise speed tracking
    "yaw_rate_tracking":  0.5,
    "upright":            0.2,    # ← increase if robot keeps falling over
    "base_height":        0.1,
    "torque_penalty":    -0.0002, # ← more negative = more energy efficient gait
    "action_smoothness": -0.01,
    "foot_slip":         -0.1,
}
```

The individual reward functions live in `env/rewards.py` and are fully
independent — add, remove, or rewrite them without touching the env code.

---

## Observation Space (45 dims)

| Index   | Content                           | Scale  |
|---------|-----------------------------------|--------|
| 0–2     | Base angular velocity (body frame)| × 0.25 |
| 3–5     | Projected gravity vector          | × 1.0  |
| 6–8     | Commanded velocity (vx, vy, yaw)  | × 1.0  |
| 9–20    | Joint positions (Δ from default)  | × 1.0  |
| 21–32   | Joint velocities                  | × 0.05 |
| 33–44   | Previous action                   | × 1.0  |

---

## Tips for Getting Good Results

1. **Start simple** — disable `foot_slip` and `action_smoothness` penalties initially.
2. **Watch TensorBoard** — `ep_rew_mean` should trend upward. Flat = bad hyperparams.
3. **More envs = faster** — `--n_envs 16` if your machine has the cores.
4. **VecNormalize matters** — don't delete the `_vecnorm.pkl` file alongside checkpoints.
5. **Tune KP/KD in `config.py`** — if joints oscillate, lower KP or raise KD.
6. **Curriculum** — once the robot walks forward, tighten `CMD_VX_RANGE` and add lateral/yaw.

---

## Upgrading to the Official Unitree A1

1. Clone [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
2. Copy the `unitree_a1/` folder into `assets/`
3. Set in `config.py`:
   ```python
   ROBOT_XML_PATH = os.path.join(PROJECT_ROOT, "assets", "unitree_a1", "scene.xml")
   ```
4. Verify `JOINT_NAMES` in `config.py` match the joint names in the MJCF.

---

## Requirements

- Python 3.10+
- MuJoCo 3.x
- See `requirements.txt` for full list
