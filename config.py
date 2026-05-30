"""
config.py — Central configuration for the quadruped RL project.

All tunable parameters live here so you only need to change one file
when iterating on reward weights, control gains, training hyperparams, etc.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Path to the Unitree A1 robot XML — the env will prefer scene.xml (floor + lighting)
# from the same directory if it exists.
ROBOT_XML_PATH = os.path.join(PROJECT_ROOT, "..", "mujoco_menagerie", "unitree_a1", "a1.xml")

# Where trained models are saved
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")

# TensorBoard log directory
TENSORBOARD_LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "tensorboard")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

SIM_DT = 0.002          # MuJoCo physics timestep (seconds)
CONTROL_DT = 0.02       # Policy control timestep (seconds) — 50 Hz
# The environment will call mujoco.mj_step() this many times per policy step:
SIM_STEPS_PER_CONTROL = int(CONTROL_DT / SIM_DT)  # = 10


# ---------------------------------------------------------------------------
# Robot / joint defaults
# ---------------------------------------------------------------------------

# Names of the 12 actuated joints in the MJCF.
# Order must match the action vector and observation vector.
JOINT_NAMES = [
    "FR_hip_joint",   "FR_thigh_joint",   "FR_calf_joint",
    "FL_hip_joint",   "FL_thigh_joint",   "FL_calf_joint",
    "RR_hip_joint",   "RR_thigh_joint",   "RR_calf_joint",
    "RL_hip_joint",   "RL_thigh_joint",   "RL_calf_joint",
]

# Default standing pose (joint positions in radians, matching JOINT_NAMES order).
# Taken from the "home" keyframe in the MuJoCo Menagerie A1 model.
DEFAULT_JOINT_POS = [
    0.0,  0.9, -1.8,   # FR
    0.0,  0.9, -1.8,   # FL
    0.0,  0.9, -1.8,   # RR
    0.0,  0.9, -1.8,   # RL
]

# The menagerie A1 uses built-in position actuators (kp=100, joint damping=1–2).
# KP / KD / TORQUE_LIMIT are kept here for reference but are not used by the
# controller — they are embedded in the MJCF itself.
KP = 100.0
KD = 2.0
TORQUE_LIMIT = 33.5  # [Nm]  (enforced by actuator forcerange in the MJCF)


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

# Observation vector layout (total = 3+3+3+12+12+12 = 45 dims):
#   [0:3]   base angular velocity  (roll_rate, pitch_rate, yaw_rate)
#   [3:6]   projected gravity vector  (3D)
#   [6:9]   commanded velocity        (vx, vy, yaw_rate_cmd)
#   [9:21]  joint positions (relative to default pose)
#   [21:33] joint velocities
#   [33:45] previous action
OBS_DIM = 45

# Scale factors to normalise observations to roughly [-1, 1]
OBS_SCALES = {
    "ang_vel":    0.25,
    "gravity":    1.0,
    "commands":   1.0,   # already in reasonable range
    "joint_pos":  1.0,
    "joint_vel":  0.05,
    "action":     1.0,
}


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

ACTION_DIM = 12
# Actions are *offsets* from DEFAULT_JOINT_POS, clipped to this range.
ACTION_SCALE = 0.25          # radians — multiply network output by this
ACTION_CLIP = 1.0            # clip raw network output to [-1, 1]


# ---------------------------------------------------------------------------
# Commands (velocity targets sampled during training)
# ---------------------------------------------------------------------------

CMD_VX_RANGE   = (0.3, 1.0)
CMD_VY_RANGE   = (-0.05, 0.05)   # very small lateral nudge
CMD_YAW_RANGE  = (0.0, 0.0)

# How often (in seconds) commands are re-sampled during training
CMD_RESAMPLE_INTERVAL = 5.0    # seconds

# Curriculum phases — pass --phase N to train.py to select one
CURRICULUM_PHASES = {
    1: dict(vx=(0.5,  1.0), vy=(0.0,  0.0),  yaw=(0.0,  0.0)),   # forward only
    2: dict(vx=(0.3,  1.0), vy=(0.0,  0.0),  yaw=(-0.5, 0.5)),   # add yaw
    3: dict(vx=(0.3,  1.0), vy=(-0.1, 0.1),  yaw=(-0.5, 0.5)),   # add small lateral
    4: dict(vx=(0.3,  1.0), vy=(-0.3, 0.3),  yaw=(-1.0, 1.0)),   # expand both
    5: dict(vx=(-1.0, 1.0), vy=(-0.5, 0.5),  yaw=(-1.0, 1.0)),   # full range
}


# ---------------------------------------------------------------------------
# Reward weights  ← tweak these to shape locomotion behaviour
# ---------------------------------------------------------------------------

REWARD_WEIGHTS = {
    "lin_vel_tracking":   3.0,
    "yaw_rate_tracking":  1.0,
    "upright":            0.5,
    "base_height":        0.2,
    "torque_penalty":     -0.00002,
    "action_smoothness":  -0.01,
    "foot_slip":          -0.05,
    "pose_regularisation": -0.01,  # keep stance natural
}

# Target base height above ground [m] — matches the A1 "home" keyframe
TARGET_BASE_HEIGHT = 0.27


# ---------------------------------------------------------------------------
# Episode termination thresholds
# ---------------------------------------------------------------------------

TERMINATION_HEIGHT   = 0.15    # [m]  fall if body COM drops below this
TERMINATION_ROLL     = 1.0     # [rad] ~57° roll
TERMINATION_PITCH    = 1.0     # [rad] ~57° pitch
EPISODE_MAX_STEPS    = 1000    # steps at CONTROL_DT → 20 s per episode


# ---------------------------------------------------------------------------
# Training (PPO via Stable-Baselines3)
# ---------------------------------------------------------------------------

PPO_CONFIG = {
    "learning_rate":    3e-4,
    "n_steps":          2048,       # steps collected per env before update
    "batch_size":       64,
    "n_epochs":         10,
    "gamma":            0.99,
    "gae_lambda":       0.95,
    "clip_range":       0.2,
    "ent_coef":         0.01,       # entropy bonus — encourages exploration
    "vf_coef":          0.5,
    "max_grad_norm":    0.5,
    "verbose":          1,
}

TOTAL_TIMESTEPS = 20_000_000    # total env steps to train for
SAVE_EVERY_STEPS = 500_000      # save a checkpoint every N steps
EVAL_EPISODES = 10              # episodes per evaluation callback run
N_ENVS = 4                      # number of parallel training environments
