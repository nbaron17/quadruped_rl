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

# Path to the MJCF robot model.
# We ship a simple 12-DOF "go1-style" model under assets/.
# Swap this path to use a different robot (e.g. full Unitree A1 from mujoco_menagerie).
ROBOT_XML_PATH = os.path.join(PROJECT_ROOT, "assets", "quadruped.xml")

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
# These are the "neutral" angles the robot returns to when actions = 0.
DEFAULT_JOINT_POS = [
    0.0,  0.8, -1.5,   # FR
    0.0,  0.8, -1.5,   # FL
    0.0,  0.8, -1.5,   # RR
    0.0,  0.8, -1.5,   # RL
]

# PD control gains applied to joint position offsets (action → torque).
KP = 20.0   # Position gain  [Nm/rad]
KD = 0.5    # Velocity gain  [Nm·s/rad]

# Max torque clamp (safety)
TORQUE_LIMIT = 33.5  # [Nm]  (Unitree A1 spec)


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

CMD_VX_RANGE   = (-1.0, 1.0)   # forward / backward  [m/s]
CMD_VY_RANGE   = (-0.5, 0.5)   # lateral             [m/s]
CMD_YAW_RANGE  = (-1.0, 1.0)   # yaw rate            [rad/s]

# How often (in seconds) commands are re-sampled during training
CMD_RESAMPLE_INTERVAL = 5.0    # seconds


# ---------------------------------------------------------------------------
# Reward weights  ← tweak these to shape locomotion behaviour
# ---------------------------------------------------------------------------

REWARD_WEIGHTS = {
    # Tracking — encourage matching commanded velocity
    "lin_vel_tracking":   1.0,    # reward for vx/vy tracking
    "yaw_rate_tracking":  0.5,    # reward for yaw rate tracking

    # Posture
    "upright":            0.2,    # penalise tilt
    "base_height":        0.1,    # reward target height

    # Efficiency / smoothness (penalties, should be negative-weighted)
    "torque_penalty":    -0.0002,  # penalise large torques
    "action_smoothness": -0.01,   # penalise large Δaction
    "foot_slip":         -0.1,    # penalise feet sliding while in contact
}

# Target base height above ground [m]
TARGET_BASE_HEIGHT = 0.28


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

TOTAL_TIMESTEPS = 10_000_000    # total env steps to train for
SAVE_EVERY_STEPS = 500_000      # save a checkpoint every N steps
EVAL_EPISODES = 10              # episodes per evaluation callback run
N_ENVS = 4                      # number of parallel training environments
