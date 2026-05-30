"""
rewards.py — Individual reward term functions.

Each function takes the MuJoCo model/data plus relevant state and returns
a scalar float.  The environment weights and sums them each step.

Keeping rewards in a separate module makes it trivial to add, remove, or
re-weight terms without touching the main env file.
"""

import numpy as np
import mujoco


# ---------------------------------------------------------------------------
# Velocity tracking
# ---------------------------------------------------------------------------

def linear_velocity_tracking(
    base_lin_vel: np.ndarray,
    cmd: np.ndarray,
    sigma_vx: float = 0.5,
    sigma_vy: float = 0.5,
) -> float:
    """
    Separate Gaussian rewards for vx and vy tracking, averaged together.

    Splitting vx and vy means lateral accuracy gets its own gradient signal
    rather than being drowned out by the larger forward error.
    """
    error_vx = (base_lin_vel[0] - cmd[0]) ** 2
    error_vy = (base_lin_vel[1] - cmd[1]) ** 2
    reward_vx = np.exp(-error_vx / sigma_vx**2)
    reward_vy = np.exp(-error_vy / sigma_vy**2)
    # Weight vx much higher — lateral is harder and less critical early on
    return float((2.0 * reward_vx + 0.25 * reward_vy) / 2.25)


def yaw_rate_tracking(
    base_ang_vel: np.ndarray,
    cmd: np.ndarray,
    sigma: float = 0.25,
) -> float:
    """
    Gaussian reward for matching commanded yaw rate.

    Parameters
    ----------
    base_ang_vel : (3,) array  world-frame base angular velocity
    cmd          : (3,) array  [vx, vy, yaw_rate]
    """
    error = (base_ang_vel[2] - cmd[2]) ** 2
    return float(np.exp(-error / sigma**2))


# ---------------------------------------------------------------------------
# Posture
# ---------------------------------------------------------------------------

def upright_reward(gravity_proj: np.ndarray) -> float:
    """
    Reward the robot for staying upright.

    `gravity_proj` is the gravity vector projected into the robot's body
    frame.  When the robot is perfectly upright the z-component is -1 and
    x/y are 0.  We reward the magnitude of the z-component.

    Returns a value in [0, 1].
    """
    # gravity_proj[2] ≈ -1 when upright, drifts toward 0 when tilted
    return float(np.clip(-gravity_proj[2], 0.0, 1.0))


def base_height_reward(
    base_height: float,
    target_height: float,
    sigma: float = 0.05,
) -> float:
    """
    Gaussian reward for maintaining target base height.

    Parameters
    ----------
    base_height   : current z position of the robot base [m]
    target_height : desired height [m]
    sigma         : tolerance window [m]
    """
    error = (base_height - target_height) ** 2
    return float(np.exp(-error / sigma**2))


# ---------------------------------------------------------------------------
# Efficiency / smoothness  (penalties — will be multiplied by negative weights)
# ---------------------------------------------------------------------------

def torque_penalty(torques: np.ndarray) -> float:
    """
    Penalise large torques to encourage efficient gaits.

    Returns sum of squared torques (positive scalar).
    """
    return float(np.sum(torques ** 2))


def action_smoothness_penalty(action: np.ndarray, prev_action: np.ndarray) -> float:
    """
    Penalise large changes between consecutive actions.

    Encourages smooth joint trajectories and reduces mechanical wear.
    """
    return float(np.sum((action - prev_action) ** 2))


def pose_regularisation(
    joint_pos: np.ndarray,
    default_pos: np.ndarray,
) -> float:
    """Penalise deviation from default standing pose — keeps stance natural."""
    return float(np.sum((joint_pos - default_pos) ** 2))


def foot_slip_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_body_names: list[str],
) -> float:
    """
    Penalise feet that are in contact with the ground but sliding.

    Detects contact by calf body name (works with unnamed foot geoms in the
    Menagerie A1 model).  Accumulates squared lateral velocity for each foot
    body in contact.

    Parameters
    ----------
    foot_body_names : list of body names for the four calf/foot bodies
    """
    penalty = 0.0

    foot_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in foot_body_names
    }

    for i in range(data.ncon):
        contact = data.contact[i]
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        if body1 in foot_body_ids or body2 in foot_body_ids:
            body_id = body1 if body1 in foot_body_ids else body2
            vel = data.cvel[body_id][:3]   # [vx, vy, vz] world frame
            penalty += float(vel[0] ** 2 + vel[1] ** 2)

    return penalty
