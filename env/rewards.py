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
    sigma: float = 0.25,
) -> float:
    """
    Gaussian reward for matching commanded (vx, vy).

    Using a Gaussian instead of absolute error keeps the reward smooth and
    bounded in [0, 1], which helps PPO training stability.

    Parameters
    ----------
    base_lin_vel : (3,) array  world-frame base linear velocity
    cmd          : (3,) array  [vx, vy, yaw_rate]
    sigma        : width of the Gaussian kernel (tune to tighten/loosen tracking)
    """
    error = np.sum((base_lin_vel[:2] - cmd[:2]) ** 2)
    return float(np.exp(-error / sigma**2))


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


def foot_slip_penalty(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_names: list[str],
) -> float:
    """
    Penalise feet that are in contact with the ground but sliding.

    For each foot geom in contact, accumulates the squared lateral velocity
    of the contact point.  Zero penalty when feet are either airborne or
    stationary.

    Parameters
    ----------
    foot_geom_names : list of geom names that correspond to foot contacts
    """
    penalty = 0.0

    # Build a set of geom IDs we care about
    foot_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in foot_geom_names
    )

    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 in foot_ids or contact.geom2 in foot_ids:
            # Get the body attached to the foot geom
            geom_id = contact.geom1 if contact.geom1 in foot_ids else contact.geom2
            body_id = model.geom_bodyid[geom_id]

            # Linear velocity of the body in world frame
            vel = data.cvel[body_id][:3]   # [vx, vy, vz]
            # Penalise lateral (xy) slip only
            penalty += float(vel[0] ** 2 + vel[1] ** 2)

    return penalty
