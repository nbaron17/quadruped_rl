"""
controller.py — PD joint position controller.

The policy outputs *position offsets* (in radians) relative to a default
standing pose.  This module converts those offsets into torque commands
that get applied to MuJoCo actuators.

Formula:  τ = Kp * (q_des - q) - Kd * dq
"""

import numpy as np
import mujoco
from config import KP, KD, DEFAULT_JOINT_POS, TORQUE_LIMIT, JOINT_NAMES


class PDController:
    """
    Stateless PD controller — one call per control step.

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Cache joint IDs for fast lookup each step
        self._joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self._actuator_ids = list(range(model.nu))  # assume 1 actuator per joint

        self._default_pos = np.array(DEFAULT_JOINT_POS, dtype=np.float64)

    # ------------------------------------------------------------------
    def apply(self, action: np.ndarray):
        """
        Compute and apply torques for a given action vector.

        Parameters
        ----------
        action : np.ndarray  shape (12,)
            Policy output in [-1, 1], will be scaled to position offsets.
        """
        from config import ACTION_SCALE

        # Desired joint positions = default pose + scaled offset
        q_des = self._default_pos + action * ACTION_SCALE

        # Read current joint state from MuJoCo data
        q     = self._get_joint_positions()
        dq    = self._get_joint_velocities()

        # PD torques
        torques = KP * (q_des - q) - KD * dq

        # Safety clamp
        torques = np.clip(torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        # Write to MuJoCo ctrl array
        self.data.ctrl[:] = torques

    # ------------------------------------------------------------------
    def _get_joint_positions(self) -> np.ndarray:
        """Return joint positions in JOINT_NAMES order."""
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids],
            dtype=np.float64,
        )

    def _get_joint_velocities(self) -> np.ndarray:
        """Return joint velocities in JOINT_NAMES order."""
        return np.array(
            [self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids],
            dtype=np.float64,
        )
