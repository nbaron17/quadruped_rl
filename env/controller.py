"""
controller.py — Joint position controller.

The policy outputs *position offsets* (in radians) relative to a default
standing pose.  The Menagerie A1 model uses built-in position actuators
(kp=100, joint damping for kd), so we simply write desired positions to
data.ctrl and let MuJoCo's actuator model compute the torques.
"""

import numpy as np
import mujoco
from config import DEFAULT_JOINT_POS, ACTION_SCALE


class PDController:
    """
    Writes desired joint positions to data.ctrl each control step.

    Parameters
    ----------
    model : mujoco.MjModel
    data  : mujoco.MjData
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.data = data
        self._default_pos = np.array(DEFAULT_JOINT_POS, dtype=np.float64)

    def apply(self, action: np.ndarray):
        """
        Parameters
        ----------
        action : np.ndarray  shape (12,)
            Policy output in [-1, 1], scaled to position offsets.
        """
        q_des = self._default_pos + action * ACTION_SCALE
        self.data.ctrl[:] = q_des
