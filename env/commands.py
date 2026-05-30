"""
commands.py — Manages velocity commands sent to the robot.

During training, commands are re-sampled randomly from configured ranges
every CMD_RESAMPLE_INTERVAL seconds so the policy learns to track arbitrary
goals. During inference the commands come from a keyboard or joystick.
"""

import numpy as np
from config import (
    CMD_VX_RANGE,
    CMD_VY_RANGE,
    CMD_YAW_RANGE,
    CMD_RESAMPLE_INTERVAL,
    CONTROL_DT,
)


class CommandSampler:
    """
    Randomly re-samples (vx, vy, yaw_rate) every few seconds.

    Call `step()` on every environment timestep to check whether it is
    time to resample.  Call `get()` to read the current command vector.
    """

    def __init__(self, vx_range=None, vy_range=None, yaw_range=None):
        self._vx_range  = vx_range  or CMD_VX_RANGE
        self._vy_range  = vy_range  or CMD_VY_RANGE
        self._yaw_range = yaw_range or CMD_YAW_RANGE
        # Steps between command resamples
        self._resample_steps = int(CMD_RESAMPLE_INTERVAL / CONTROL_DT)
        self._step_counter = 0
        self._cmd = np.zeros(3, dtype=np.float32)
        self.resample()

    # ------------------------------------------------------------------
    def resample(self):
        """Draw a new random command from the configured ranges."""
        self._cmd[0] = np.random.uniform(*self._vx_range)
        self._cmd[1] = np.random.uniform(*self._vy_range)
        self._cmd[2] = np.random.uniform(*self._yaw_range)

    def step(self):
        """Advance internal counter; resample when interval expires."""
        self._step_counter += 1
        if self._step_counter >= self._resample_steps:
            self._step_counter = 0
            self.resample()

    def reset(self):
        """Call at the start of each episode."""
        self._step_counter = 0
        self.resample()

    def get(self) -> np.ndarray:
        """Return the current command as a (3,) float32 array [vx, vy, yaw]."""
        return self._cmd.copy()

    def set(self, vx: float, vy: float, yaw: float):
        """
        Override the current command directly (used during inference when
        the user is driving with a joystick or keyboard).
        """
        self._cmd[0] = float(vx)
        self._cmd[1] = float(vy)
        self._cmd[2] = float(yaw)
