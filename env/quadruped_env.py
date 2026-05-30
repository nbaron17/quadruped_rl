"""
quadruped_env.py — Custom Gymnasium environment for quadruped locomotion.

Overview
--------
This environment wraps a MuJoCo simulation of a 12-DOF quadruped robot.
A policy (e.g. PPO) observes a 45-dimensional vector and outputs 12
continuous actions (joint position offsets).  A PD controller converts
those offsets to torques and steps the physics simulation.

Observation vector (45 dims)
-----------------------------
 [0:3]   Base angular velocity (roll_rate, pitch_rate, yaw_rate) in body frame
 [3:6]   Projected gravity vector in body frame (tells the robot its tilt)
 [6:9]   Commanded velocity  (vx, vy, yaw_rate)
 [9:21]  Joint positions relative to default standing pose
 [21:33] Joint velocities
 [33:45] Previous action

Action vector (12 dims)
------------------------
 Each element is a position offset in [-1, 1].
 Multiplied by ACTION_SCALE to get radians, then added to DEFAULT_JOINT_POS.
 A PD controller converts desired positions → torques.
"""

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

import sys, os
# Allow imports from project root when running this file directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ROBOT_XML_PATH,
    SIM_STEPS_PER_CONTROL,
    OBS_DIM,
    OBS_SCALES,
    ACTION_DIM,
    ACTION_CLIP,
    EPISODE_MAX_STEPS,
    TERMINATION_HEIGHT,
    TERMINATION_ROLL,
    TERMINATION_PITCH,
    TARGET_BASE_HEIGHT,
    REWARD_WEIGHTS,
    JOINT_NAMES,
    DEFAULT_JOINT_POS,
)
from env.commands import CommandSampler
from env.controller import PDController
from env.rewards import (
    linear_velocity_tracking,
    yaw_rate_tracking,
    upright_reward,
    base_height_reward,
    torque_penalty,
    action_smoothness_penalty,
    foot_slip_penalty,
    pose_regularisation,
)


class QuadrupedEnv(gym.Env):
    """
    MuJoCo-based quadruped locomotion environment.

    Parameters
    ----------
    render_mode : str | None
        "human"  → open a MuJoCo viewer window (use for eval/inference)
        "rgb_array" → return rendered frames via render()
        None     → no rendering (fastest, use for training)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None,
                 cmd_vx_range=None, cmd_vy_range=None, cmd_yaw_range=None):
        super().__init__()

        self.render_mode = render_mode

        # ------------------------------------------------------------------ #
        # Load MuJoCo model
        # ------------------------------------------------------------------ #
        # Prefer scene.xml (includes floor) over the bare robot XML
        scene_path = os.path.join(os.path.dirname(ROBOT_XML_PATH), "scene.xml")
        xml_path = scene_path if os.path.exists(scene_path) else ROBOT_XML_PATH

        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"Robot MJCF not found at: {xml_path}\n"
                "Run `python assets/generate_model.py` or place your own MJCF there."
            )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        # ------------------------------------------------------------------ #
        # Spaces
        # ------------------------------------------------------------------ #
        obs_low  = np.full(OBS_DIM, -np.inf, dtype=np.float32)
        obs_high = np.full(OBS_DIM,  np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Actions are raw policy outputs clipped to [-ACTION_CLIP, ACTION_CLIP]
        self.action_space = spaces.Box(
            low  = -ACTION_CLIP,
            high =  ACTION_CLIP,
            shape = (ACTION_DIM,),
            dtype = np.float32,
        )

        # ------------------------------------------------------------------ #
        # Sub-modules
        # ------------------------------------------------------------------ #
        self.cmd_sampler = CommandSampler(cmd_vx_range, cmd_vy_range, cmd_yaw_range)
        self.controller  = PDController(self.model, self.data)

        # ------------------------------------------------------------------ #
        # Internal state
        # ------------------------------------------------------------------ #
        self._prev_action    = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step_count     = 0

        # Cache joint IDs once for speed
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]

        # Calf body names used by foot_slip_penalty (geoms are unnamed in the A1 model)
        self._foot_body_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]

        # ------------------------------------------------------------------ #
        # Viewer (created lazily on first render call)
        # ------------------------------------------------------------------ #
        self._viewer       = None
        self._renderer     = None  # for rgb_array mode

    # ======================================================================= #
    # Gymnasium API
    # ======================================================================= #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Reset MuJoCo state to keyframe or default
        mujoco.mj_resetData(self.model, self.data)
        self._set_initial_pose()

        # Forward kinematics so observations are valid immediately
        mujoco.mj_forward(self.model, self.data)

        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._step_count  = 0
        self.cmd_sampler.reset()

        obs  = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        # Clip action to valid range
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP).astype(np.float32)

        # Apply PD control and advance simulation (multiple physics steps per policy step)
        self.controller.apply(action)
        for _ in range(SIM_STEPS_PER_CONTROL):
            mujoco.mj_step(self.model, self.data)

        # Advance command sampler (may trigger a resample)
        self.cmd_sampler.step()
        self._step_count += 1

        # ------------------------------------------------------------------ #
        # Compute reward
        # ------------------------------------------------------------------ #
        reward, reward_info = self._compute_reward(action)

        # ------------------------------------------------------------------ #
        # Termination check
        # ------------------------------------------------------------------ #
        terminated = self._is_terminated()
        truncated  = self._step_count >= EPISODE_MAX_STEPS

        # ------------------------------------------------------------------ #
        # Observation
        # ------------------------------------------------------------------ #
        obs = self._get_obs()

        # Store action for smoothness penalty next step
        self._prev_action = action.copy()

        # Debug info dict (useful for TensorBoard custom logging)
        base_quat = self.data.qpos[3:7].copy()
        actual_vel = self._world_to_body_vec(base_quat, self.data.qvel[:3].copy())

        info = {
            "reward_breakdown": reward_info,
            "base_height":      float(self.data.qpos[2]),
            "cmd":              self.cmd_sampler.get().tolist(),
            "actual_vel":       actual_vel[:2].tolist(),       # [vx, vy] body frame
            "actual_yaw_rate":  float(self.data.qvel[5]),      # world-frame yaw rate
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model)
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            del self._renderer
            self._renderer = None

    # ======================================================================= #
    # Observation
    # ======================================================================= #

    def _get_obs(self) -> np.ndarray:
        """
        Build the 45-dimensional observation vector.

        All values are scaled to be roughly in [-1, 1] to help neural net
        training.  The scale factors are defined in config.OBS_SCALES.
        """
        # --- Base angular velocity (body frame) ---
        # qvel[3:6] = body angular velocity in world frame.
        # Rotate into body frame using the base orientation quaternion.
        base_quat    = self.data.qpos[3:7].copy()   # (w, x, y, z)
        base_ang_vel_world = self.data.qvel[3:6].copy()
        base_ang_vel = self._world_to_body_vec(base_quat, base_ang_vel_world)
        ang_vel_obs  = base_ang_vel * OBS_SCALES["ang_vel"]

        # --- Projected gravity vector (body frame) ---
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity_body  = self._world_to_body_vec(base_quat, gravity_world)
        gravity_obs   = gravity_body * OBS_SCALES["gravity"]

        # --- Commanded velocity ---
        cmd_obs = self.cmd_sampler.get() * OBS_SCALES["commands"]

        # --- Joint positions relative to default pose ---
        joint_pos_raw = self._get_joint_positions()
        joint_pos_obs = (
            (joint_pos_raw - np.array(DEFAULT_JOINT_POS)) * OBS_SCALES["joint_pos"]
        ).astype(np.float32)

        # --- Joint velocities ---
        joint_vel_raw = self._get_joint_velocities()
        joint_vel_obs = (joint_vel_raw * OBS_SCALES["joint_vel"]).astype(np.float32)

        # --- Previous action ---
        prev_action_obs = self._prev_action * OBS_SCALES["action"]

        obs = np.concatenate([
            ang_vel_obs.astype(np.float32),     # 3
            gravity_obs.astype(np.float32),     # 3
            cmd_obs.astype(np.float32),         # 3
            joint_pos_obs,                      # 12
            joint_vel_obs,                      # 12
            prev_action_obs.astype(np.float32), # 12
        ])  # total = 45

        return obs

    # ======================================================================= #
    # Reward
    # ======================================================================= #

    def _compute_reward(self, action: np.ndarray):
        """
        Sum all reward terms using weights from config.REWARD_WEIGHTS.

        Returns
        -------
        total_reward : float
        reward_info  : dict  (for logging individual terms)
        """
        cmd = self.cmd_sampler.get()

        # Velocities — convert linear velocity to body frame so the command
        # "move forward" means forward relative to the robot, not the world
        base_quat    = self.data.qpos[3:7].copy()
        base_lin_vel = self._world_to_body_vec(base_quat, self.data.qvel[:3].copy())
        base_ang_vel = self.data.qvel[3:6].copy()
        base_height  = float(self.data.qpos[2])

        # Projected gravity (needed for upright reward)
        base_quat   = self.data.qpos[3:7].copy()
        grav_body   = self._world_to_body_vec(base_quat, np.array([0.0, 0.0, -1.0]))

        # Actual torques produced by the position actuators
        torques = self.data.actuator_force.copy()

        terms = {
            "lin_vel_tracking":   linear_velocity_tracking(base_lin_vel, cmd),
            "yaw_rate_tracking":  yaw_rate_tracking(base_ang_vel, cmd),
            "upright":            upright_reward(grav_body),
            "base_height":        base_height_reward(base_height, TARGET_BASE_HEIGHT),
            "torque_penalty":     torque_penalty(torques),
            "action_smoothness":  action_smoothness_penalty(action, self._prev_action),
            "foot_slip":          foot_slip_penalty(
                                    self.model, self.data, self._foot_body_names
                                  ),
            "pose_regularisation": pose_regularisation(
                                    self._get_joint_positions(),
                                    np.array(DEFAULT_JOINT_POS, dtype=np.float32),
                                  ),
        }

        w = REWARD_WEIGHTS
        total = sum(w[k] * v for k, v in terms.items())

        return float(total), terms

    # ======================================================================= #
    # Termination
    # ======================================================================= #

    def _is_terminated(self) -> bool:
        """Return True if the episode should end due to a fall or bad state."""
        base_height = float(self.data.qpos[2])
        if base_height < TERMINATION_HEIGHT:
            return True

        # Roll and pitch from quaternion
        roll, pitch, _ = self._quat_to_euler(self.data.qpos[3:7])
        if abs(roll) > TERMINATION_ROLL or abs(pitch) > TERMINATION_PITCH:
            return True

        return False

    # ======================================================================= #
    # Helpers
    # ======================================================================= #

    def _set_initial_pose(self):
        """Place the robot in a standing pose with small random perturbation."""
        # Base position: z slightly above ground
        self.data.qpos[2] = TARGET_BASE_HEIGHT + 0.05

        # Base orientation: identity quaternion (upright)
        self.data.qpos[3] = 1.0   # w
        self.data.qpos[4] = 0.0   # x
        self.data.qpos[5] = 0.0   # y
        self.data.qpos[6] = 0.0   # z

        # Set joints to default standing pose
        default = np.array(DEFAULT_JOINT_POS, dtype=np.float64)
        for i, jid in enumerate(self._joint_ids):
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] = default[i]

        # Small random noise on joint positions (helps explore diverse states)
        for jid in self._joint_ids:
            addr = self.model.jnt_qposadr[jid]
            self.data.qpos[addr] += np.random.uniform(-0.05, 0.05)

    def _get_joint_positions(self) -> np.ndarray:
        return np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self._joint_ids],
            dtype=np.float32,
        )

    def _get_joint_velocities(self) -> np.ndarray:
        return np.array(
            [self.data.qvel[self.model.jnt_dofadr[jid]] for jid in self._joint_ids],
            dtype=np.float32,
        )

    @staticmethod
    def _world_to_body_vec(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """
        Rotate a world-frame 3D vector into the body frame defined by `quat`.

        quat is (w, x, y, z) as MuJoCo stores it.
        Equivalent to: R^T @ vec  where R is the rotation matrix for quat.
        """
        # MuJoCo quaternion: (w, x, y, z)
        w, x, y, z = quat
        # Build rotation matrix (row-major)
        R = np.array([
            [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
            [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
        ])
        return R.T @ vec   # world → body = R^T

    @staticmethod
    def _quat_to_euler(quat: np.ndarray):
        """
        Convert MuJoCo (w, x, y, z) quaternion to (roll, pitch, yaw) in radians.
        Uses the ZYX / aerospace convention.
        """
        w, x, y, z = quat
        roll  = np.arctan2(2*(w*x + y*z),  1 - 2*(x*x + y*y))
        pitch = np.arcsin( np.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw   = np.arctan2(2*(w*z + x*y),  1 - 2*(y*y + z*z))
        return roll, pitch, yaw
