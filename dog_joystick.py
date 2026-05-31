"""
dog_joystick.py — Drive the trained dog policy in real time with the keyboard.

Loads the Brax/MJX-trained policy (dog_policy) and runs it in a plain MuJoCo
passive viewer. The observation is assembled exactly as in the training env
(DogJoystickEnv) so the policy behaves the same as in simulation training.

Controls:
  Up / Down     forward / backward   (vx)
  Left / Right  left / right strafe  (vy)
  , / .         turn left / right    (yaw rate)
  Space         stop (zero command)
  R             reset to standing pose
  Esc           quit

Usage:
  python dog_joystick.py
  python dog_joystick.py --policy dog_policy --scene assets/dog_scene.xml
"""

import os
import sys
import time
import argparse
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

# ---- must match the training env (DogJoystickEnv) ----
ACTION_SCALE = 0.5
CTRL_DT = 0.02
SIM_DT = 0.004
N_SUBSTEPS = int(round(CTRL_DT / SIM_DT))
DEFAULT_POSE = np.array([0.0, 0.785, -1.6] * 4, dtype=np.float64)

# Command ranges the policy was trained on (command_config.a)
CMD_VX_MAX = 0.6
CMD_VY_MAX = 0.4
CMD_YAW_MAX = 1.0


# ---------------------------------------------------------------------------
# Thread-safe command + key state
# ---------------------------------------------------------------------------
class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.cmd = np.zeros(3, dtype=np.float64)  # vx, vy, yaw
        self.quit = False
        self.reset = False

    def set_cmd(self, vx, vy, yaw):
        with self.lock:
            self.cmd[:] = (vx, vy, yaw)

    def get_cmd(self):
        with self.lock:
            return self.cmd.copy()


def keyboard_thread(shared):
    from pynput import keyboard as kb
    pressed = set()

    def update():
        vx = CMD_VX_MAX if kb.Key.up in pressed else (
            -CMD_VX_MAX if kb.Key.down in pressed else 0.0)
        vy = CMD_VY_MAX if kb.Key.left in pressed else (
            -CMD_VY_MAX if kb.Key.right in pressed else 0.0)
        yaw = CMD_YAW_MAX if kb.KeyCode(char=',') in pressed else (
            -CMD_YAW_MAX if kb.KeyCode(char='.') in pressed else 0.0)
        shared.set_cmd(vx, vy, yaw)

    def on_press(key):
        if key == kb.Key.esc:
            shared.quit = True
            return False
        if key == kb.Key.space:
            pressed.clear()
            shared.set_cmd(0, 0, 0)
            return
        if key == kb.KeyCode(char='r'):
            shared.reset = True
            return
        pressed.add(key)
        update()

    def on_release(key):
        pressed.discard(key)
        update()

    print("\nControls:  ↑/↓ forward/back | ←/→ strafe | ,/. turn | Space stop | R reset | Esc quit\n")
    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Observation assembly (must match DogJoystickEnv._get_obs, no noise)
# ---------------------------------------------------------------------------
class ObsBuilder:
    def __init__(self, model):
        self.model = model
        self.imu_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self._adr = {}
        for name in ("local_linvel", "gyro"):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self._adr[name] = (model.sensor_adr[sid], model.sensor_dim[sid])

    def _sensor(self, data, name):
        a, d = self._adr[name]
        return data.sensordata[a:a + d]

    def gravity(self, data):
        R = data.site_xmat[self.imu_site].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    def build(self, data, last_act, cmd):
        return np.concatenate([
            self._sensor(data, "local_linvel"),       # 3
            self._sensor(data, "gyro"),               # 3
            self.gravity(data),                       # 3
            data.qpos[7:] - DEFAULT_POSE,             # 12
            data.qvel[6:],                            # 12
            last_act,                                 # 12
            cmd,                                      # 3
        ]).astype(np.float32)                         # = 48


def reset_to_home(model, data):
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)


def main(args):
    # ---- load model ----
    model_mj = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model_mj)
    reset_to_home(model_mj, data)

    # ---- load policy ----
    print(f"Loading policy: {args.policy}")
    params = model.load_params(args.policy)
    network = ppo_networks.make_ppo_networks(
        observation_size={"state": (48,), "privileged_state": (119,)},
        action_size=12,
        # Training normalized observations; the saved params include the running
        # mean/std. Must rebuild with the same preprocessor or the policy gets
        # raw (out-of-distribution) inputs and barely tracks commands.
        preprocess_observations_fn=running_statistics.normalize,
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    inference = ppo_networks.make_inference_fn(network)(params, deterministic=True)
    inference = jax.jit(inference)
    rng = jax.random.PRNGKey(0)
    dummy_priv = jp.zeros(119)

    obs_builder = ObsBuilder(model_mj)
    last_act = np.zeros(12, dtype=np.float64)

    # warm up JIT
    print("Compiling policy (first inference)...")
    s0 = obs_builder.build(data, last_act, np.zeros(3))
    inference({"state": jp.asarray(s0), "privileged_state": dummy_priv}, rng)
    print("Ready.")

    # ---- input thread ----
    shared = Shared()
    threading.Thread(target=keyboard_thread, args=(shared,), daemon=True).start()

    # ---- control loop ----
    with mujoco.viewer.launch_passive(model_mj, data) as viewer:
        while viewer.is_running() and not shared.quit:
            t0 = time.time()

            if shared.reset:
                reset_to_home(model_mj, data)
                last_act = np.zeros(12)
                shared.reset = False

            cmd = shared.get_cmd()
            if args.hold_still and np.linalg.norm(cmd) < 1e-6:
                # No command: hold the standing pose directly instead of running
                # the policy, which otherwise jitters the feet and drifts slightly.
                data.ctrl[:] = DEFAULT_POSE
                last_act = np.zeros(12)
            else:
                obs = obs_builder.build(data, last_act, cmd)
                action, _ = inference(
                    {"state": jp.asarray(obs), "privileged_state": dummy_priv}, rng)
                action = np.asarray(action, dtype=np.float64)
                data.ctrl[:] = DEFAULT_POSE + action * ACTION_SCALE
                last_act = action

            for _ in range(N_SUBSTEPS):
                mujoco.mj_step(model_mj, data)

            # auto-reset if it falls over (gravity z-projection flips)
            if obs_builder.gravity(data)[2] > 0.0:
                reset_to_home(model_mj, data)
                last_act = np.zeros(12)

            viewer.sync()
            dt = time.time() - t0
            if dt < CTRL_DT:
                time.sleep(CTRL_DT - dt)

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="dog_policy")
    p.add_argument("--scene", default=os.path.join("assets", "dog_scene.xml"))
    p.add_argument("--no-hold-still", dest="hold_still", action="store_false",
                   help="Keep running the policy at zero command (don't hold pose)")
    main(p.parse_args())
