"""
joystick_control.py — Drive the trained robot with a keyboard or gamepad.

Controls (keyboard fallback)
-----------------------------
  W / S   →  forward / backward  (vx)
  A / D   →  strafe left / right (vy)
  Q / E   →  rotate left / right (yaw rate)
  Space   →  stop (zero all commands)
  Esc     →  quit

Controls (gamepad — Xbox / PlayStation layout via pygame)
---------------------------------------------------------
  Left stick Y    → vx  (forward/backward)
  Left stick X    → vy  (strafe)
  Right stick X   → yaw rate
  B / Circle      → quit

Usage
-----
    python joystick_control.py --model models/best/best_model.zip
    python joystick_control.py --model models/best/best_model.zip --keyboard
"""

import os
import sys
import argparse
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env.quadruped_env import QuadrupedEnv
from config import CONTROL_DT, MODEL_SAVE_DIR, CMD_VX_RANGE, CMD_VY_RANGE, CMD_YAW_RANGE


# ---------------------------------------------------------------------------
# Shared command state (written by input thread, read by control loop)
# ---------------------------------------------------------------------------

class SharedCommand:
    """Thread-safe velocity command holder."""

    def __init__(self):
        self._lock = threading.Lock()
        self._vx  = 0.0
        self._vy  = 0.0
        self._yaw = 0.0
        self._quit = False

    def set(self, vx, vy, yaw):
        with self._lock:
            self._vx  = float(vx)
            self._vy  = float(vy)
            self._yaw = float(yaw)

    def get(self):
        with self._lock:
            return self._vx, self._vy, self._yaw

    def signal_quit(self):
        with self._lock:
            self._quit = True

    @property
    def quit(self):
        with self._lock:
            return self._quit


# ---------------------------------------------------------------------------
# Keyboard input (pynput)
# ---------------------------------------------------------------------------

def run_keyboard_thread(shared: SharedCommand):
    """
    Reads WASD/QE keys and updates SharedCommand.
    Runs in a daemon thread so it dies when the main thread exits.
    """
    try:
        from pynput import keyboard as kb
    except ImportError:
        print("[keyboard] pynput not installed — pip install pynput")
        return

    # Track which keys are currently held down
    pressed = set()

    VX_STEP  = 0.75   # within trained range (0.5–1.0)
    VY_STEP  = 0.2
    YAW_STEP = 0.3

    def _update():
        vx  = VX_STEP  if kb.KeyCode(char='w') in pressed else (
             -VX_STEP  if kb.KeyCode(char='s') in pressed else 0.0)
        vy  = VY_STEP  if kb.KeyCode(char='a') in pressed else (
             -VY_STEP  if kb.KeyCode(char='d') in pressed else 0.0)
        yaw = YAW_STEP if kb.KeyCode(char='q') in pressed else (
             -YAW_STEP if kb.KeyCode(char='e') in pressed else 0.0)
        shared.set(vx, vy, yaw)

    def on_press(key):
        if key == kb.Key.esc:
            shared.signal_quit()
            return False   # stops the listener
        if key == kb.Key.space:
            shared.set(0, 0, 0)
        else:
            pressed.add(key)
            _update()

    def on_release(key):
        pressed.discard(key)
        _update()

    print("\nKeyboard controls:")
    print("  W/S = forward/back | A/D = strafe | Q/E = yaw | Space = stop | Esc = quit\n")

    with kb.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


# ---------------------------------------------------------------------------
# Gamepad input (pygame)
# ---------------------------------------------------------------------------

def run_gamepad_thread(shared: SharedCommand):
    """
    Reads Xbox/PS joystick axes via pygame and updates SharedCommand.
    Falls back to a warning if no joystick is found.
    """
    try:
        import pygame
    except ImportError:
        print("[gamepad] pygame not installed — pip install pygame")
        return

    pygame.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("[gamepad] No joystick detected. Falling back to keyboard.")
        run_keyboard_thread(shared)
        return

    joy = pygame.joystick.Joystick(0)
    joy.init()
    print(f"[gamepad] Connected: {joy.get_name()}")
    print("  Left stick = move | Right stick X = yaw | B/Circle = quit\n")

    # Axis mapping (works for most XInput / DualShock controllers)
    AXIS_VX  = 1   # Left stick Y (inverted: push forward = negative)
    AXIS_VY  = 0   # Left stick X
    AXIS_YAW = 3   # Right stick X

    DEADZONE = 0.10   # Ignore tiny stick drift

    def apply_deadzone(v):
        return 0.0 if abs(v) < DEADZONE else v

    while not shared.quit:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared.signal_quit()
            if event.type == pygame.JOYBUTTONDOWN:
                # Button 1 = B (Xbox) / Circle (PS) — quit
                if event.button == 1:
                    shared.signal_quit()

        vx  = -apply_deadzone(joy.get_axis(AXIS_VX))   # invert Y axis
        vy  =  apply_deadzone(joy.get_axis(AXIS_VY))
        yaw = -apply_deadzone(joy.get_axis(AXIS_YAW))  # right stick X

        # Scale axes (range -1..1) to command ranges
        vx  *= abs(CMD_VX_RANGE[1])
        vy  *= abs(CMD_VY_RANGE[1])
        yaw *= abs(CMD_YAW_RANGE[1])

        shared.set(vx, vy, yaw)
        time.sleep(0.02)   # 50 Hz gamepad polling

    pygame.quit()


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run(args):
    shared = SharedCommand()

    # Start input thread
    if args.keyboard:
        t = threading.Thread(target=run_keyboard_thread, args=(shared,), daemon=True)
    else:
        t = threading.Thread(target=run_gamepad_thread,  args=(shared,), daemon=True)
    t.start()

    # ------------------------------------------------------------------ #
    # Load model
    # ------------------------------------------------------------------ #
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model: {model_path}")

    def make_env():
        env = QuadrupedEnv(render_mode="human")
        # Override command sampler to use joystick input
        return env

    env = DummyVecEnv([make_env])

    norm_path = model_path.replace(".zip", "_vecnorm.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training    = False
        env.norm_reward = False

    model = PPO.load(model_path, env=env)

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #
    print("\nStarting joystick control loop. Close MuJoCo window or press Esc to stop.\n")
    obs = env.reset()

    while not shared.quit:
        # Inject user commands directly into the environment's command sampler
        vx, vy, yaw = shared.get()
        # Access the underlying QuadrupedEnv through VecNormalize → DummyVecEnv
        inner_env = env.venv.envs[0] if hasattr(env, 'venv') else env.envs[0]
        inner_env.cmd_sampler.set(vx, vy, yaw)

        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        if done[0]:
            obs = env.reset()

        time.sleep(CONTROL_DT)

    env.close()
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joystick/keyboard control of trained quadruped")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_SAVE_DIR, "best", "best_model"),
        help="Path to trained .zip model",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Force keyboard input even if a gamepad is connected",
    )
    args = parser.parse_args()
    run(args)
