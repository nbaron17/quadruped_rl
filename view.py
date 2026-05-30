"""
Quick viewer — choose what to view:

    python view.py          # A1 env with random actions
    python view.py --dog    # Custom dog model (static, no policy)
"""

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dog", action="store_true", help="View the custom dog MJCF")
args = parser.parse_args()

if args.dog:
    import mujoco
    import mujoco.viewer
    import os

    scene = os.path.join(os.path.dirname(__file__), "assets", "dog_scene.xml")
    model = mujoco.MjModel.from_xml_path(scene)
    data  = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
else:
    from env.quadruped_env import QuadrupedEnv
    from config import CONTROL_DT

    env = QuadrupedEnv(render_mode="human")
    obs, _ = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(CONTROL_DT)
        if terminated or truncated:
            obs, _ = env.reset()
