"""Quick viewer — runs the env with random actions to verify joints are moving."""

import time
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
