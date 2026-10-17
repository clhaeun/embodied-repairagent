import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.tasks import PutRubbishInBin
import importlib
from rlbench.observation_config import ObservationConfig, CameraConfig
from pyrep.const import RenderMode


cam_config = CameraConfig(rgb=True, depth=True, mask=True, render_mode=RenderMode.OPENGL)
cam_config = CameraConfig(rgb=True, depth=True, mask=True, render_mode=RenderMode.OPENGL)
cam_config.image_size = (640, 640)

obs_config = ObservationConfig()
obs_config.set_all(True)
obs_config.front_camera = cam_config
obs_config.wrist_camera = cam_config
obs_config.overhead_camera = cam_config
obs_config.left_shoulder_camera = cam_config
obs_config.right_shoulder_camera = cam_config






"""
HEADLESS_MODE = True

def setup_environment(task_cls, robot_setup='panda'):
    np.random.seed(4)
    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )
    
    # UR5 robot
    obs_config.gripper_touch_forces = False
    env = Environment(action_mode=action_mode, obs_config=obs_config, robot_setup=robot_setup, headless=HEADLESS_MODE)
    print("Launching RLBench environment...")
    env.launch()
    print("Environment launched successfully.")
    task = env.get_task(task_cls)
    task.set_variation(0)
    return env, task
    """

HEADLESS_MODE = True

def setup_environment(task_cls, robot_setup='panda', seed: int | None = None, variation: int = 0):
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
    """

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaPlanning(),
        gripper_action_mode=Discrete(),
    )

    obs_config.gripper_touch_forces = False
    env = Environment(action_mode=action_mode, obs_config=obs_config, robot_setup=robot_setup, headless=HEADLESS_MODE)
    print("Launching RLBench environment...")
    env.launch()
    print("Environment launched successfully.")
    task = env.get_task(task_cls)

    task.set_variation(0)
    return env, task


def shutdown_environment(env):
    env.shutdown()