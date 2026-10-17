
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

from skill_code import pick, place, move, align_to_quaternion, align_two_axes, open_gripper, close_gripper, push


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    target = Shape('pick_and_lift_target')
    success = ProximitySensor('pick_and_lift_success')
    
    target_pos = target.get_position()
    success_pos = success.get_position()

    target_pick_pos = target_pos + [0.0, 0.0, 0.0]
    obs, reward, done = pick(env, task, target_pos=target_pick_pos, timeout=10.0)

    target_move_pos = success_pos + [0.0, 0.0, 0.0]
    obs, reward, done = move(env, task, target_pos=target_move_pos, timeout=10.0)

    if done:
        print("Success! Target has been moved to the base.")
    else:
        print("Failed! Target was not moved to the base.")

    return obs, reward, done