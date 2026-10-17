
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

from skill_code import pick, place, move, align_to_quaternion, align_two_axes, open_gripper, close_gripper, push


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    phone = Shape('phone')
    success = ProximitySensor('success')
    
    phone_pos = phone.get_position()
    success_pos = success.get_position()

    pick_pos = phone_pos + [0.0, 0.0, 0.0]
    obs, reward, done = pick(env, task, target_pos=pick_pos, timeout=10.0)

    above_place_pos = success_pos + [0.0, 0.0, 0.2]
    obs, reward, done = move(env, task, target_pos=above_place_pos, timeout=10.0)

    place_pos = success_pos + [0.0, 0.0, 0.05]
    obs, reward, done = place(env, task, target_pos=place_pos, timeout=10.0)

    if done:
        print("Success! Phone has been placed on the base.")
    else:
        print("Failed! Phone was not placed on the base.")

    return obs, reward, done