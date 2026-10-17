
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

from skill_code import pick, place, move, align_to_quaternion, align_two_axes, open_gripper, close_gripper, push


def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    charger = Shape('charger')

    charger_pos = charger.get_position()

    pick_pos = charger_pos + [0.0, 0.0, 0.0]
    obs, reward, done = pick(env, task, target_pos=pick_pos, timeout=10.0)

    unplug_pos = charger_pos + [-0.2, 0.0, 0.0]
    obs, reward, done = place(env, task, target_pos=unplug_pos, approach_axis='x', approach_distance=0.1, timeout=10.0)

    if done:
        print("Success! Charger has been unplugged.")
    else:
        print("Failed! Charger was not unplugged.")

    return obs, reward, done