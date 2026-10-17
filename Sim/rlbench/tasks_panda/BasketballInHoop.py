
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

from skill_code import pick, place, move, align_to_quaternion, open_gripper, close_gripper, push



def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):

    ball = Shape('ball')
    hoop = Shape('basket_ball_hoop_respondable')
    success_sensor = ProximitySensor('success')

    ball_pos = ball.get_position()
    hoop_pos = hoop.get_position()


    before_place_pos = hoop_pos + [-0.2, 0.0, 0.2]


    target_pos = hoop_pos + [0.0, 0.0, 0.15]


    obs, reward, done = pick(env, task, target_pos=ball_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)


    obs, reward, done = move(env, task, target_pos=before_place_pos, timeout=5.0)


    obs, reward, done = place(env, task, target_pos=target_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)


    if done:
        print("Success! Ball has been placed in the hoop.")
    else:
        print("Failed! Ball was not placed in the hoop.")

    return obs, reward, done