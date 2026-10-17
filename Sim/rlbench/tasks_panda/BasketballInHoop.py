
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

    # Define the target position for moving before placing
    before_place_pos = hoop_pos + [-0.2, 0.0, 0.2]

    # Define the target position for placing the ball into the hoop
    target_pos = hoop_pos + [0.0, 0.0, 0.15]

    # Step 1: Pick up the ball
    obs, reward, done = pick(env, task, target_pos=ball_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)

    # Step 2: Move before place the ball
    obs, reward, done = move(env, task, target_pos=before_place_pos, timeout=5.0)

    # Step 3: Place the ball into the hoop
    obs, reward, done = place(env, task, target_pos=target_pos, approach_axis='z', approach_distance=0.1, timeout=5.0)

    # Step 5: Check if the ball has successfully entered the hoop using the proximity sensor
    if done:
        print("Success! Ball has been placed in the hoop.")
    else:
        print("Failed! Ball was not placed in the hoop.")

    return obs, reward, done