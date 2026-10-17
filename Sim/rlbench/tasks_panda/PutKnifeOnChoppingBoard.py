
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    knife_object = Shape('knife')
    chopping_board_object = Shape('chopping_board')
    knife_position = knife_object.get_position()
    chopping_board_position = chopping_board_object.get_position()
    knife_sizes, _ = get_bbox_sizes(knife_object)
    l = float(np.max(knife_sizes))
    

    knife_handle_pos = knife_position.copy()
    knife_handle_pos[2] += l / 4.0
    obs, reward, done = pick(env, task, target_pos=knife_handle_pos, approach_distance=0.15, approach_axis='z', timeout=10.0)


    knife_pos = knife_handle_pos.copy()
    knife_pos[2] += l
    obs, reward, done = move(env, task, target_pos=knife_pos, timeout=10.0)


    hover_pos = chopping_board_position.copy()
    hover_pos[2] += l
    obs, reward, done = place(env, task, target_pos=hover_pos, approach_distance=0.15, approach_axis='z', timeout=10.0)

    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")
    