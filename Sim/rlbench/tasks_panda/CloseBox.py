# tasks/close_box.py - done
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, quat_mul, get_bbox_sizes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    box_object = Joint('box_joint')
    box_position = box_object.get_position()
    lid_position = np.array(box_position.copy(), dtype=float)

    # Move behind the lid
    lid_position[2] += 0.05
    hover_pos = lid_position.copy()
    hover_pos[1] -= 0.3
    hover_pos[2] += 0.1
    obs, reward, done = align_two_axes(env, task, local_axes=('z', 'y'), world_axes=('y', 'x'), axis_dirs=(1, -1), tol_rad=2e-3, timeout=20.0)

    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)
    hover_pos[2] -= 0.05
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)
   
    # Push the lid
    hover_pos[1] += 0.4
    obs, reward, done = push(env, task, target_pos=lid_position, approach_distance=0.01, approach_axis='y', timeout=10.0)

    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")
    