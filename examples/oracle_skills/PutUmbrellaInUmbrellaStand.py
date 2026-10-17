# tasks/put_umbrella_in_umbrella_stand.py - done
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, quat_mul, get_bbox_sizes, normalize_quaternion, normalize_vector, angle_diff, align_to_quaternion, open_gripper, close_gripper, push

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    umbrella_object = Shape('umbrella')
    stand_object = Shape('stand')
    umbrella_position = np.array(umbrella_object.get_position(), dtype=np.float64)
    stand_position = np.array(stand_object.get_position(), dtype=np.float64)
    umbrella_sizes, _ = get_bbox_sizes(umbrella_object)
    l, d = float(np.max(umbrella_sizes)), float(np.median(umbrella_sizes))
    
    # Move to umbrella handle hover pos
    handle_pos = umbrella_position.copy()
    handle_pos[1] -= 0.4*l
    handle_pos[2] += 0.05
    obs, reward, done = move(env, task, target_pos=handle_pos, timeout=10.0)

    # Perpendicular to umbrella handle
    obs, reward, done = align_two_axes(env, task, local_axes=('y', 'z'), world_axes=('x','z'), axis_dirs=(1,-1), tol_rad=2e-3, timeout=10.0)
    
    # Pick handle, approach_axis = z
    handle_pos[2] -= 0.05
    obs, reward, done = pick(env, task, target_pos=handle_pos, approach_distance=0.06, approach_axis='z', timeout=10.0)

    # Move to safe_z
    hover_pos = stand_position.copy()
    hover_pos[2] += 2*l
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)

    # Rotate umbrella
    obs, reward, done = align_two_axes(env, task, local_axes=('y','z'), world_axes=('x','y'), axis_dirs=(1,-1), tol_rad=2e-3, timeout=10.0)

    # Place umbrella in umbrella stand
    hover_pos[2] -= l
    obs, reward, done = place(env, task, target_pos=hover_pos, approach_distance=l, approach_axis='z', timeout=10.0)

    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")