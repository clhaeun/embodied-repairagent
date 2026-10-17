# tasks/change_channel.py -  done
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, quat_mul, get_bbox_sizes, normalize_quaternion, normalize_vector, angle_diff, align_to_quaternion, open_gripper, close_gripper, push

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    remote_object = Shape('tv_remote')
    button1_object = Shape('target_button_wrap1')

    remote_position = remote_object.get_position()
    button1_position = np.array(button1_object.get_position(), dtype=float)

    # Move to hover position
    hover_pos = remote_position.copy()
    hover_pos[2] += 0.15
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)

    # Perpendicular to remote
    remote_quat = normalize_quaternion(np.array(remote_object.get_quaternion(), dtype=np.float64))
    remote_axis = R.from_quat(remote_quat).apply([0,1,0])  # remote 벡터(로컬 y축)이 월드에서는 어떤 벡터인지 
    remote_axis[2] = 0.0
    remote_axis = normalize_vector(remote_axis)

    remote_perp = normalize_vector(np.array([-remote_axis[1], remote_axis[0], 0.0], dtype=np.float64))  # 그리퍼가 align할 벡터
    obs, reward, done = align_two_axes(env, task, local_axes=('z','x'), world_axes=('z', remote_perp), axis_dirs=(-1,1), tol_rad=2e-3, timeout=30.0)

    # Pick the remote
    obs, reward, done = pick(env, task, target_pos=remote_position, approach_distance=0.01, approach_axis='z', timeout=10.0)
    
    # Align to TV
    obs, reward, done = align_two_axes(env, task, local_axes=('z','y'), world_axes=('z', 'y'), axis_dirs=(-1,1), tol_rad=2e-3, timeout=30.0)
    remote_position[0] -= 0.043
    remote_position[1] += 0.06
    obs, reward, done = place(env, task, target_pos=remote_position, approach_distance=0.01, approach_axis='z', timeout=10.0)

    # Close gripper, Push the button
    push_pos = button1_position.copy()
    offset = np.array([-0.016, 0.014, 0.04])
    push_pos += offset
    obs, reward, done = move(env, task, target_pos=push_pos, timeout=10.0)
    obs, reward, done = close_gripper(env, task, timeout=5.0)
    
    push_pos[2] -= 0.05
    obs, reward, done = push(env, task, target_pos=push_pos, approach_distance=0.01, approach_axis='z', timeout=10.0)


    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")

    