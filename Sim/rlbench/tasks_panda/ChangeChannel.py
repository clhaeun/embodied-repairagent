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
    success0 = ProximitySensor('success0')

    remote_position = remote_object.get_position()
    remote_quat = remote_object.get_quaternion()
    button1_position = np.array(button1_object.get_position(), dtype=float)
    success0_quat = success0.get_quaternion()

    # Move to hover position
    hover_pos = remote_position.copy()
    hover_pos[2] += 0.15
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)

    # Perpendicular to remote
    obs, reward, done = align_to_quaternion(env, task, object_quaternion=remote_quat, yaw_align='perpendicular', approach_direction='down')

    # Pick the remote
    obs, reward, done = pick(env, task, target_pos=remote_position, approach_distance=0.01, approach_axis='z', timeout=10.0)
    
    # Align to TV
    obs, reward, done = align_to_quaternion(env, task, object_quaternion=success0_quat, yaw_align='perpendicular', approach_direction='down')

    # Place the remote
    remote_position[0] -= 0.043
    remote_position[1] += 0.06
    obs, reward, done = place(env, task, target_pos=remote_position, approach_distance=0.01, approach_axis='z', timeout=10.0)

    # Close gripper, Push the button
    push_pos = button1_position.copy() + [0.0, 0.0, -0.01]
    obs, reward, done = push(env, task, target_pos=push_pos, approach_distance=0.01, approach_axis='z', timeout=10.0)

    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")

    