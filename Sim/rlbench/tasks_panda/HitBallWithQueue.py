
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, align_to_quaternion, open_gripper, close_gripper, push
from env_utils import quat_mul, get_bbox_sizes, normalize_vector

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    queue_object = Shape('queue')
    ball_object = Shape('ball')
    queue_position = queue_object.get_position()
    ball_position = ball_object.get_position()
    queue_sizes, _ = get_bbox_sizes(queue_object)
    l = float(np.max(queue_sizes)) / 2.0
    ball_sizes, _ = get_bbox_sizes(ball_object)
    r = float(np.max(ball_sizes)) / 2.0


    hover_pos = queue_position.copy()
    hover_pos[2] += 0.15
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)


    gripper_quat = normalize_quaternion(obs.gripper_pose[3:7])
    queue_quat = normalize_quaternion(queue_object.get_quaternion())
    queue_y = R.from_quat(queue_quat).apply([0.0, 1.0, 0.0])
    queue_y = queue_y / np.linalg.norm(queue_y)
    yaw = np.arctan2(queue_y[1], queue_y[0])
    yaw = 0.5*yaw
    q_yaw = np.array([0.0, 0.0, np.sin(yaw/2), np.cos(yaw/2)], dtype=np.float64)
    
    target_quat = normalize_quaternion(quat_mul(q_yaw, gripper_quat))
    obs, reward, done = align_to_quaternion(env, task, target_quat=target_quat, tol_rad=1e-3, timeout=10.0)
    

    queue_position[1] -= 0.08
    obs, reward, done = pick(env, task, target_pos=queue_position, approach_distance=0.01, approach_axis='z', timeout=10.0)
    

    queue_pos = queue_position.copy()
    queue_pos[1] -= l + r + 0.08
    queue_pos[2] = 0.77
    obs, reward, done = move(env, task, target_pos=queue_pos, timeout=10.0)
    
    queue_pos[0] += ball_position[0] - queue_position[0] + r
    obs, reward, done = move(env, task, target_pos=queue_pos, timeout=10.0)


    obs, reward, done = push(env, task, target_pos=ball_position, approach_distance=0.01, approach_axis='y', timeout=10.0)


    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")
    