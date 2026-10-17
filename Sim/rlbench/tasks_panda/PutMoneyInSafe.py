# tasks/put_money_in_safe.py - done
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, align_to_quaternion, open_gripper, close_gripper, push
from env_utils import quat_mul, get_bbox_sizes, normalize_vector, _to_world_vec

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    money_object = Shape('dollar_stack')
    money_boundary = Shape('dollar_stack_boundary')
    safe_object = Dummy('dummy_shelf0')
    money_position = money_object.get_position()
    boundary_position = np.array(money_boundary.get_position(), dtype=np.float64)
    safe_position = np.array(safe_object.get_position(), dtype=np.float64)
    money_quat = normalize_quaternion(money_object.get_quaternion())
    money_sizes, _ = get_bbox_sizes(money_object)
    Lm, Tm = float(np.max(money_sizes)), float(np.min(money_sizes))
    safe_sizes, _ = get_bbox_sizes(safe_object)
    l, h = float(np.max(safe_sizes)), float(np.min(safe_sizes))

    # Align to money_axis, move to hover point
    hover_pos = money_position.copy()
    hover_pos[0] -= l
    hover_pos[1] += l/2
    hover_pos[2] += 0.1+0.05
    obs, reward, done = move(env, task, target_pos=hover_pos, timeout=10.0)
    obs, reward, done = align_two_axes(env, task, local_axes=('z', 'y'), world_axes=('x', 'y'), axis_dirs=(1, 1), tol_rad=1e-3, timeout=40.0)

    # Pick money on the side
    money_side = np.array([money_position[0]-0.05, money_position[1], money_position[2]+h/2], dtype=np.float64)
    obs, reward, done = pick(env, task, target_pos=money_side, approach_distance=0.1, approach_axis='z', timeout=10.0)

    # Move to hover2_pos
    hover2_pos = np.array([money_side[0]-2*Lm, money_side[1], money_side[2]+2*Lm], dtype=np.float64)
    obs, reward, done = move(env, task, target_pos=hover2_pos, timeout=10.0)
    obs, reward, done = align_two_axes(env, task, local_axes=('z', 'y'), world_axes=('z', 'x'), axis_dirs=(-1, 1), tol_rad=1e-3, timeout=40.0)

    # Rotate grip to move in a direction of -x+y
    approach_dir = normalize_vector(np.array([-1.0,1.0,0.0], dtype=np.float64))

    # Prepare to slide into safe's mouth
    pre_pos = safe_position + 2*approach_dir*Lm
    pre_pos[0] += 0.075
    pre_pos[2] = safe_position[2] + 0.07
    obs, reward, done = move(env, task, target_pos=pre_pos, timeout=20.0)
    obs, reward, done = align_two_axes(env, task, local_axes=('z','y'), world_axes=(approach_dir, 'x'), axis_dirs=(-1,1), tol_rad=1e-3, timeout=30.0)
    pre_pos = safe_position.copy()
    pre_pos[2] = safe_position[2] + 0.07
    obs, reward, done = move(env, task, target_pos=pre_pos, timeout=10.0)
    obs, reward, done = open_gripper(env, task, timeout=5.0)
    
    if done:
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")