
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push
from env_utils import quat_mul, get_bbox_sizes, normalize_vector
import numpy as np

def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):
    stick_object = Shape("stick")
    cube_object = Shape("cube")
    target0_object = Shape("target0")
    stick_position = np.array(stick_object.get_position(), dtype=np.float64)
    cube_position = np.array(cube_object.get_position(), dtype=np.float64)
    target0_position = np.array(target0_object.get_position(), dtype=np.float64)
    
    stick_sizes, _ = get_bbox_sizes(stick_object)
    l = float(np.max(stick_sizes))
    
    stick_quat = normalize_quaternion(np.array(stick_object.get_quaternion(), dtype=np.float64))
    stick_axis = R.from_quat(stick_quat).apply([0,1,0])
    stick_axis[2] = 0.0
    stick_axis = normalize_vector(stick_axis)

    stick_perp = normalize_vector(np.array([-stick_axis[1], stick_axis[0], 0.0], dtype=np.float64))
    

    grasp_end = stick_position.copy()
    grasp_end[0] += 0.03
    grasp_end[1] += l/3
    grasp_end[2] -= 0.04
    stick_hover = grasp_end.copy()
    stick_hover[2] += 0.2
    obs, reward, done = move(env, task, target_pos=stick_hover, timeout=20.0)
    obs, reward, done = align_two_axes(env, task, local_axes=('z','x'), world_axes=('z', stick_perp), axis_dirs=(-1,1), tol_rad=2e-3, timeout=30.0)
    obs, reward, done = pick(env, task, target_pos=grasp_end, approach_distance=0.04, approach_axis='z', timeout=10.0)
    

    obs, reward, done = move(env, task, target_pos=stick_hover, timeout=10.0)
    drag_dir = (target0_position - cube_position).astype(np.float64)
    drag_dir[2] = 0.0
    drag_dir = normalize_vector(drag_dir)
    prep_drag = grasp_end - 0.1*drag_dir
    obs, reward, done = move(env, task, target_pos=prep_drag, timeout=10.0)
    

    drag_pos = target0_position + np.array([0.0, 0.2, 0.0])
    obs, reward, done = push(env, task, target_pos=drag_pos, approach_distance=0.1, approach_axis='y', timeout=20.0)

    
    if done: 
        print("[Task] Task successfully completed (done=True).")
        return obs, reward, done
    else:
        print("[Task] Task not completed yet (done=False).")