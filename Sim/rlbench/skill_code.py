import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
import time
from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError, PyRepError
from rlbench.backend.exceptions import InvalidActionError
from utils.feedback import FeedbackWithError

from rlbench.backend.observation import Observation
import imageio
import cv2
def save_camera_image(obs: Observation, cam_name, save_path, use_cv2=False):
    cam_map = {
        'cam_front': obs.front_rgb,
        'cam_over_shoulder_left': obs.left_shoulder_rgb,
        'cam_over_shoulder_right': obs.right_shoulder_rgb,
        'cam_wrist': obs.wrist_rgb,
        'cam_overhead': obs.overhead_rgb,
    }
    if cam_name not in cam_map:
        raise ValueError(f"Unknown camera name: {cam_name}")

    img = cam_map[cam_name]


    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)


    if not use_cv2:

        imageio.imwrite(save_path, img)
    else:

        import cv2
        img_bgr = img[..., ::-1]
        cv2.imwrite(save_path, img_bgr)
   
   
def normalize_quaternion(quat):
    return quat / np.linalg.norm(quat)

def euler_from_quat(quat_xyzw, seq='xyz'):
    rot = R.from_quat(quat_xyzw)
    return rot.as_euler(seq, degrees=True)

def safe_step(env, task, action, skill_type, step_index=-1, waypoint_index=-1, waypoints=None, threshold=None, original_pos=None):
    """
    Call task.step(action) and handle exceptions
    """
    try:
        obs, reward, done = task.step(action)
    except ValueError as e:
        fd_error = FeedbackWithError(
            env=env,
            task=task,
            skill_type=skill_type,
            attempted_action=action,
            object_positions={},
            robot_pos=task.get_observation().gripper_pose,
            waypoints=waypoints,
            waypoint_index=waypoint_index,
            step_index=step_index,
            original_robot_pos=original_pos,
            error_message=f"Invalid action format: {e}"
        )
        raise SkillFailure(fd_error)
    
    except (ConfigurationError, ConfigurationPathError, IKError, PyRepError, InvalidActionError) as e:
        fd_error = FeedbackWithError(
            env=env,
            task=task,
            skill_type=skill_type,
            attempted_action=action,
            object_positions={},
            robot_pos=task.get_observation().gripper_pose,
            waypoints=waypoints,
            waypoint_index=waypoint_index,
            step_index=step_index,
            original_robot_pos=original_pos,
            error_message=f"Invalid action format: {e}"
        )
        raise PathOutOfWorkspace(fd_error)
    
    return obs, reward, done

def pick(env, task, target_pos, approach_distance=0.15, max_steps=100, threshold=0.01, approach_axis='z', timeout=10.0):
    print("========== [pick] START ==========")

    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()
    print(f"[pick] Initial gripper pos: {start_pos}")
    print(f"[pick] Initial gripper quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")

    approach_dir_map = {
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
    }

    approach_dir = approach_dir_map[approach_axis]
    approach_pos = target_pos + approach_dir * approach_distance
    
    print(f"[pick] target_pos: {target_pos}")
    print(f"[pick] approach_pos: {approach_pos} (approach_axis: {approach_axis})")
    
    waypoints = []
    NUM_WAYPOINT_1, NUM_WAYPOINT_2 = 5, 10
    for t in range(NUM_WAYPOINT_1):
        alpha = t / (NUM_WAYPOINT_1 - 1) if NUM_WAYPOINT_1 > 1 else 1
        wp = start_pos + alpha * (approach_pos - start_pos)
        waypoints.append(wp)

    for t in range(NUM_WAYPOINT_2):
        alpha = t / (NUM_WAYPOINT_2 - 1) if NUM_WAYPOINT_2 > 1 else 1
        wp = approach_pos + alpha * (target_pos - approach_pos)
        waypoints.append(wp)

    action = np.zeros(env.action_shape)
    action[-1] = 1.0
    for i, waypoint in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps:
            obs = task.get_observation()
            current_pos = obs.gripper_pose[:3]
            dist_to_wp = np.linalg.norm(waypoint - current_pos)
            if dist_to_wp < threshold:
                print(f"      -> Reached waypoint {i}.")
                break
            action[:3] = waypoint
            action[3:7] = start_quat

            obs, reward, done = safe_step(env, task, action, skill_type="pick", waypoint_index=i, step_index=step_count, 
                                          waypoints=waypoints, original_pos=original_pos, threshold=threshold)
            
            step_count += 1
            if done:
                if reward >= 1.0:
                    print("[pick] Task successfully ended during movement!")
                    return obs, reward, done
                else:
                    print("[pick] Task ended with failure during movement!")
                    fd_error = FeedbackWithError(env=env, task=task,skill_type="pick",attempted_action=action,robot_pos=obs.gripper_pose,
                                                 waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                                 original_robot_pos=original_pos,
                                                 error_message=f"Task ended with failure at waypoint {i}.")
                    raise SkillFailure(fd_error)
                    
            if time.time() - start_time > timeout:
                print(f"[pick] Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="pick",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                raise SkillFailure(fd_error)
        else:
            fd_error = FeedbackWithError(env=env, task=task,skill_type="pick",attempted_action=action,robot_pos=obs.gripper_pose,
                                        waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                        original_robot_pos=original_pos,
                                        error_message=f"Exceeded max_steps({max_steps}) at waypoint {i}.")
            raise SkillFailure(fd_error)
        save_camera_image(obs, 'cam_front', f'/workspace/rlbench_server/logs/pick_step{i}_front.png')
    
    print("[pick] Ready to Pick...")

    for _ in range(5):
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    print("[pick] Closing gripper...")
    action[-1] = -1.0
    
    for _ in range(10):
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    print("[pick] Done pick process.")
    
    final_gripper_state = getattr(obs, 'gripper_open', 1.0)










        
    return obs, reward, done

def place(env, task, target_pos, approach_distance=0.15, max_steps=100, threshold=0.01, approach_axis='z', timeout=10.0):
    print("========== [place] START ==========")
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()

    print(f"[place] Initial gripper pos: {start_pos}")
    print(f"[place] Initial gripper quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")
    approach_dir_map = {
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
    }
    approach_dir = approach_dir_map[approach_axis]
    approach_pos = target_pos + approach_dir * approach_distance
    print(f"[place] target_pos: {target_pos}")
    print(f"[place] approach_pos: {approach_pos} (approach_axis: {approach_axis})")
    waypoints = []
    NUM_WAYPOINT_1, NUM_WAYPOINT_2 = 5, 10

    for t in range(NUM_WAYPOINT_1):
        alpha = t / (NUM_WAYPOINT_1 - 1) if NUM_WAYPOINT_1 > 1 else 1
        wp = start_pos + alpha * (approach_pos - start_pos)
        waypoints.append(wp)
    for t in range(NUM_WAYPOINT_2):
        alpha = t / (NUM_WAYPOINT_2 - 1) if NUM_WAYPOINT_2 > 1 else 1
        wp = approach_pos + alpha * (target_pos - approach_pos)
        waypoints.append(wp)
    action = np.zeros(env.action_shape)
    action[-1] = -1.0

    for i, waypoint in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps:
            obs = task.get_observation()
            current_pos = obs.gripper_pose[:3]
            dist_to_wp = np.linalg.norm(waypoint - current_pos)
            if dist_to_wp < threshold:
                print(f"      -> Reached waypoint {i}.")
                break
            action[:3] = waypoint
            action[3:7] = start_quat

            obs, reward, done = safe_step(env, task, action, skill_type="place", waypoint_index=i, step_index=step_count, 
                                          waypoints=waypoints, original_pos=original_pos, threshold=threshold)
            
            step_count += 1
            if done:
                if reward >= 1.0:
                    print("[place] Task successfully ended during movement!")
                    return obs, reward, done
                else:
                    print("[place] Task ended with failure during movement!")
                    fd_error = FeedbackWithError(env=env, task=task,skill_type="place",attempted_action=action,robot_pos=obs.gripper_pose,
                                                 waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                                 original_robot_pos=original_pos,
                                                 error_message=f"Task ended with failure at waypoint {i}.")
                    raise SkillFailure(fd_error)
                    
            if time.time() - start_time > timeout:
                print(f"[place] Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="place",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                raise SkillFailure(fd_error)
        else:
            fd_error = FeedbackWithError(env=env, task=task,skill_type="place",attempted_action=action,robot_pos=obs.gripper_pose,
                                        waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                        original_robot_pos=original_pos,
                                        error_message=f"Exceeded max_steps({max_steps}) at waypoint {i}.")
            raise SkillFailure(fd_error)
        save_camera_image(obs, 'cam_front', f'/workspace/rlbench_server/logs/place_step{i}_front.png')
                
    for _ in range(5):
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    print("[place] Opening gripper...")
    action[-1] = 1.0
    for _ in range(10):
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done
    print("[place] Done place process.")

    final_gripper_state = getattr(obs, 'gripper_open', 1.0)

    if final_gripper_state == 1.0:
        print("[place] Gripper successfully open.")
    else:
        print("[place] Warning: Gripper did not open properly.")
        fd_error = FeedbackWithError(env=env, task=task,skill_type="place",attempted_action=action,robot_pos=obs.gripper_pose,
                                    waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                    original_robot_pos=original_pos,
                                    error_message="Gripper did not open properly after place operation.")
        raise SkillFailure(fd_error)

    return obs, reward, done

def move(env, task, target_pos, max_steps=100, threshold=0.01, timeout=10.0):
    """
    Move the gripper to a target position while maintaining current orientation and gripper state.
    
    Args:
        env: The environment object providing action shape and other properties.
        task: The task object providing observations and step functionality.
        target_pos (np.ndarray): Target position [x, y, z] to move the gripper to.
        max_steps (int): Maximum steps to reach each waypoint (default: 100).
        threshold (float): Distance threshold to consider a waypoint reached (default: 0.01).
        timeout (float): Maximum time in seconds to reach a waypoint before timeout (default: 10.0).
    
    Returns:
        tuple: (obs, reward, done) from the final step.
    """
    print("========== [move] START ==========")
    

    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()
    


    initial_gripper_state = getattr(obs, 'gripper_open', 1.0)
    
    print(f"[move] Initial gripper state: {initial_gripper_state}")
    print(f"[move] Initial gripper pos: {start_pos}")
    print(f"[move] Initial gripper quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")
    print(f"[move] Target pos: {target_pos}")
    

    NUM_WAYPOINT = 10
    waypoints = []
    for t in range(NUM_WAYPOINT):
        alpha = t / (NUM_WAYPOINT - 1) if NUM_WAYPOINT > 1 else 1
        wp = start_pos + alpha * (target_pos - start_pos)
        waypoints.append(wp)
    

    action = np.zeros(env.action_shape)
    action[-1] = initial_gripper_state
    

    for i, waypoint in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps:
            obs = task.get_observation()
            current_pos = obs.gripper_pose[:3]
            dist_to_wp = np.linalg.norm(waypoint - current_pos)
            if dist_to_wp < threshold:
                print(f"      -> Reached waypoint {i}.")
                break
            action[:3] = waypoint
            action[3:7] = start_quat

            obs, reward, done = safe_step(env, task, action, skill_type="move", waypoint_index=i, step_index=step_count, 
                                          waypoints=waypoints, original_pos=original_pos, threshold=threshold)

            step_count += 1

            if done:
                if reward >= 1.0:
                    print("[move] Task successfully ended during movement!")
                    return obs, reward, done
                else:
                    print("[move] Task ended with failure during movement!")
                    fd_error = FeedbackWithError(env=env, task=task,skill_type="move",attempted_action=action,robot_pos=obs.gripper_pose,
                                                 waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                                 original_robot_pos=original_pos,
                                                 error_message=f"Task ended with failure at waypoint {i}.")
                    raise SkillFailure(fd_error)
                    
            if time.time() - start_time > timeout:
                print(f"[move] Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="move",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                raise SkillFailure(fd_error)
        else:
            fd_error = FeedbackWithError(env=env, task=task,skill_type="move",attempted_action=action,robot_pos=obs.gripper_pose,
                                        waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                        original_robot_pos=original_pos,
                                        error_message=f"Exceeded max_steps({max_steps}) at waypoint {i}.")
            raise SkillFailure(fd_error)
    

    for _ in range(5):
        action[:3] = target_pos
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done
    
    print("[move] Done move process.")
    return obs, reward, done





import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np, time
from env_utils import _to_world_vec

def align_two_axes(env, task,
                   local_axes=('x','y'),
                   world_axes=('z','y'),
                   axis_dirs=(1,1),
                   tol_rad=1e-3,
                   max_steps=100, timeout=10.0):
    """
    그리퍼 로컬 두 축을 월드 두 축에 동시에 정렬하여
    완전한 방향 고정을 수행.
    """

    obs = task.get_observation()
    pos = obs.gripper_pose[:3]
    curr_quat = normalize_quaternion(obs.gripper_pose[3:7])
    curr_rot = R.from_quat(curr_quat)
    original_pos = obs.gripper_pose.copy()


    axis_map = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z': np.array([0,0,1])}
    a = []
    b = []
    for la, wa, d in zip(local_axes, world_axes, axis_dirs):
        a.append(curr_rot.apply(axis_map[la]))
        b.append(d * _to_world_vec(wa))



    R_opt, _ = R.align_vectors(b, a)
    target_quat = (R_opt * curr_rot).as_quat()





    action = np.zeros(env.action_shape)
    action[-1] = getattr(obs, 'gripper_open', 1.0)
    start = time.time()
    step = 0
    while True:
        obs = task.get_observation()
        curr_quat = normalize_quaternion(obs.gripper_pose[3:7])

        if angle_diff(curr_quat, target_quat) < tol_rad or step >= max_steps:
            break
        action[:3] = pos
        action[3:7] = target_quat

        obs, reward, done = safe_step(env, task, action, skill_type="align_two_axes", waypoint_index=-1, step_index=step, 
                                          waypoints=None, original_pos=original_pos)

        step += 1
        if done:
            if reward >= 1.0:
                print("[align_two_axes] Task successfully ended during movement!")
                return obs, reward, done
            else:
                print("[align_two_axes] Task ended with failure during movement!")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="align_two_axes",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=None, waypoint_index=-1, step_index=step, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Task ended with failure at step {step}.")
                raise SkillFailure(fd_error)

        if time.time() - start > timeout:
            break

    for _ in range(5):
        action[:3] = pos
        action[3:7] = target_quat
        obs, reward, done = task.step(action)
        if done:
            break

    return obs, reward, done

def normalize_quaternion(q):
    """쾌속 정규화 함수"""
    return q / np.linalg.norm(q)

def angle_diff(q1, q2):
    """두 쿼터니언 간 회전각 차이 계산"""
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()

import math
import inspect

def _quat_mul(q1, q2):

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def _quat_from_euler(roll, pitch, yaw):
    cr = math.cos(roll/2);  sr = math.sin(roll/2)
    cp = math.cos(pitch/2); sp = math.sin(pitch/2)
    cy = math.cos(yaw/2);   sy = math.sin(yaw/2)

    return np.array([
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
        cr*cp*cy + sr*sp*sy
    ], dtype=float)

_flip_dict = {

    'down' : np.array([1.0, 0.0, 0.0, 0.0]),


    'right' : _quat_from_euler(0.0, math.radians(90), 0.0),


    'left'  : _quat_from_euler(0.0, math.radians(-90), 0.0),


    'front' : _quat_from_euler(math.radians(-90), 0.0, 0.0),


    'back'  : _quat_from_euler(math.radians(90), 0.0, 0.0),
}

def _yaw_from_quat(q):
    x, y, z, w = q


    t0 = 2.0*(w*z + x*y)
    t1 = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(t0, t1)

_approach_axis_dict = {
    'down' : 'z',
    'right' : 'x',
    'left' : '-x',
    'front' : 'y',
    'back' : '-y',
}

def align_to_quaternion(env, task,
                        object_quaternion,
                        yaw_align,
                        approach_direction,
                        tol_rad=1e-3,
                        max_steps=100,
                        timeout=10.0):
    """
    현재 그리퍼 방향을 target_quat에 맞추도록 회전시킴.
    
    Args:
        env: RLBench 환경 객체
        task: RLBench TaskEnvironment 객체
        target_quat (array-like, shape=(4,)): 목표 방향 쿼터니언 [x, y, z, w]
        tol_rad: 회전 수렴 허용 오차 (라디안)
        max_steps: 최대 스텝 수
        timeout: 최대 대기 시간 (초)
    Returns:
        obs, reward, done: 마지막 step 결과
    """

    obs = task.get_observation()
    pos = obs.gripper_pose[:3]
    action = np.zeros(env.action_shape)
    action[-1] = getattr(obs, 'gripper_open', 1.0)
    original_pos = obs.gripper_pose.copy()


    yaw = _yaw_from_quat(object_quaternion)
    flip_quat = _flip_dict[approach_direction]
    yaw_offset = (0.0 if (yaw_align == 'parallel') else (math.pi / 2.0))
    yaw_quat = _quat_from_euler(0.0, 0.0, (yaw + yaw_offset))
    target_quat = _quat_mul(yaw_quat, flip_quat)
    target_quat = (target_quat / np.linalg.norm(target_quat))


    target_quat = normalize_quaternion(target_quat)


    start_time = time.time()
    step = 0
    while True:
        obs = task.get_observation()
        curr_quat = normalize_quaternion(obs.gripper_pose[3:7])


        if angle_diff(curr_quat, target_quat) < tol_rad:
            break
        if step >= max_steps:
            break

        if time.time() - start_time > timeout:
            print(f"[align_to_quaternion] Timeout: Failed to reach waypoint {step} within {timeout} seconds.")





            print("??")
            return obs, reward, done


        action[:3] = pos
        action[3:7] = target_quat
        
        obs, reward, done = safe_step(env, task, action, skill_type="align_to_quaternion", waypoint_index=-1, step_index=step, 
                                          waypoints=None, original_pos=original_pos, threshold=None)
        if done:
            if reward >= 1.0:
                print("[align_to_quaternion] Task successfully ended during movement!")
                return obs, reward, done
            else:
                print("[align_to_quaternion] Task ended with failure during movement!")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="align_to_quaternion",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=None, waypoint_index=-1, step_index=step, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Task ended with failure at waypoint {i}.")
                raise SkillFailure(fd_error)

        step += 1


    for _ in range(5):
        action[:3] = pos
        action[3:7] = target_quat
        obs, reward, done = task.step(action)
        if done:
            break

    return obs, reward, done

def open_gripper(env, task,
                 num_steps: int = 10,
                 timeout: float = 5.0):
    """
    Open the gripper smoothly.
    """
    obs = task.get_observation()
    success, terminate = task._task.success()
    if success:
        return obs, 1.0, True
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    action = np.zeros(env.action_shape)
    action[:3] = start_pos
    action[3:7] = start_quat
    action[-1] = 1.0

    for i in range(10):
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done
        
    return obs, reward, done

def close_gripper(env, task,
                 num_steps: int = 10,
                 timeout: float = 5.0):
    """
    Close the gripper smoothly.
    """
    obs = task.get_observation()
    success, terminate = task._task.success()
    if success:
        return obs, 1.0, True
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    action = np.zeros(env.action_shape)
    action[:3] = start_pos
    action[3:7] = start_quat
    action[-1] = 0.0

    for _ in range(10):
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done
        
    return obs, reward, done

def push(env, task, target_pos, approach_distance=0.1, max_steps=100, threshold=0.01, approach_axis='z', timeout=10.0):
    print("========== [push] START ==========")

    obs = task.get_observation()
    success, terminate = task._task.success()
    if success:
        return obs, 1.0, True
        
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()
    print(f"[push] Initial gripper pos: {start_pos}")
    print(f"[push] Initial gripper quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")

    approach_dir_map = {
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
    }

    approach_dir = approach_dir_map[approach_axis]
    approach_pos = target_pos + approach_dir * approach_distance
    
    print(f"[push] target_pos: {target_pos}")
    print(f"[push] approach_pos: {approach_pos} (approach_axis: {approach_axis})")
    
    waypoints = []
    NUM_WAYPOINT_1, NUM_WAYPOINT_2 = 5, 10
    for t in range(NUM_WAYPOINT_1):
        alpha = t / (NUM_WAYPOINT_1 - 1) if NUM_WAYPOINT_1 > 1 else 1
        wp = start_pos + alpha * (approach_pos - start_pos)
        waypoints.append(wp)

    for t in range(NUM_WAYPOINT_2):
        alpha = t / (NUM_WAYPOINT_2 - 1) if NUM_WAYPOINT_2 > 1 else 1
        wp = approach_pos + alpha * (target_pos - approach_pos)
        waypoints.append(wp)

    action = np.zeros(env.action_shape)
    action[-1] = -1.0
    
    for i, waypoint in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps:
            obs = task.get_observation()
            current_pos = obs.gripper_pose[:3]
            dist_to_wp = np.linalg.norm(waypoint - current_pos)
            if dist_to_wp < threshold:
                print(f"      -> Reached waypoint {i}.")
                break
            action[:3] = waypoint
            action[3:7] = start_quat

            obs, reward, done = safe_step(env, task, action, skill_type="push", waypoint_index=i, step_index=step_count, 
                                          waypoints=waypoints, original_pos=original_pos, threshold=threshold)
            
            step_count += 1
            if done:
                if reward >= 1.0:
                    print("[push] Task successfully ended during movement!")
                    return obs, reward, done
                else:
                    print("[push] Task ended with failure during movement!")
                    fd_error = FeedbackWithError(env=env, task=task,skill_type="push",attempted_action=action,robot_pos=obs.gripper_pose,
                                                 waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                                 original_robot_pos=original_pos,
                                                 error_message=f"Task ended with failure at waypoint {i}.")
                    raise SkillFailure(fd_error)
                    
            if time.time() - start_time > timeout:
                print(f"[push] Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                fd_error = FeedbackWithError(env=env, task=task,skill_type="push",attempted_action=action,robot_pos=obs.gripper_pose,
                                            waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                            original_robot_pos=original_pos,
                                            error_message=f"Timeout: Failed to reach waypoint {i} within {timeout} seconds.")
                raise SkillFailure(fd_error)
        else:
            fd_error = FeedbackWithError(env=env, task=task,skill_type="push",attempted_action=action,robot_pos=obs.gripper_pose,
                                        waypoints=waypoints, waypoint_index=i, step_index=step_count, 
                                        original_robot_pos=original_pos,
                                        error_message=f"Exceeded max_steps({max_steps}) at waypoint {i}.")
            raise SkillFailure(fd_error)
        obs = task.get_observation()
        save_camera_image(obs, 'cam_front', f'/workspace/rlbench_server/logs/push_step{i}_front.png')
    
    for _ in range(5):
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    print("[push] Done push process.")
        
    return obs, reward, done





def open_and_retract(env, task,
                     retract_dir=np.array([0, 0, 0.005]),
                     dist=0.04,
                     steps=15):
    """
    현재 위치에서 그리퍼를 열면서(release) 지정한 방향으로 dist만큼 직선 retract.
    - 열기 = 1.0, 닫기 = -1.0 가정
    - 안전/간단 버전: 복잡한 보정, wiggle 없음
    """

    obs = task.get_observation()
    pos = obs.gripper_pose[:3].copy()
    quat = obs.gripper_pose[3:7].copy()
    action = np.zeros(env.action_shape, dtype=float)


    retract_dir = np.asarray(retract_dir, dtype=float)
    if np.linalg.norm(retract_dir) < 1e-9:
        retract_dir = np.array([0, 0, 1.0])
    retract_dir = retract_dir / np.linalg.norm(retract_dir)
    end_pos = pos - retract_dir * float(dist)


    action[:3] = pos
    action[3:7] = quat
    action[-1] = 1.0
    obs, reward, done = task.step(action)
    if done:
        return obs, reward, done


    steps = max(1, int(steps))
    for k in range(steps):
        a = (k + 1) / steps
        p = pos * (1 - a) + end_pos * a
        action[:3] = p
        action[3:7] = quat
        action[-1] = 1.0
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    return obs, reward, done
