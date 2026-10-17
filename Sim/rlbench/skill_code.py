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

    # 1) 스케일 처리: float[0,1] -> uint8[0,255], 이미 uint8이면 그대로
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

    # 2) 저장
    if not use_cv2:
        # imageio는 RGB 그대로 저장 -> 색 뒤집힘 없음
        imageio.imwrite(save_path, img)
    else:
        # OpenCV는 BGR 기대 -> RGB를 BGR로 바꿔서 저장해야 색이 정상
        import cv2
        img_bgr = img[..., ::-1]  # RGB->BGR
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
    # gripper state 확인
    # if final_gripper_state < 1.0:
    #     print("[pick] Gripper successfully closed.")
    # else:
    #     print("[pick] Warning: Gripper did not close properly.")
    #     fd_error = FeedbackWithError(env=env, task=task,skill_type="pick", attempted_action=action, robot_pos=obs.gripper_pose,
    #                                     waypoints=None, waypoint_index=-1, step_index=-1, 
    #                                     original_robot_pos=original_pos,
    #                                     error_message=f"Gripper did not close properly after pick operation.")
    #     raise SkillFailure(fd_error)
        
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
    # gripper state 확인
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
    
    # Get initial observation
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()
    
    # Assume gripper_open exists in observation (1.0 for open, -1.0 for closed)
    # Note: This assumption is based on typical robotic environments; adjust if different
    initial_gripper_state = getattr(obs, 'gripper_open', 1.0)  # Default to 1.0 if not found
    
    print(f"[move] Initial gripper state: {initial_gripper_state}")
    print(f"[move] Initial gripper pos: {start_pos}")
    print(f"[move] Initial gripper quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")
    print(f"[move] Target pos: {target_pos}")
    
    # Create waypoints from start to target position
    NUM_WAYPOINT = 10
    waypoints = []
    for t in range(NUM_WAYPOINT):
        alpha = t / (NUM_WAYPOINT - 1) if NUM_WAYPOINT > 1 else 1
        wp = start_pos + alpha * (target_pos - start_pos)
        waypoints.append(wp)
    
    # Initialize action array
    action = np.zeros(env.action_shape)
    action[-1] = initial_gripper_state  # Maintain current gripper state
    
    # Move to each waypoint
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
    
    # Hold at target position for stability
    for _ in range(5):
        action[:3] = target_pos
        action[3:7] = start_quat
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done
    
    print("[move] Done move process.")
    return obs, reward, done

###########################################################################################
############################### New Skill Code for Rotation ###############################
###########################################################################################

import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np, time
from env_utils import _to_world_vec

def align_two_axes(env, task,
                   local_axes=('x','y'),    # 그리퍼 로컬에서 정렬할 축 2개
                   world_axes=('z','y'),    # 월드에서 목표 축 2개
                   axis_dirs=(1,1),         # 각 축의 ± 방향
                   tol_rad=1e-3,            # 허용 회전각 [rad]
                   max_steps=100, timeout=10.0):
    """
    그리퍼 로컬 두 축을 월드 두 축에 동시에 정렬하여
    완전한 방향 고정을 수행.
    """
    # 1) 현재 포즈
    obs = task.get_observation()
    pos = obs.gripper_pose[:3]
    curr_quat = normalize_quaternion(obs.gripper_pose[3:7])
    curr_rot = R.from_quat(curr_quat)
    original_pos = obs.gripper_pose.copy()

    # 2) 벡터 준비
    axis_map = {'x': np.array([1,0,0]), 'y': np.array([0,1,0]), 'z': np.array([0,0,1])}
    a = []  # 현재 로컬→월드 벡터
    b = []  # 목표 벡터
    for la, wa, d in zip(local_axes, world_axes, axis_dirs):
        a.append(curr_rot.apply(axis_map[la]))
        b.append(d * _to_world_vec(wa))

    # 3) 두 쌍에 대한 최적 회전 계산
    #   R_opt: a_i → b_i 가 가장 최소 오차가 나도록 하는 회전
    R_opt, _ = R.align_vectors(b, a)
    target_quat = (R_opt * curr_rot).as_quat()

    # 4) step 루프: 회전각 기준으로 수렴 여부 판단
#    def angle_diff(q1, q2):
#        return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()

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
    # 안정화
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
    # (x, y, z, w)
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
    # XYZ (roll, pitch, yaw) → quaternion (x,y,z,w)
    return np.array([
        sr*cp*cy - cr*sp*sy,   # x
        cr*sp*cy + sr*cp*sy,   # y
        cr*cp*sy - sr*sp*cy,   # z
        cr*cp*cy + sr*sp*sy    # w
    ], dtype=float)

_flip_dict = {
    # Top-down (현재 코드와 동일: +Z → -Z)
    'down' : np.array([1.0, 0.0, 0.0, 0.0]),  # x축 180도

    # 접근축이 월드 +X (오른쪽에서)
    'right' : _quat_from_euler(0.0, math.radians(90), 0.0),   # pitch=+90°

    # 접근축이 월드 -X (왼쪽에서)
    'left'  : _quat_from_euler(0.0, math.radians(-90), 0.0),  # pitch=-90°

    # 접근축이 월드 +Y (앞에서)
    'front' : _quat_from_euler(math.radians(-90), 0.0, 0.0),  # roll=-90°

    # 접근축이 월드 -Y (뒤에서)
    'back'  : _quat_from_euler(math.radians(90), 0.0, 0.0),   # roll=+90°
}

def _yaw_from_quat(q):
    x, y, z, w = q
    # ZYX 컨벤션에서 yaw(psi) 추출
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
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
                        tol_rad=1e-3,      # 허용 회전 오차 [rad]
                        max_steps=100,     # 최대 스텝
                        timeout=10.0):     # 타임아웃 [초]
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
    # 1) 초기 포즈 및 action 벡터 준비
    obs = task.get_observation()
    pos = obs.gripper_pose[:3]
    action = np.zeros(env.action_shape)
    action[-1] = getattr(obs, 'gripper_open', 1.0)  # 그리퍼 개방도 고정
    original_pos = obs.gripper_pose.copy()

    # target quat 계산
    yaw = _yaw_from_quat(object_quaternion)
    flip_quat = _flip_dict[approach_direction]
    yaw_offset = (0.0 if (yaw_align == 'parallel') else (math.pi / 2.0))
    yaw_quat = _quat_from_euler(0.0, 0.0, (yaw + yaw_offset))
    target_quat = _quat_mul(yaw_quat, flip_quat)
    target_quat = (target_quat / np.linalg.norm(target_quat))

    # 2) 쿼터니언 정규화
    target_quat = normalize_quaternion(target_quat)

    # 3) 수렴 루프
    start_time = time.time()
    step = 0
    while True:
        obs = task.get_observation()
        curr_quat = normalize_quaternion(obs.gripper_pose[3:7])

        # 수렴 조건
        if angle_diff(curr_quat, target_quat) < tol_rad:
            break
        if step >= max_steps:
            break

        if time.time() - start_time > timeout:
            print(f"[align_to_quaternion] Timeout: Failed to reach waypoint {step} within {timeout} seconds.")
            # fd_error = FeedbackWithError(env=env, task=task,skill_type="align_to_quaternion",attempted_action=action,robot_pos=obs.gripper_pose,
            #                             waypoints=None, waypoint_index=-1, step_index=step, 
            #                             original_robot_pos=original_pos,
            #                             error_message=f"Timeout while reaching waypoint {step} during align_to_quaternion process.")
            # raise SkillFailure(fd_error)
            print("??")
            return obs, reward, done

        # 위치 유지, 방향 목표로 설정
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

    # 4) 안정화를 위해 몇 번 더 반복
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

##########################################################################################
########################## New Skill Code for Higher level Exec ##########################
##########################################################################################

def open_and_retract(env, task,
                     retract_dir=np.array([0, 0, 0.005]),  # 빼는 방향(기본: +z 위로)
                     dist=0.04,                          # 4cm 정도만 빼기
                     steps=15):                          # 보간 스텝
    """
    현재 위치에서 그리퍼를 열면서(release) 지정한 방향으로 dist만큼 직선 retract.
    - 열기 = 1.0, 닫기 = -1.0 가정
    - 안전/간단 버전: 복잡한 보정, wiggle 없음
    """
    # 현재 포즈
    obs = task.get_observation()
    pos = obs.gripper_pose[:3].copy()
    quat = obs.gripper_pose[3:7].copy()  # 별도 정규화 안 함
    action = np.zeros(env.action_shape, dtype=float)

    # 목표점 계산
    retract_dir = np.asarray(retract_dir, dtype=float)
    if np.linalg.norm(retract_dir) < 1e-9:
        retract_dir = np.array([0, 0, 1.0])  # 방향이 0이면 위로
    retract_dir = retract_dir / np.linalg.norm(retract_dir)
    end_pos = pos - retract_dir * float(dist)

    # 1) 일단 현재에서 열기 명령
    action[:3] = pos
    action[3:7] = quat
    action[-1] = 1.0
    obs, reward, done = task.step(action)
    if done:
        return obs, reward, done

    # 2) 열린 상태 유지하며 직선 retract
    steps = max(1, int(steps))
    for k in range(steps):
        a = (k + 1) / steps
        p = pos * (1 - a) + end_pos * a
        action[:3] = p
        action[3:7] = quat
        action[-1] = 1.0  # 계속 열린 상태 유지
        obs, reward, done = task.step(action)
        if done:
            return obs, reward, done

    return obs, reward, done
