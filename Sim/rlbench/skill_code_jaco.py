"""
Jaco Primitive Skills for RLBench

Jaco는 6 DOF manipulator로, Sawyer와 마찬가지로 path planning 기반의
안정적인 동작을 사용합니다. Step-by-step 방식 대신 get_linear_path /
get_nonlinear_path를 사용하여 IK 실패 문제를 해결합니다.

Primitives:
  - jaco_move_to: 위치 이동 (orientation 유지)
  - jaco_move_arc: 호 궤적 이동
  - jaco_align_gripper: 방향 정렬 (position 유지)
  - jaco_open_gripper: 그리퍼 열기
  - jaco_close_gripper: 그리퍼 닫기
  - jaco_grasp_at: 특정 위치에서 grasp 수행
  - jaco_release_at: 특정 위치에서 release 수행
  - jaco_move_to_home: 홈 위치로 이동
"""

import numpy as np
import math
import time
import imageio
from scipy.spatial.transform import Rotation as R

from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError, IKError

from rlbench.backend.observation import Observation
from rlbench.backend.exceptions import InvalidActionError
from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.feedback import FeedbackWithError


# =============================================================================
# Constants
# =============================================================================

# Best home joint configuration found via find_jaco_home.py
# This configuration puts the tip at [0.156, -0.111, 1.335] which is inside workspace
JACO_HOME_JOINTS = [0.0, 1.571, 1.571, 3.142, 1.571, 0.0]

# =============================================================================
# Common Utilities
# =============================================================================

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


def normalize_quaternion(q):
    """Quaternion 정규화 (xyzw 형식)"""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


def euler_from_quat(quat_xyzw, seq='xyz'):
    rot = R.from_quat(quat_xyzw)
    return rot.as_euler(seq, degrees=True)


def angle_diff(q1, q2):
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()


def _get_arm(task):
    """task에서 arm 객체 추출"""
    if hasattr(task, '_robot'):
        return task._robot.arm
    elif hasattr(task, '_task') and hasattr(task._task, 'robot'):
        return task._task.robot.arm
    raise RuntimeError("Cannot access robot arm from task")


def _get_gripper(task):
    """task에서 gripper 객체 추출"""
    if hasattr(task, '_robot'):
        return task._robot.gripper
    elif hasattr(task, '_task') and hasattr(task._task, 'robot'):
        return task._task.robot.gripper
    raise RuntimeError("Cannot access robot gripper from task")


def _get_scene(env):
    """env에서 scene 객체 추출"""
    if hasattr(env, '_scene'):
        return env._scene
    raise RuntimeError("Cannot access scene from env")


def _check_success(task):
    """task success 체크"""
    if hasattr(task, '_task'):
        success, _ = task._task.success()
        return success
    return False


def _execute_path(env, task, path, skill_type, timeout_s=60.0, target_pos=None):
    """
    Path를 실행하고 결과 반환

    Args:
        env: RLBench environment
        task: RLBench task
        path: ArmConfigurationPath 객체
        skill_type: 스킬 이름 (에러 메시지용)
        timeout_s: 타임아웃 (초)
        target_pos: 목표 위치 (디버그 및 fallback용)

    Returns:
        obs, reward, done
    """
    scene = _get_scene(env)
    arm = _get_arm(task)
    start_time = time.time()

    # 디버그: path 정보
    path_configs = path._path_points if hasattr(path, '_path_points') else None
    print(f"[_execute_path] Starting path execution, timeout={timeout_s}s")
    if path_configs is not None:
        print(f"[_execute_path] Path has {len(path_configs)} configurations")

    start_pos = np.array(arm.get_tip().get_position())
    print(f"[_execute_path] Start tip position: {start_pos}")
    if target_pos is not None:
        print(f"[_execute_path] Target position: {target_pos}")

    done_path = False
    step_count = 0
    prev_pos = start_pos.copy()

    print(f"[_execute_path] Starting while loop...")

    # 직접 joint interpolation 방식 사용 (path.step()이 블로킹되는 문제 우회)
    # path._path_points에서 joint configurations를 가져와서 직접 설정
    use_direct_interpolation = True

    if use_direct_interpolation and path_configs is not None and len(path_configs) > 0:
        num_joints = len(arm.joints)

        # path_configs 구조 분석: 1D flattened array인지 2D array인지 확인
        path_configs_arr = np.array(path_configs)
        print(f"[_execute_path] path_configs shape: {path_configs_arr.shape}, num_joints: {num_joints}")

        # 1D flattened array인 경우 reshape
        if path_configs_arr.ndim == 1:
            total_configs = len(path_configs_arr) // num_joints
            path_configs_arr = path_configs_arr.reshape(total_configs, num_joints)
            print(f"[_execute_path] Reshaped to {path_configs_arr.shape}")

        print(f"[_execute_path] Using direct joint interpolation with {len(path_configs_arr)} configs...")

        for config_idx, joint_pos in enumerate(path_configs_arr):
            # 타임아웃 체크
            if time.time() - start_time > timeout_s:
                obs = task.get_observation()
                fd_error = FeedbackWithError(
                    env=env, task=task, skill_type=skill_type,
                    attempted_action=None, object_positions={},
                    robot_pos=obs.gripper_pose,
                    waypoints=None, waypoint_index=-1, step_index=-1,
                    original_robot_pos=None,
                    error_message=f"Path execution timeout after {timeout_s}s"
                )
                raise SkillFailure(fd_error)

            arm.set_joint_target_positions(list(joint_pos))
            scene.step()
            step_count += 1

            # 30 step마다 로그 출력
            if step_count % 30 == 0:
                current_pos_now = np.array(arm.get_tip().get_position())
                if target_pos is not None:
                    dist_to_target = np.linalg.norm(current_pos_now - np.array(target_pos))
                    print(f"[_execute_path] Config {config_idx+1}/{len(path_configs_arr)}: pos={current_pos_now}, dist={dist_to_target:.4f}m")
                else:
                    print(f"[_execute_path] Config {config_idx+1}/{len(path_configs_arr)}: pos={current_pos_now}")

            # Task success 체크
            if _check_success(task):
                obs = task.get_observation()
                print(f"[_execute_path] Task success at config {config_idx+1}")
                return obs, 1.0, True

        done_path = True
        print(f"[_execute_path] Direct interpolation completed in {step_count} steps")
    else:
        # 기존 path.step() 방식 (fallback)
        print(f"[_execute_path] Using path.step() method...")
        while not done_path:
            # 타임아웃 체크
            if time.time() - start_time > timeout_s:
                obs = task.get_observation()
                fd_error = FeedbackWithError(
                    env=env, task=task, skill_type=skill_type,
                    attempted_action=None, object_positions={},
                    robot_pos=obs.gripper_pose,
                    waypoints=None, waypoint_index=-1, step_index=-1,
                    original_robot_pos=None,
                    error_message=f"Path execution timeout after {timeout_s}s"
                )
                raise SkillFailure(fd_error)

            # Path step 실행
            if step_count == 0:
                print(f"[_execute_path] About to call path.step() for step 0...")
            done_path = path.step()
            if step_count == 0:
                print(f"[_execute_path] path.step() returned, about to call scene.step()...")
            scene.step()
            step_count += 1

            # 디버그: 매 step마다 위치 변화 추적
            current_pos_now = np.array(arm.get_tip().get_position())

            # 10 step마다 또는 마지막 step에서 출력
            if step_count % 10 == 0 or done_path:
                if target_pos is not None:
                    dist_to_target = np.linalg.norm(current_pos_now - np.array(target_pos))
                    print(f"[_execute_path] Step {step_count}: pos={current_pos_now}, dist_to_target={dist_to_target:.4f}m, done_path={done_path}")
                else:
                    print(f"[_execute_path] Step {step_count}: pos={current_pos_now}, done_path={done_path}")

            # 움직임이 거의 없는데 done_path=True면 조기종료 감지
            if done_path and step_count < 50:
                total_moved = np.linalg.norm(current_pos_now - start_pos)
                if target_pos is not None:
                    remaining = np.linalg.norm(current_pos_now - np.array(target_pos))
                    print(f"[_execute_path] WARNING: Path ended early! steps={step_count}, moved={total_moved:.4f}m, remaining={remaining:.4f}m")

                    # Joint limit 체크
                    print(f"[_execute_path] Checking joint limits...")
                    for j_idx, joint in enumerate(arm.joints):
                        interval = joint.get_joint_interval()
                        pos = joint.get_joint_position()
                        at_limit = ""
                        if interval[0]:  # cyclic이 아닌 경우만
                            if pos <= interval[1][0] + 0.02:
                                at_limit = " <-- AT LOWER LIMIT!"
                            elif pos >= interval[1][1] - 0.02:
                                at_limit = " <-- AT UPPER LIMIT!"
                        print(f"[_execute_path]   Joint {j_idx}: pos={pos:.4f}, limits=[{interval[1][0]:.4f}, {interval[1][1]:.4f}]{at_limit}")

            prev_pos = current_pos_now.copy()

            # Task success 체크
            if _check_success(task):
                obs = task.get_observation()
                print(f"[_execute_path] Task success at step {step_count}")
                return obs, 1.0, True

    # Path 완료 후 디버그 출력
    end_pos = np.array(arm.get_tip().get_position())
    print(f"[_execute_path] Path completed in {step_count} steps")
    print(f"[_execute_path] End tip position: {end_pos}")

    if target_pos is not None:
        target_pos_arr = np.array(target_pos)
        error = np.linalg.norm(end_pos - target_pos_arr)
        print(f"[_execute_path] Position error to target: {error:.4f}m")
        print(f"[_execute_path] Position diff: {end_pos - target_pos_arr}")

        # Path가 목표에 도달하지 못했으면 direct joint interpolation 시도
        if error > 0.05:  # 5cm 이상 오차
            print(f"[_execute_path] Path didn't reach target, trying direct joint interpolation...")
            current_quat = arm.get_tip().get_quaternion()

            try:
                # IK로 목표 joint position 계산
                target_joints = arm.solve_ik_via_jacobian(target_pos_arr.tolist(), quaternion=current_quat)
                print(f"[_execute_path] IK solution found, interpolating...")

                current_joints = np.array(arm.get_joint_positions())
                target_joints = np.array(target_joints)
                n_interp = 50

                for interp_i in range(1, n_interp + 1):
                    alpha = interp_i / n_interp
                    interp_joints = current_joints + alpha * (target_joints - current_joints)
                    arm.set_joint_positions(interp_joints.tolist())
                    scene.step()

                    # Task success 체크
                    if _check_success(task):
                        obs = task.get_observation()
                        print(f"[_execute_path] Task success during direct interpolation!")
                        return obs, 1.0, True

                    # 진행 상황 출력
                    if interp_i % 10 == 0:
                        curr_pos = np.array(arm.get_tip().get_position())
                        dist = np.linalg.norm(curr_pos - target_pos_arr)
                        print(f"[_execute_path] Direct interp step {interp_i}/{n_interp}: dist_to_target={dist:.4f}m")

                final_pos = np.array(arm.get_tip().get_position())
                final_error = np.linalg.norm(final_pos - target_pos_arr)
                print(f"[_execute_path] Direct interpolation done. Final error: {final_error:.4f}m")

            except IKError as e:
                print(f"[_execute_path] IK failed for direct interpolation: {e}")

    # Path 완료 후 observation 반환
    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    return obs, reward, success


def _get_gripper_state(obs):
    """Get current gripper state from observation."""
    return float(getattr(obs, 'gripper_open', 1.0))


# =============================================================================
# Quaternion Utilities for Alignment
# =============================================================================

def _quat_from_euler(roll, pitch, yaw):
    """Euler angles (XYZ) to quaternion (xyzw)"""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    return np.array([
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
        cr * cp * cy + sr * sp * sy   # w
    ], dtype=float)


def _quat_mul(q1, q2):
    """Quaternion multiplication (xyzw format)"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ], dtype=float)


def _yaw_from_quat(q):
    """Quaternion에서 yaw 추출 (xyzw format)"""
    x, y, z, w = q
    t0 = 2.0 * (w * z + x * y)
    t1 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t0, t1)


# Approach direction별 flip quaternion (gripper z축을 해당 방향으로)
_FLIP_QUATS = {
    'down':  np.array([1.0, 0.0, 0.0, 0.0], dtype=float),  # 180° about X (z -> -z)
    'up':    np.array([0.0, 0.0, 0.0, 1.0], dtype=float),  # identity (z -> +z)
    'right': _quat_from_euler(0.0, math.radians(90), 0.0),   # pitch +90°
    'left':  _quat_from_euler(0.0, math.radians(-90), 0.0),  # pitch -90°
    'front': _quat_from_euler(math.radians(-90), 0.0, 0.0),  # roll -90°
    'back':  _quat_from_euler(math.radians(90), 0.0, 0.0),   # roll +90°
    'z':     np.array([1.0, 0.0, 0.0, 0.0], dtype=float),    # same as 'down'
    '-z':    np.array([0.0, 0.0, 0.0, 1.0], dtype=float),    # same as 'up'
    'x':     _quat_from_euler(0.0, math.radians(90), 0.0),
    '-x':    _quat_from_euler(0.0, math.radians(-90), 0.0),
    'y':     _quat_from_euler(math.radians(-90), 0.0, 0.0),
    '-y':    _quat_from_euler(math.radians(90), 0.0, 0.0),
}


def _compute_align_quaternion(approach_direction, reference_quat=None, yaw_mode='parallel'):
    """
    Approach direction과 reference quaternion으로 목표 quaternion 계산

    Args:
        approach_direction: 'down', 'up', 'left', 'right', 'front', 'back'
        reference_quat: 참조 quaternion (yaw 추출용, optional)
        yaw_mode: 'parallel' or 'perpendicular'

    Returns:
        목표 quaternion (xyzw)
    """
    if approach_direction not in _FLIP_QUATS:
        raise ValueError(f"Unknown approach_direction: {approach_direction}. "
                        f"Must be one of {list(_FLIP_QUATS.keys())}")

    flip_quat = _FLIP_QUATS[approach_direction]

    # Reference quat에서 yaw 추출
    if reference_quat is not None:
        ref_q = normalize_quaternion(reference_quat)
        yaw = _yaw_from_quat(ref_q)
    else:
        yaw = 0.0

    # Yaw mode에 따른 offset
    yaw_offset = 0.0 if yaw_mode == 'parallel' else (math.pi / 2.0)

    # Yaw quaternion
    yaw_quat = _quat_from_euler(0.0, 0.0, yaw + yaw_offset)

    # 최종 quaternion: yaw 적용 후 flip
    target_quat = _quat_mul(yaw_quat, flip_quat)

    return normalize_quaternion(target_quat)


# =============================================================================
# Primitive Skills
# =============================================================================

def _move_joints_to_config(env, task, target_joints, n_steps=50):
    """
    Directly move joints to target configuration using interpolation.
    This bypasses IK and path planning - use only for recovery.

    Args:
        env: RLBench environment
        task: RLBench task
        target_joints: Target joint positions (list of 6 floats for Jaco)
        n_steps: Number of interpolation steps
    """
    arm = _get_arm(task)
    scene = _get_scene(env)

    current_joints = arm.get_joint_positions()
    target_joints = np.array(target_joints, dtype=float)
    current_joints = np.array(current_joints, dtype=float)

    print(f"[_move_joints_to_config] current_joints: {current_joints}")
    print(f"[_move_joints_to_config] target_joints: {target_joints}")

    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        interp_joints = current_joints + alpha * (target_joints - current_joints)
        arm.set_joint_target_positions(interp_joints.tolist())
        scene.step()

    # Final stabilization
    for _ in range(10):
        arm.set_joint_target_positions(target_joints.tolist())
        scene.step()

    final_pos = arm.get_tip().get_position()
    print(f"[_move_joints_to_config] Final tip position: {final_pos}")


def _find_ik_solution(arm, target_pos, target_quat=None, trials=300):
    """
    Try to find IK solution using sampling, with fallback to position-only.

    Args:
        arm: Robot arm object
        target_pos: Target position [x, y, z]
        target_quat: Target quaternion (optional)
        trials: Number of IK trials

    Returns:
        joint_positions or None
    """
    from pyrep.errors import IKError

    # Try with full pose constraint first
    if target_quat is not None:
        try:
            configs = arm.solve_ik_via_sampling(
                position=target_pos,
                quaternion=target_quat,
                ignore_collisions=True,
                trials=trials,
                max_configs=1
            )
            if len(configs) > 0:
                return configs[0] if isinstance(configs[0], (list, np.ndarray)) else configs
        except (IKError, RuntimeError) as e:
            print(f"[_find_ik_solution] Full pose IK failed: {e}")

    # Try position-only (relaxed constraint)
    try:
        # Disable orientation constraint temporarily
        arm.set_ik_element_properties(
            constraint_x=True,
            constraint_y=True,
            constraint_z=True,
            constraint_alpha_beta=False,
            constraint_gamma=False
        )

        configs = arm.solve_ik_via_sampling(
            position=target_pos,
            quaternion=target_quat if target_quat is not None else [0, 0, 0, 1],
            ignore_collisions=True,
            trials=trials,
            max_configs=1
        )

        # Restore full constraints
        arm.set_ik_element_properties(
            constraint_x=True,
            constraint_y=True,
            constraint_z=True,
            constraint_alpha_beta=True,
            constraint_gamma=True
        )

        if len(configs) > 0:
            return configs[0] if isinstance(configs[0], (list, np.ndarray)) else configs

    except (IKError, RuntimeError) as e:
        print(f"[_find_ik_solution] Position-only IK failed: {e}")
        # Restore constraints
        try:
            arm.set_ik_element_properties(
                constraint_x=True,
                constraint_y=True,
                constraint_z=True,
                constraint_alpha_beta=True,
                constraint_gamma=True
            )
        except:
            pass

    return None


def jaco_move_to_home(
    env, task,
    home_pos: np.ndarray = None,
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Move Jaco robot to a safe home position inside the workspace.
    Uses multiple fallback strategies:
    1. Path planning with orientation
    2. IK sampling with position-only constraint
    3. Direct joint interpolation

    Args:
        env: RLBench environment
        task: RLBench task
        home_pos: Target home position [x, y, z]. If None, uses workspace center.
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_move_to_home] START ==========")

    arm = _get_arm(task)
    scene = _get_scene(env)
    current_pos = np.array(arm.get_tip().get_position())
    current_quat = arm.get_tip().get_quaternion()
    current_joints = arm.get_joint_positions()

    # Default home position: workspace center (safe position)
    if home_pos is None:
        home_pos = np.array([0.25, 0.0, 0.9], dtype=float)
    else:
        home_pos = np.array(home_pos, dtype=float)

    print(f"[jaco_move_to_home] current_pos: {current_pos}")
    print(f"[jaco_move_to_home] current_joints: {current_joints}")
    print(f"[jaco_move_to_home] target home_pos: {home_pos}")

    # Check workspace bounds
    workspace_x = [-0.275, 0.775]
    workspace_y = [-0.655, 0.655]
    workspace_z = [0.652, 1.752]

    in_workspace = (
        workspace_x[0] <= current_pos[0] <= workspace_x[1] and
        workspace_y[0] <= current_pos[1] <= workspace_y[1] and
        workspace_z[0] <= current_pos[2] <= workspace_z[1]
    )

    print(f"[jaco_move_to_home] in_workspace: {in_workspace}")

    # Check if already near home
    if np.linalg.norm(current_pos - home_pos) < 0.1:
        print("[jaco_move_to_home] Already near home position.")
        obs = task.get_observation()
        return obs, 0.0, False

    # Strategy 1: Try path planning
    path = None
    try:
        print("[jaco_move_to_home] Strategy 1: Trying linear path...")
        path = arm.get_linear_path(
            home_pos.tolist(),
            quaternion=current_quat,
            steps=50,
            ignore_collisions=True
        )
        print("[jaco_move_to_home] Linear path found.")
    except (ConfigurationPathError, RuntimeError) as e:
        print(f"[jaco_move_to_home] Linear path failed: {e}")

        try:
            print("[jaco_move_to_home] Trying nonlinear path...")
            path = arm.get_nonlinear_path(
                home_pos.tolist(),
                quaternion=current_quat,
                ignore_collisions=True,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
            print("[jaco_move_to_home] Nonlinear path found.")
        except (ConfigurationPathError, RuntimeError) as e2:
            print(f"[jaco_move_to_home] Nonlinear path also failed: {e2}")

    # Execute path if found
    if path is not None:
        obs, reward, done = _execute_path(env, task, path, "jaco_move_to_home", timeout_s, target_pos=home_pos.tolist())
        final_pos = arm.get_tip().get_position()
        print(f"[jaco_move_to_home] Final position: {final_pos}")
        return obs, reward, done

    # Strategy 2: Try IK sampling (position only)
    print("[jaco_move_to_home] Strategy 2: Trying IK sampling...")
    target_joints = _find_ik_solution(arm, home_pos.tolist(), None, trials)

    if target_joints is not None:
        print(f"[jaco_move_to_home] IK solution found: {target_joints}")
        _move_joints_to_config(env, task, target_joints, n_steps=100)
        obs = task.get_observation()
        final_pos = arm.get_tip().get_position()
        print(f"[jaco_move_to_home] Final position: {final_pos}")
        return obs, 0.0, False

    # Strategy 3: Just skip and continue
    print("[jaco_move_to_home] All strategies failed, continuing with current position...")
    obs = task.get_observation()
    return obs, 0.0, False


def jaco_move_to(
    env, task,
    target_pos: np.ndarray,
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Move gripper to target position (keep current orientation).

    Uses path planning (linear first, then nonlinear) to ensure
    stable motion even for 6 DOF Jaco arm.

    Args:
        env: RLBench environment
        task: RLBench task
        target_pos: Target position [x, y, z]
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_move_to] START ==========")

    arm = _get_arm(task)
    scene = _get_scene(env)
    target_pos = np.asarray(target_pos, dtype=float)

    # Keep current orientation
    current_quat = arm.get_tip().get_quaternion()
    current_pos = arm.get_tip().get_position()
    current_joint_pos = arm.get_joint_positions()

    print(f"[jaco_move_to] current_pos: {current_pos}")
    print(f"[jaco_move_to] target_pos: {target_pos}")
    print(f"[jaco_move_to] distance: {np.linalg.norm(target_pos - current_pos):.4f}m")
    print(f"[jaco_move_to] current_quat: {current_quat}")
    print(f"[jaco_move_to] current_joint_pos: {current_joint_pos}")

    # Try direct IK first to see if target is even reachable
    try:
        ik_joint_pos = arm.solve_ik_via_jacobian(target_pos, quaternion=current_quat)
        print(f"[jaco_move_to] IK via Jacobian succeeded: {ik_joint_pos}")
    except IKError as e:
        print(f"[jaco_move_to] IK via Jacobian failed: {e}")
        # Try without quaternion constraint
        try:
            ik_joint_pos = arm.solve_ik_via_jacobian(target_pos)
            print(f"[jaco_move_to] IK via Jacobian (no quat) succeeded: {ik_joint_pos}")
        except IKError as e2:
            print(f"[jaco_move_to] IK via Jacobian (no quat) also failed: {e2}")

    # Path planning with multiple fallback strategies
    path = None

    # Strategy 1: Try linear path with collision checking
    try:
        print("[jaco_move_to] Trying linear path...")
        path = arm.get_linear_path(
            target_pos.tolist(),
            quaternion=current_quat,
            steps=50,
            ignore_collisions=False
        )
        print("[jaco_move_to] Linear path found.")
    except (ConfigurationPathError, RuntimeError) as e:
        print(f"[jaco_move_to] Linear path failed: {e}")

    # Strategy 2: Try nonlinear path with collision checking
    if path is None:
        try:
            print("[jaco_move_to] Trying nonlinear path...")
            path = arm.get_nonlinear_path(
                target_pos.tolist(),
                quaternion=current_quat,
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
            print("[jaco_move_to] Nonlinear path found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_move_to] Nonlinear path failed: {e}")

    # Strategy 3: Try linear path ignoring collisions
    if path is None:
        try:
            print("[jaco_move_to] Trying linear path (ignore collisions)...")
            path = arm.get_linear_path(
                target_pos.tolist(),
                quaternion=current_quat,
                steps=50,
                ignore_collisions=True
            )
            print("[jaco_move_to] Linear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_move_to] Linear path (ignore collisions) failed: {e}")

    # Strategy 4: Try nonlinear path ignoring collisions with more trials
    if path is None:
        try:
            print("[jaco_move_to] Trying nonlinear path (ignore collisions, more trials)...")
            path = arm.get_nonlinear_path(
                target_pos.tolist(),
                quaternion=current_quat,
                ignore_collisions=True,
                trials=trials * 3,
                max_configs=max_configs * 2,
                algorithm=Algos.RRTConnect
            )
            print("[jaco_move_to] Nonlinear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_move_to] Nonlinear path (ignore collisions) failed: {e}")

    # NOTE: Removed "different orientation" strategies (4.5, 4.6) because they
    # generate paths that don't reach the target position accurately.
    # Instead, go directly to multi-waypoint approach which works better.

    # Strategy 5: Try multi-waypoint path planning (break long movements into smaller path segments)
    # Key insight: Short-distance path planning works! Use path planning for each segment.
    if path is None:
        try:
            print("[jaco_move_to] Trying multi-waypoint path planning approach...")

            # Calculate distance to target
            start_pos = np.array(current_pos)
            end_pos = np.array(target_pos)
            distance = np.linalg.norm(end_pos - start_pos)
            print(f"[jaco_move_to] Distance to target: {distance:.3f}")

            # If distance is large, break into waypoints
            # Even for short distances, if we're here, it means all simpler strategies failed
            if distance > 0.05:  # Lower threshold
                # Create intermediate waypoints with conservative segment size
                n_waypoints = max(2, int(np.ceil(distance / 0.10)))  # Segments of ~10cm
                print(f"[jaco_move_to] Creating {n_waypoints} waypoints (segments ~{distance/n_waypoints:.2f}m each)...")

                success = True

                for i in range(1, n_waypoints + 1):
                    alpha = i / n_waypoints
                    waypoint_pos = start_pos + alpha * (end_pos - start_pos)

                    print(f"[jaco_move_to] Segment {i}/{n_waypoints}: current={arm.get_tip().get_position()} -> waypoint={waypoint_pos}")

                    # Try path planning for this SHORT segment
                    segment_path = None
                    current_quat = arm.get_tip().get_quaternion()

                    # Try linear path first (works best for short distances)
                    try:
                        segment_path = arm.get_linear_path(
                            waypoint_pos.tolist(),
                            quaternion=current_quat,
                            ignore_collisions=False
                        )
                        print(f"[jaco_move_to]   Segment {i}: Linear path found")
                    except (ConfigurationPathError, RuntimeError) as e:
                        print(f"[jaco_move_to]   Segment {i}: Linear path failed: {e}")

                    # If linear failed, try with collision ignore
                    if segment_path is None:
                        try:
                            segment_path = arm.get_linear_path(
                                waypoint_pos.tolist(),
                                quaternion=current_quat,
                                ignore_collisions=True
                            )
                            print(f"[jaco_move_to]   Segment {i}: Linear path (ignore collisions) found")
                        except (ConfigurationPathError, RuntimeError) as e:
                            print(f"[jaco_move_to]   Segment {i}: Linear path (ignore collisions) failed: {e}")

                    # If still failed, try nonlinear path
                    if segment_path is None:
                        try:
                            print(f"[jaco_move_to]   Segment {i}: Trying nonlinear path...")
                            segment_path = arm.get_path(
                                waypoint_pos.tolist(),
                                quaternion=current_quat,
                                ignore_collisions=True,
                                trials=300,
                                max_configs=60,
                                max_time_ms=10000
                            )
                            print(f"[jaco_move_to]   Segment {i}: Nonlinear path found")
                        except (ConfigurationPathError, RuntimeError) as e:
                            print(f"[jaco_move_to]   Segment {i}: Nonlinear path failed: {e}")

                    # If still failed, try nonlinear without quaternion constraint
                    if segment_path is None:
                        try:
                            print(f"[jaco_move_to]   Segment {i}: Trying nonlinear (no quaternion)...")
                            segment_path = arm.get_path(
                                waypoint_pos.tolist(),
                                ignore_collisions=True,
                                trials=300,
                                max_configs=60,
                                max_time_ms=10000
                            )
                            print(f"[jaco_move_to]   Segment {i}: Nonlinear (no quaternion) found")
                        except (ConfigurationPathError, RuntimeError) as e:
                            print(f"[jaco_move_to]   Segment {i}: Nonlinear (no quaternion) failed: {e}")

                    # If still failed, try breaking into micro-segments from current position
                    if segment_path is None:
                        print(f"[jaco_move_to]   Segment {i}: Trying micro-segment approach...")
                        micro_target = waypoint_pos
                        micro_completed = 0
                        max_micro_attempts = 50  # Safety limit

                        for _ in range(max_micro_attempts):
                            current_pos_now = np.array(arm.get_tip().get_position())
                            remaining_dist = np.linalg.norm(micro_target - current_pos_now)

                            if remaining_dist < 0.01:  # Close enough
                                print(f"[jaco_move_to]     Micro: Close enough to target ({remaining_dist:.3f}m)")
                                micro_completed += 1
                                break

                            # Try progressively smaller step sizes
                            step_sizes = [0.02, 0.01, 0.005]
                            micro_path = None
                            used_step = None

                            for step_size in step_sizes:
                                if step_size > remaining_dist:
                                    continue

                                direction = (micro_target - current_pos_now) / remaining_dist
                                micro_waypoint = current_pos_now + direction * step_size
                                micro_quat = arm.get_tip().get_quaternion()

                                # Try linear first
                                try:
                                    micro_path = arm.get_linear_path(
                                        micro_waypoint.tolist(),
                                        quaternion=micro_quat,
                                        ignore_collisions=True
                                    )
                                    used_step = step_size
                                    break
                                except (ConfigurationPathError, RuntimeError):
                                    pass

                            # If direct path failed, try moving z first then xy
                            if micro_path is None and remaining_dist > 0.02:
                                z_diff = micro_target[2] - current_pos_now[2]
                                if abs(z_diff) > 0.005:
                                    # Try moving only in z direction first
                                    z_step = np.sign(z_diff) * min(0.01, abs(z_diff))
                                    z_waypoint = current_pos_now.copy()
                                    z_waypoint[2] += z_step
                                    try:
                                        micro_path = arm.get_linear_path(
                                            z_waypoint.tolist(),
                                            quaternion=micro_quat,
                                            ignore_collisions=True
                                        )
                                        used_step = abs(z_step)
                                        print(f"[jaco_move_to]     Micro: Using z-only step")
                                    except (ConfigurationPathError, RuntimeError):
                                        pass

                            # Try nonlinear as last resort
                            if micro_path is None:
                                for step_size in [0.01, 0.005]:
                                    if step_size > remaining_dist:
                                        continue
                                    direction = (micro_target - current_pos_now) / remaining_dist
                                    micro_waypoint = current_pos_now + direction * step_size
                                    try:
                                        micro_path = arm.get_path(
                                            micro_waypoint.tolist(),
                                            ignore_collisions=True,
                                            trials=100,
                                            max_configs=30,
                                            max_time_ms=2000
                                        )
                                        used_step = step_size
                                        break
                                    except (ConfigurationPathError, RuntimeError):
                                        pass

                            # Last resort: direct joint interpolation (bypass path planning)
                            if micro_path is None:
                                print(f"[jaco_move_to]     Micro: Trying direct joint interpolation...")
                                for step_size in [0.01, 0.005, 0.002]:
                                    if step_size > remaining_dist:
                                        continue
                                    direction = (micro_target - current_pos_now) / remaining_dist
                                    micro_waypoint = current_pos_now + direction * step_size

                                    # Try to find IK solution
                                    try:
                                        target_joints = arm.solve_ik_via_jacobian(
                                            micro_waypoint.tolist(),
                                            quaternion=micro_quat
                                        )
                                        # Got IK solution, move directly
                                        current_joints = np.array(arm.get_joint_positions())
                                        target_joints = np.array(target_joints)

                                        # Interpolate in joint space
                                        n_interp = 10
                                        for interp_i in range(1, n_interp + 1):
                                            alpha_interp = interp_i / n_interp
                                            interp_joints = current_joints + alpha_interp * (target_joints - current_joints)
                                            arm.set_joint_positions(interp_joints.tolist())
                                            scene.step()

                                        used_step = step_size
                                        micro_path = "direct"  # Mark as success
                                        print(f"[jaco_move_to]     Micro: Direct joint interpolation succeeded for {step_size}m")
                                        break
                                    except IKError:
                                        pass

                            if micro_path is not None:
                                if micro_path != "direct":
                                    # Execute path planning result
                                    micro_path.visualize()
                                    done_micro = False
                                    while not done_micro:
                                        done_micro = micro_path.step()
                                        scene.step()
                                        if _check_success(task):
                                            obs = task.get_observation()
                                            print(f"[jaco_move_to]   Task succeeded during micro-segment!")
                                            return obs, 1.0, True
                                else:
                                    # Direct joint interpolation already executed, just check success
                                    if _check_success(task):
                                        obs = task.get_observation()
                                        print(f"[jaco_move_to]   Task succeeded during direct interpolation!")
                                        return obs, 1.0, True
                                micro_completed += 1
                                new_dist = np.linalg.norm(micro_target - np.array(arm.get_tip().get_position()))
                                print(f"[jaco_move_to]     Micro {micro_completed}: OK step={used_step:.3f}m (remaining: {new_dist:.3f}m)")
                            else:
                                # Debug: print detailed failure info
                                print(f"[jaco_move_to]     Micro: All step sizes failed at remaining dist {remaining_dist:.3f}m")
                                print(f"[jaco_move_to]     DEBUG: current_pos={current_pos_now}")
                                print(f"[jaco_move_to]     DEBUG: micro_target={micro_target}")
                                print(f"[jaco_move_to]     DEBUG: final_target={end_pos}")
                                joint_positions = arm.get_joint_positions()
                                print(f"[jaco_move_to]     DEBUG: current_joints={joint_positions}")
                                print(f"[jaco_move_to]     DEBUG: joint_limits:")
                                for j_idx, joint in enumerate(arm.joints):
                                    interval = joint.get_joint_interval()
                                    pos = joint.get_joint_position()
                                    at_limit = ""
                                    if interval[0] and (pos <= interval[1][0] + 0.01 or pos >= interval[1][1] - 0.01):
                                        at_limit = " <-- NEAR LIMIT!"
                                    print(f"[jaco_move_to]       Joint {j_idx}: pos={pos:.4f}, limits={interval}{at_limit}")

                                break

                        # Consider partial success - if we moved at least some distance, continue
                        final_pos = np.array(arm.get_tip().get_position())
                        final_dist_to_target = np.linalg.norm(end_pos - final_pos)
                        print(f"[jaco_move_to]   Segment {i}: Completed {micro_completed} micro-steps, dist to final target: {final_dist_to_target:.3f}m")

                        if micro_completed > 0:
                            segment_path = True  # Mark as success to continue
                            print(f"[jaco_move_to]   Segment {i}: Partial progress made, continuing...")

                    if segment_path is not None:
                        if segment_path is not True:  # Skip if already executed via micro-segments
                            # Execute this segment
                            segment_path.visualize()
                            done_moving = False
                            while not done_moving:
                                done_moving = segment_path.step()
                                scene.step()

                                # Check task success during movement
                                if _check_success(task):
                                    obs = task.get_observation()
                                    print(f"[jaco_move_to]   Task succeeded during segment {i}!")
                                    return obs, 1.0, True

                            print(f"[jaco_move_to]   Segment {i}: Executed successfully")

                        # Check task success after segment completion
                        if _check_success(task):
                            obs = task.get_observation()
                            print(f"[jaco_move_to]   Task succeeded after segment {i}!")
                            return obs, 1.0, True
                    else:
                        print(f"[jaco_move_to]   Segment {i}: Failed - could not find path")
                        success = False
                        break

                if success:
                    print("[jaco_move_to] Multi-waypoint path planning succeeded!")
                    obs = task.get_observation()
                    return obs, 0.0, False
                else:
                    # Check how close we got to the target
                    final_pos = np.array(arm.get_tip().get_position())
                    final_dist = np.linalg.norm(end_pos - final_pos)
                    initial_dist = np.linalg.norm(end_pos - start_pos)
                    progress = 1.0 - (final_dist / initial_dist) if initial_dist > 0 else 0.0

                    print(f"[jaco_move_to] Multi-waypoint partial: moved {progress*100:.1f}% of distance")
                    print(f"[jaco_move_to]   Initial dist: {initial_dist:.3f}m, Final dist: {final_dist:.3f}m")

                    # If we made significant progress (>50%) or got within 10cm, consider it acceptable
                    if progress > 0.5 or final_dist < 0.10:
                        print(f"[jaco_move_to] Accepting partial progress - close enough to target")
                        obs = task.get_observation()
                        return obs, 0.0, False
                    else:
                        print("[jaco_move_to] Multi-waypoint path planning failed (insufficient progress)")
            else:
                # Distance is very short (< 5cm), try direct movement with very small steps
                print(f"[jaco_move_to] Distance very short ({distance:.4f}m), trying micro-step approach...")

                # Try linear interpolation in joint space
                try:
                    # Get target joint configuration via IK (without path planning)
                    target_joint_pos = None
                    try:
                        target_joint_pos = arm.solve_ik_via_jacobian(target_pos, quaternion=current_quat)
                        print(f"[jaco_move_to] Target IK found: {target_joint_pos}")
                    except IKError:
                        # Try without quaternion constraint
                        try:
                            target_joint_pos = arm.solve_ik_via_jacobian(target_pos)
                            print(f"[jaco_move_to] Target IK (no quat) found: {target_joint_pos}")
                        except IKError as e:
                            print(f"[jaco_move_to] Could not solve IK for target: {e}")

                    if target_joint_pos is not None:
                        # Move in small joint-space steps
                        current_joints = np.array(arm.get_joint_positions())
                        target_joints = np.array(target_joint_pos)
                        n_steps = 20

                        for step in range(1, n_steps + 1):
                            alpha = step / n_steps
                            intermediate_joints = current_joints + alpha * (target_joints - current_joints)
                            arm.set_joint_positions(intermediate_joints.tolist())
                            scene.step()

                            # Check task success
                            if _check_success(task):
                                obs = task.get_observation()
                                print(f"[jaco_move_to] Task succeeded during micro-step!")
                                return obs, 1.0, True

                        print("[jaco_move_to] Micro-step approach succeeded!")
                        obs = task.get_observation()
                        return obs, 0.0, False
                    else:
                        print("[jaco_move_to] Micro-step approach failed (no IK solution)")
                except Exception as e:
                    print(f"[jaco_move_to] Micro-step approach error: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[jaco_move_to] Multi-waypoint strategy error: {e}")
            import traceback
            traceback.print_exc()

    # If all strategies failed, raise error
    if path is None:
        obs = task.get_observation()
        fd_error = FeedbackWithError(
            env=env, task=task, skill_type="jaco_move_to",
            attempted_action=target_pos.tolist(), object_positions={},
            robot_pos=obs.gripper_pose,
            waypoints=None, waypoint_index=-1, step_index=-1,
            original_robot_pos=current_pos,
            error_message=f"All path planning strategies failed"
        )
        raise PathOutOfWorkspace(fd_error)

    # Execute path
    obs, reward, done = _execute_path(env, task, path, "jaco_move_to", timeout_s, target_pos=target_pos)

    print(f"[jaco_move_to] Done. reward={reward}, done={done}")
    return obs, reward, done


def jaco_move_arc(
    env, task,
    target_pos: np.ndarray,
    arc_height: float = 0.1,
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Arc movement to target position (curved trajectory via waypoints).

    Args:
        env: RLBench environment
        task: RLBench task
        target_pos: Target position [x, y, z]
        arc_height: Height of the arc above the straight line
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_move_arc] START ==========")

    arm = _get_arm(task)
    target_pos = np.asarray(target_pos, dtype=float)
    start_pos = np.array(arm.get_tip().get_position())
    current_quat = arm.get_tip().get_quaternion()

    # Calculate midpoint with arc
    mid_pos = (start_pos + target_pos) / 2.0
    mid_pos[2] += arc_height

    print(f"[jaco_move_arc] start_pos: {start_pos}")
    print(f"[jaco_move_arc] mid_pos: {mid_pos}")
    print(f"[jaco_move_arc] target_pos: {target_pos}")

    # Move to midpoint first
    try:
        path1 = arm.get_linear_path(
            mid_pos.tolist(),
            quaternion=current_quat,
            steps=30,
            ignore_collisions=False
        )
    except (ConfigurationPathError, RuntimeError):
        try:
            path1 = arm.get_nonlinear_path(
                mid_pos.tolist(),
                quaternion=current_quat,
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
        except (ConfigurationPathError, RuntimeError) as e:
            obs = task.get_observation()
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="jaco_move_arc",
                attempted_action=mid_pos.tolist(), object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=None, waypoint_index=-1, step_index=-1,
                original_robot_pos=start_pos.tolist(),
                error_message=f"Arc midpoint path failed: {e}"
            )
            raise PathOutOfWorkspace(fd_error)

    # Execute first part
    obs, reward, done = _execute_path(env, task, path1, "jaco_move_arc", timeout_s / 2, target_pos=mid_pos.tolist())
    if done:
        return obs, reward, done

    # Move from midpoint to target
    current_quat = arm.get_tip().get_quaternion()
    try:
        path2 = arm.get_linear_path(
            target_pos.tolist(),
            quaternion=current_quat,
            steps=30,
            ignore_collisions=False
        )
    except (ConfigurationPathError, RuntimeError):
        try:
            path2 = arm.get_nonlinear_path(
                target_pos.tolist(),
                quaternion=current_quat,
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
        except (ConfigurationPathError, RuntimeError) as e:
            obs = task.get_observation()
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="jaco_move_arc",
                attempted_action=target_pos.tolist(), object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=None, waypoint_index=-1, step_index=-1,
                original_robot_pos=mid_pos.tolist(),
                error_message=f"Arc target path failed: {e}"
            )
            raise PathOutOfWorkspace(fd_error)

    # Execute second part
    obs, reward, done = _execute_path(env, task, path2, "jaco_move_arc", timeout_s / 2, target_pos=target_pos.tolist())

    print(f"[jaco_move_arc] Done. reward={reward}, done={done}")
    return obs, reward, done


def jaco_align_gripper(
    env, task,
    approach_direction: str = "down",
    reference_quat: np.ndarray = None,
    yaw_mode: str = 'parallel',
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Align gripper to a specific approach direction (keep current position).

    Args:
        env: RLBench environment
        task: RLBench task
        approach_direction: 'down', 'up', 'left', 'right', 'front', 'back', 'z', '-z', etc.
        reference_quat: Reference quaternion for yaw alignment
        yaw_mode: 'parallel' or 'perpendicular'
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_align_gripper] START ==========")

    arm = _get_arm(task)
    scene = _get_scene(env)

    # Keep current position
    current_pos = np.array(arm.get_tip().get_position())
    current_quat = arm.get_tip().get_quaternion()

    # Calculate target quaternion
    target_quat = _compute_align_quaternion(approach_direction, reference_quat, yaw_mode)

    print(f"[jaco_align_gripper] current_pos: {current_pos}")
    print(f"[jaco_align_gripper] current_quat: {current_quat}")
    print(f"[jaco_align_gripper] approach_direction: {approach_direction}")
    print(f"[jaco_align_gripper] yaw_mode: {yaw_mode}")
    print(f"[jaco_align_gripper] target_quat: {target_quat}")

    path = None

    # Strategy 1: Linear path
    try:
        print("[jaco_align_gripper] Trying linear path...")
        path = arm.get_linear_path(
            current_pos.tolist(),
            quaternion=target_quat.tolist(),
            steps=50,
            ignore_collisions=False
        )
        print("[jaco_align_gripper] Linear path found.")
    except (ConfigurationPathError, RuntimeError) as e:
        print(f"[jaco_align_gripper] Linear path failed: {e}")

    # Strategy 2: Linear path (ignore collisions)
    if path is None:
        try:
            print("[jaco_align_gripper] Trying linear path (ignore collisions)...")
            path = arm.get_linear_path(
                current_pos.tolist(),
                quaternion=target_quat.tolist(),
                steps=50,
                ignore_collisions=True
            )
            print("[jaco_align_gripper] Linear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_align_gripper] Linear path (ignore collisions) failed: {e}")

    # Strategy 3: Nonlinear path
    if path is None:
        try:
            print("[jaco_align_gripper] Trying nonlinear path...")
            path = arm.get_nonlinear_path(
                current_pos.tolist(),
                quaternion=target_quat.tolist(),
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
            print("[jaco_align_gripper] Nonlinear path found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_align_gripper] Nonlinear path failed: {e}")

    # Strategy 4: Nonlinear path (ignore collisions)
    if path is None:
        try:
            print("[jaco_align_gripper] Trying nonlinear path (ignore collisions)...")
            path = arm.get_nonlinear_path(
                current_pos.tolist(),
                quaternion=target_quat.tolist(),
                ignore_collisions=True,
                trials=trials * 2,
                max_configs=max_configs * 2,
                algorithm=Algos.RRTConnect
            )
            print("[jaco_align_gripper] Nonlinear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[jaco_align_gripper] Nonlinear path (ignore collisions) failed: {e}")

    # Strategy 5: Direct joint interpolation with IK
    if path is None:
        print("[jaco_align_gripper] Trying direct joint interpolation...")
        try:
            # Get IK solution for target orientation
            target_joints = arm.solve_ik_via_jacobian(
                current_pos.tolist(),
                quaternion=target_quat.tolist()
            )
            print(f"[jaco_align_gripper] IK solution found: {target_joints}")

            # Interpolate in joint space
            current_joints = np.array(arm.get_joint_positions())
            target_joints = np.array(target_joints)
            n_interp = 20

            for interp_i in range(1, n_interp + 1):
                alpha = interp_i / n_interp
                interp_joints = current_joints + alpha * (target_joints - current_joints)
                arm.set_joint_positions(interp_joints.tolist())
                scene.step()

                # Check task success
                if _check_success(task):
                    obs = task.get_observation()
                    print(f"[jaco_align_gripper] Task succeeded during interpolation!")
                    return obs, 1.0, True

            print("[jaco_align_gripper] Direct joint interpolation succeeded.")
            obs = task.get_observation()
            success = _check_success(task)
            reward = 1.0 if success else 0.0
            print(f"[jaco_align_gripper] Done. reward={reward}, done={success}")
            return obs, reward, success

        except IKError as e:
            print(f"[jaco_align_gripper] IK failed: {e}")

    # Strategy 6: Incremental quaternion interpolation with micro-steps
    if path is None:
        print("[jaco_align_gripper] Trying incremental quaternion interpolation...")
        from scipy.spatial.transform import Slerp

        try:
            # Create quaternion interpolation
            current_quat_arr = np.array(current_quat)
            target_quat_arr = np.array(target_quat)

            # scipy Rotation uses [x, y, z, w] format
            rotations = R.from_quat([current_quat_arr, target_quat_arr])
            slerp = Slerp([0, 1], rotations)

            n_steps = 20
            success_steps = 0

            for step_i in range(1, n_steps + 1):
                alpha = step_i / n_steps
                interp_quat = slerp(alpha).as_quat().tolist()

                # Try to move to this intermediate orientation
                micro_path = None
                try:
                    micro_path = arm.get_linear_path(
                        current_pos.tolist(),
                        quaternion=interp_quat,
                        steps=10,
                        ignore_collisions=True
                    )
                except (ConfigurationPathError, RuntimeError):
                    pass

                if micro_path is None:
                    # Try direct IK
                    try:
                        target_joints = arm.solve_ik_via_jacobian(
                            current_pos.tolist(),
                            quaternion=interp_quat
                        )
                        current_joints = np.array(arm.get_joint_positions())
                        target_joints = np.array(target_joints)

                        for interp_i in range(1, 6):
                            a = interp_i / 5
                            interp_joints = current_joints + a * (target_joints - current_joints)
                            arm.set_joint_positions(interp_joints.tolist())
                            scene.step()

                        success_steps += 1
                        print(f"[jaco_align_gripper]   Step {step_i}/{n_steps}: OK (direct IK)")
                    except IKError:
                        print(f"[jaco_align_gripper]   Step {step_i}/{n_steps}: Failed")
                        break
                else:
                    # Execute micro path
                    micro_path.visualize()
                    done_micro = False
                    while not done_micro:
                        done_micro = micro_path.step()
                        scene.step()
                    success_steps += 1
                    print(f"[jaco_align_gripper]   Step {step_i}/{n_steps}: OK (path)")

                # Check task success
                if _check_success(task):
                    obs = task.get_observation()
                    print(f"[jaco_align_gripper] Task succeeded during step {step_i}!")
                    return obs, 1.0, True

            if success_steps > 0:
                print(f"[jaco_align_gripper] Incremental interpolation: {success_steps}/{n_steps} steps succeeded")
                obs = task.get_observation()
                success = _check_success(task)
                reward = 1.0 if success else 0.0
                print(f"[jaco_align_gripper] Done. reward={reward}, done={success}")
                return obs, reward, success

        except Exception as e:
            print(f"[jaco_align_gripper] Incremental interpolation error: {e}")

    # If all strategies failed
    if path is None:
        obs = task.get_observation()
        fd_error = FeedbackWithError(
            env=env, task=task, skill_type="jaco_align_gripper",
            attempted_action=None, object_positions={},
            robot_pos=obs.gripper_pose,
            waypoints=None, waypoint_index=-1, step_index=-1,
            original_robot_pos=None,
            error_message="All align path planning strategies failed"
        )
        raise PathOutOfWorkspace(fd_error)

    # Execute path
    obs, reward, done = _execute_path(env, task, path, "jaco_align_gripper", timeout_s)

    print(f"[jaco_align_gripper] Done. reward={reward}, done={done}")
    return obs, reward, done


def jaco_open_gripper(
    env, task,
    amount: float = 1.0,
    velocity: float = 0.2,
    timeout_s: float = 10.0
):
    """
    Open the gripper (no arm movement).

    Args:
        env: RLBench environment
        task: RLBench task
        amount: Open amount (0.0=closed, 1.0=fully open)
        velocity: Gripper velocity
        timeout_s: Timeout in seconds

    Returns:
        obs, reward, done
    """
    print("========== [jaco_open_gripper] START ==========")
    print(f"[jaco_open_gripper] amount: {amount}, velocity: {velocity}")

    gripper = _get_gripper(task)
    scene = _get_scene(env)

    # Clamp amount
    amount = float(np.clip(amount, 0.0, 1.0))

    # Release any grasped objects first
    gripper.release()

    # Actuate gripper
    done_gripper = False
    max_steps = 100
    step = 0
    start_time = time.time()

    while not done_gripper and step < max_steps:
        if time.time() - start_time > timeout_s:
            break

        done_gripper = gripper.actuate(amount, velocity=velocity)
        scene.step()
        step += 1

        # Check task success
        if _check_success(task):
            obs = task.get_observation()
            print(f"[jaco_open_gripper] Task success during gripper open.")
            return obs, 1.0, True

    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    gripper_state = getattr(obs, 'gripper_open', None)
    print(f"[jaco_open_gripper] Done. gripper_open={gripper_state}, steps={step}")

    return obs, reward, success


def jaco_close_gripper(
    env, task,
    amount: float = 0.0,
    velocity: float = 0.2,
    timeout_s: float = 10.0
):
    """
    Close the gripper (no arm movement).

    Args:
        env: RLBench environment
        task: RLBench task
        amount: Close amount (0.0=fully closed, 1.0=open)
        velocity: Gripper velocity
        timeout_s: Timeout in seconds

    Returns:
        obs, reward, done
    """
    print("========== [jaco_close_gripper] START ==========")
    print(f"[jaco_close_gripper] amount: {amount}, velocity: {velocity}")

    gripper = _get_gripper(task)
    scene = _get_scene(env)

    # Clamp amount
    amount = float(np.clip(amount, 0.0, 1.0))

    # Actuate gripper
    done_gripper = False
    max_steps = 100
    step = 0
    start_time = time.time()

    while not done_gripper and step < max_steps:
        if time.time() - start_time > timeout_s:
            break

        done_gripper = gripper.actuate(amount, velocity=velocity)
        scene.step()
        step += 1

        # Check task success
        if _check_success(task):
            obs = task.get_observation()
            print(f"[jaco_close_gripper] Task success during gripper close.")
            return obs, 1.0, True

    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    gripper_state = getattr(obs, 'gripper_open', None)
    print(f"[jaco_close_gripper] Done. gripper_open={gripper_state}, steps={step}")

    return obs, reward, success


def jaco_grasp_at(
    env, task,
    grasp_pos: np.ndarray,
    approach_distance: float = 0.15,
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Approach from above -> descend to grasp position -> close gripper.

    Args:
        env: RLBench environment
        task: RLBench task
        grasp_pos: Grasp position [x, y, z]
        approach_distance: Distance to approach from above
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_grasp_at] START ==========")

    grasp_pos = np.asarray(grasp_pos, dtype=float)
    approach_pos = grasp_pos + np.array([0, 0, approach_distance])

    print(f"[jaco_grasp_at] grasp_pos: {grasp_pos}")
    print(f"[jaco_grasp_at] approach_pos: {approach_pos}")

    # Open gripper first
    obs, reward, done = jaco_open_gripper(env, task, amount=1.0, timeout_s=5.0)
    if done:
        return obs, reward, done

    # Move to approach position
    obs, reward, done = jaco_move_to(env, task, approach_pos, timeout_s=timeout_s/3, trials=trials, max_configs=max_configs)
    if done:
        return obs, reward, done

    # Descend to grasp position
    obs, reward, done = jaco_move_to(env, task, grasp_pos, timeout_s=timeout_s/3, trials=trials, max_configs=max_configs)
    if done:
        return obs, reward, done

    # Close gripper
    obs, reward, done = jaco_close_gripper(env, task, amount=0.0, timeout_s=5.0)

    # Try to grasp objects
    gripper = _get_gripper(task)
    if hasattr(task, '_task'):
        for g_obj in task._task.get_graspable_objects():
            gripper.grasp(g_obj)

    print(f"[jaco_grasp_at] Done. reward={reward}, done={done}")
    return obs, reward, done


def jaco_release_at(
    env, task,
    place_pos: np.ndarray,
    approach_distance: float = 0.15,
    timeout_s: float = 60.0,
    trials: int = 300,
    max_configs: int = 10
):
    """
    Approach from above -> descend to place position -> open gripper.

    Args:
        env: RLBench environment
        task: RLBench task
        place_pos: Place position [x, y, z]
        approach_distance: Distance to approach from above
        timeout_s: Timeout in seconds
        trials: IK sampling attempts
        max_configs: Maximum configurations

    Returns:
        obs, reward, done
    """
    print("========== [jaco_release_at] START ==========")

    place_pos = np.asarray(place_pos, dtype=float)
    approach_pos = place_pos + np.array([0, 0, approach_distance])

    print(f"[jaco_release_at] place_pos: {place_pos}")
    print(f"[jaco_release_at] approach_pos: {approach_pos}")

    # Move to approach position
    obs, reward, done = jaco_move_to(env, task, approach_pos, timeout_s=timeout_s/3, trials=trials, max_configs=max_configs)
    if done:
        return obs, reward, done

    # Descend to place position
    obs, reward, done = jaco_move_to(env, task, place_pos, timeout_s=timeout_s/3, trials=trials, max_configs=max_configs)
    if done:
        return obs, reward, done

    # Open gripper to release
    obs, reward, done = jaco_open_gripper(env, task, amount=1.0, timeout_s=5.0)

    print(f"[jaco_release_at] Done. reward={reward}, done={done}")
    return obs, reward, done


# =============================================================================
# Composite Skills (Pick and Place) - Sawyer-compatible interface
# =============================================================================

def jaco_pick(
    env, task,
    target_object,
    target_pos=None,
    grasp_offset=np.array([0.0, 0.0, 0.0]),
    velocity=0.1,
    approach_height=0.15
):
    """
    Pick an object (composite skill).

    1. Open gripper
    2. Move to approach position (above target)
    3. Descend to grasp position
    4. Close gripper
    5. Explicitly attach object to gripper

    Args:
        env: RLBench environment
        task: RLBench task
        target_object: Object to pick (PyRep Shape object)
        target_pos: Target position (if None, uses target_object.get_position())
        grasp_offset: Offset from object position [x, y, z]
        velocity: Gripper closing velocity
        approach_height: Height above target to approach from

    Returns:
        obs, reward, done
    """
    print("========== [jaco_pick] START ==========")

    arm = _get_arm(task)
    scene = _get_scene(env)

    # Get target position
    if target_pos is None:
        target_pos = np.array(target_object.get_position(), dtype=float)
    else:
        target_pos = np.asarray(target_pos, dtype=float)

    grasp_offset = np.asarray(grasp_offset, dtype=float)
    grasp_pos = target_pos + grasp_offset
    approach_pos = grasp_pos + np.array([0.0, 0.0, approach_height])

    print(f"[jaco_pick] target_pos: {target_pos}")
    print(f"[jaco_pick] grasp_offset: {grasp_offset}")
    print(f"[jaco_pick] grasp_pos: {grasp_pos}")
    print(f"[jaco_pick] approach_pos: {approach_pos}")

    # Step 0: Open gripper first
    print(f"[jaco_pick] Opening gripper...")
    obs, reward, done = jaco_open_gripper(env, task, amount=1.0, timeout_s=5.0)
    if done:
        return obs, reward, done

    # Step 1: Move to approach position (above grasp)
    print(f"[jaco_pick] Moving to approach position...")
    obs, reward, done = jaco_move_to(env, task, approach_pos)
    if done:
        return obs, reward, done

    # Check current position and try to get closer if needed
    current_pos = np.array(arm.get_tip().get_position())
    approach_error = np.linalg.norm(current_pos - approach_pos)
    print(f"[jaco_pick] After approach move: current_pos={current_pos}, error={approach_error:.4f}m")

    # NOTE: Removed gripper alignment step - it was causing position drift
    # The robot will maintain its current orientation during descent

    # Step 2: Descend to grasp position
    print(f"[jaco_pick] Descending to grasp position...")
    obs, reward, done = jaco_move_to(env, task, grasp_pos)
    if done:
        return obs, reward, done

    # Check grasp position accuracy
    current_pos = np.array(arm.get_tip().get_position())
    grasp_error = np.linalg.norm(current_pos - grasp_pos)
    print(f"[jaco_pick] After grasp move: current_pos={current_pos}, error={grasp_error:.4f}m")

    # If still too far, try direct IK approach (without orientation constraint)
    if grasp_error > 0.03:  # More than 3cm error
        print(f"[jaco_pick] Position error too large ({grasp_error:.4f}m), trying direct IK (no orientation)...")
        try:
            # Try IK without orientation constraint first
            target_joints = arm.solve_ik_via_jacobian(grasp_pos.tolist())
            current_joints = np.array(arm.get_joint_positions())
            target_joints = np.array(target_joints)
            print(f"[jaco_pick] IK solution found (no orientation): {target_joints}")

            # Interpolate to target
            for i in range(50):
                alpha = (i + 1) / 50
                interp_joints = current_joints + alpha * (target_joints - current_joints)
                arm.set_joint_target_positions(interp_joints.tolist())
                scene.step()

            final_pos = np.array(arm.get_tip().get_position())
            final_error = np.linalg.norm(final_pos - grasp_pos)
            print(f"[jaco_pick] After direct IK: pos={final_pos}, error={final_error:.4f}m")
        except Exception as e:
            print(f"[jaco_pick] Direct IK (no orientation) failed: {e}")
            # Try with sampling-based IK (position only)
            try:
                print(f"[jaco_pick] Trying sampling-based IK...")
                # Disable orientation constraint for sampling
                arm.set_ik_element_properties(
                    constraint_x=True,
                    constraint_y=True,
                    constraint_z=True,
                    constraint_alpha_beta=False,
                    constraint_gamma=False
                )
                configs = arm.solve_ik_via_sampling(
                    position=grasp_pos.tolist(),
                    quaternion=[0, 0, 0, 1],  # Dummy quaternion (will be ignored due to constraints)
                    ignore_collisions=True,
                    trials=500,
                    max_configs=1
                )
                # Restore constraints
                arm.set_ik_element_properties(
                    constraint_x=True,
                    constraint_y=True,
                    constraint_z=True,
                    constraint_alpha_beta=True,
                    constraint_gamma=True
                )
                if len(configs) > 0:
                    target_joints = np.array(configs[0] if isinstance(configs[0], (list, np.ndarray)) else configs)
                    current_joints = np.array(arm.get_joint_positions())
                    print(f"[jaco_pick] Sampling IK solution found: {target_joints}")

                    for i in range(50):
                        alpha = (i + 1) / 50
                        interp_joints = current_joints + alpha * (target_joints - current_joints)
                        arm.set_joint_target_positions(interp_joints.tolist())
                        scene.step()

                    final_pos = np.array(arm.get_tip().get_position())
                    final_error = np.linalg.norm(final_pos - grasp_pos)
                    print(f"[jaco_pick] After sampling IK: pos={final_pos}, error={final_error:.4f}m")
            except Exception as e2:
                print(f"[jaco_pick] Sampling IK also failed: {e2}")

    # Step 3: Close gripper and grasp
    print(f"[jaco_pick] Closing gripper...")
    gripper = _get_gripper(task)

    # Try to grasp first (before closing) - this attaches object if close enough
    print(f"[jaco_pick] Attempting early grasp...")
    grasped = gripper.grasp(target_object)
    if grasped:
        print(f"[jaco_pick] Early grasp succeeded!")

    # Now close the gripper
    obs, reward, done = jaco_close_gripper(env, task, amount=0.0, velocity=velocity)
    if done:
        return obs, reward, done

    # Small delay for gripper to settle
    for _ in range(20):
        scene.step()

    # Step 4: Try to grasp again after gripper closed
    if not grasped:
        print(f"[jaco_pick] Attempting grasp after close...")
        grasped = gripper.grasp(target_object)

    if grasped:
        print(f"[jaco_pick] Object successfully grasped and attached to gripper!")
    else:
        print(f"[jaco_pick] WARNING: Failed to grasp object - not detected by proximity sensor")

    # Check for task success
    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    print(f"[jaco_pick] Done. grasped={grasped}, reward={reward}, done={success}")
    return obs, reward, success


def jaco_place(
    env, task,
    place_pos,
    place_offset=np.array([0.0, 0.0, 0.0])
):
    """
    Place an object (composite skill).

    1. Move to place_pos
    2. Explicitly detach object from gripper
    3. Open gripper

    Args:
        env: RLBench environment
        task: RLBench task
        place_pos: Position to place [x, y, z]
        place_offset: Offset from place_pos [x, y, z]

    Returns:
        obs, reward, done
    """
    print("========== [jaco_place] START ==========")

    place_pos = np.asarray(place_pos, dtype=float)
    place_offset = np.asarray(place_offset, dtype=float)
    final_pos = place_pos + place_offset

    print(f"[jaco_place] place_pos: {place_pos}")
    print(f"[jaco_place] place_offset: {place_offset}")
    print(f"[jaco_place] final_pos: {final_pos}")

    # Step 1: Move to place position
    print(f"[jaco_place] Moving to place position...")
    obs, reward, done = jaco_move_to(env, task, final_pos)
    if done:
        return obs, reward, done

    # Step 2: Release the object (detach from gripper)
    print(f"[jaco_place] Releasing object from gripper...")
    gripper = _get_gripper(task)
    gripper.release()

    # Small delay for object to settle
    scene = _get_scene(env)
    for _ in range(10):
        scene.step()

    # Step 3: Open gripper
    print(f"[jaco_place] Opening gripper...")
    obs, reward, done = jaco_open_gripper(env, task)
    if done:
        return obs, reward, done

    # Check for task success
    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    print(f"[jaco_place] Done. reward={reward}, done={success}")
    return obs, reward, success
