"""
Sawyer Primitive Skills for RLBench

Sawyer는 7 DOF redundant manipulator이므로, Jacobian IK 대신
path planning (sampling 기반)을 사용하여 안정적인 동작을 보장합니다.

Primitives:
  - sawyer_move_to: 위치 이동 (orientation 유지)
  - sawyer_align_gripper: 방향 정렬 (position 유지)
  - sawyer_open_gripper: 그리퍼 열기
  - sawyer_close_gripper: 그리퍼 닫기
"""

import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as R

from pyrep.const import ConfigurationPathAlgorithms as Algos
from pyrep.errors import ConfigurationPathError, IKError

from rlbench.backend.exceptions import InvalidActionError
from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.feedback import FeedbackWithError


SAWYER_MOVE_TO_DEBUG = False
EXECUTE_PATH_DEBUG = False

def _move_to_log(*args, **kwargs):
    """sawyer_move_to 전용 디버그 로그"""
    if SAWYER_MOVE_TO_DEBUG:
        print(*args, **kwargs)

def _exec_path_log(*args, **kwargs):
    """_execute_path 전용 디버그 로그"""
    if EXECUTE_PATH_DEBUG:
        print(*args, **kwargs)






def normalize_quaternion(q):
    """Quaternion 정규화 (xyzw 형식)"""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


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
        target_pos: 목표 위치 (디버그용)

    Returns:
        obs, reward, done
    """
    scene = _get_scene(env)
    arm = _get_arm(task)
    start_time = time.time()


    path_configs = path._path_points if hasattr(path, '_path_points') else None
    _exec_path_log(f"[_execute_path] Starting path execution, timeout={timeout_s}s")
    if path_configs is not None:
        _exec_path_log(f"[_execute_path] Path has {len(path_configs)} configurations")

    start_pos = np.array(arm.get_tip().get_position())
    _exec_path_log(f"[_execute_path] Start tip position: {start_pos}")
    if target_pos is not None:
        _exec_path_log(f"[_execute_path] Target position: {target_pos}")

    done_path = False
    step_count = 0
    prev_pos = start_pos.copy()

    while not done_path:

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


        done_path = path.step()
        scene.step()
        step_count += 1


        current_pos_now = np.array(arm.get_tip().get_position())


        if step_count % 10 == 0 or done_path:
            if target_pos is not None:
                dist_to_target = np.linalg.norm(current_pos_now - np.array(target_pos))
                _exec_path_log(f"[_execute_path] Step {step_count}: pos={current_pos_now}, dist_to_target={dist_to_target:.4f}m, done_path={done_path}")
            else:
                _exec_path_log(f"[_execute_path] Step {step_count}: pos={current_pos_now}, done_path={done_path}")


        if done_path and step_count < 50:
            total_moved = np.linalg.norm(current_pos_now - start_pos)
            if target_pos is not None:
                remaining = np.linalg.norm(current_pos_now - np.array(target_pos))
                _exec_path_log(f"[_execute_path] WARNING: Path ended early! steps={step_count}, moved={total_moved:.4f}m, remaining={remaining:.4f}m")


                _exec_path_log(f"[_execute_path] Checking joint limits...")
                for j_idx, joint in enumerate(arm.joints):
                    interval = joint.get_joint_interval()
                    pos = joint.get_joint_position()
                    at_limit = ""
                    if interval[0]:
                        if pos <= interval[1][0] + 0.02:
                            at_limit = " <-- AT LOWER LIMIT!"
                        elif pos >= interval[1][1] - 0.02:
                            at_limit = " <-- AT UPPER LIMIT!"
                    _exec_path_log(f"[_execute_path]   Joint {j_idx}: pos={pos:.4f}, limits=[{interval[1][0]:.4f}, {interval[1][1]:.4f}]{at_limit}")

        prev_pos = current_pos_now.copy()


        if _check_success(task):
            obs = task.get_observation()
            _exec_path_log(f"[_execute_path] Task success at step {step_count}")
            return obs, 1.0, True


    end_pos = np.array(arm.get_tip().get_position())
    _exec_path_log(f"[_execute_path] Path completed in {step_count} steps")
    _exec_path_log(f"[_execute_path] End tip position: {end_pos}")

    if target_pos is not None:
        target_pos_arr = np.array(target_pos)
        error = np.linalg.norm(end_pos - target_pos_arr)
        _exec_path_log(f"[_execute_path] Position error to target: {error:.4f}m")
        _exec_path_log(f"[_execute_path] Position diff: {end_pos - target_pos_arr}")


        if error > 0.05:
            _exec_path_log(f"[_execute_path] Path didn't reach target, trying direct joint interpolation...")
            current_quat = arm.get_tip().get_quaternion()

            try:

                target_joints = arm.solve_ik_via_jacobian(target_pos_arr.tolist(), quaternion=current_quat)
                _exec_path_log(f"[_execute_path] IK solution found, interpolating...")

                current_joints = np.array(arm.get_joint_positions())
                target_joints = np.array(target_joints)
                n_interp = 50

                for interp_i in range(1, n_interp + 1):
                    alpha = interp_i / n_interp
                    interp_joints = current_joints + alpha * (target_joints - current_joints)
                    arm.set_joint_positions(interp_joints.tolist())
                    scene.step()


                    if _check_success(task):
                        obs = task.get_observation()
                        _exec_path_log(f"[_execute_path] Task success during direct interpolation!")
                        return obs, 1.0, True


                    if interp_i % 10 == 0:
                        curr_pos = np.array(arm.get_tip().get_position())
                        dist = np.linalg.norm(curr_pos - target_pos_arr)
                        _exec_path_log(f"[_execute_path] Direct interp step {interp_i}/{n_interp}: dist_to_target={dist:.4f}m")

                final_pos = np.array(arm.get_tip().get_position())
                final_error = np.linalg.norm(final_pos - target_pos_arr)
                _exec_path_log(f"[_execute_path] Direct interpolation done. Final error: {final_error:.4f}m")

            except IKError as e:
                _exec_path_log(f"[_execute_path] IK failed for direct interpolation: {e}")


    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    return obs, reward, success






def _quat_from_euler(roll, pitch, yaw):
    """Euler angles (XYZ) to quaternion (xyzw)"""
    cr = math.cos(roll / 2)
    sr = math.sin(roll / 2)
    cp = math.cos(pitch / 2)
    sp = math.sin(pitch / 2)
    cy = math.cos(yaw / 2)
    sy = math.sin(yaw / 2)

    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy
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



_FLIP_QUATS = {
    'down':  np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    'up':    np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
    'right': _quat_from_euler(0.0, math.radians(90), 0.0),
    'left':  _quat_from_euler(0.0, math.radians(-90), 0.0),
    'front': _quat_from_euler(math.radians(-90), 0.0, 0.0),
    'back':  _quat_from_euler(math.radians(90), 0.0, 0.0),
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


    if reference_quat is not None:
        ref_q = normalize_quaternion(reference_quat)
        yaw = _yaw_from_quat(ref_q)
    else:
        yaw = 0.0


    yaw_offset = 0.0 if yaw_mode == 'parallel' else (math.pi / 2.0)


    yaw_quat = _quat_from_euler(0.0, 0.0, yaw + yaw_offset)


    target_quat = _quat_mul(yaw_quat, flip_quat)

    return normalize_quaternion(target_quat)






def sawyer_move_to(
    env, task,
    target_pos,
    timeout_s=60.0,
    trials=300,
    max_configs=10
):
    """
    그리퍼를 목표 위치로 이동 (현재 orientation 유지)

    Path planning (linear 우선, 실패 시 nonlinear)을 사용하여
    7 DOF redundancy 문제를 안정적으로 해결합니다.

    Args:
        env: RLBench environment
        task: RLBench task
        target_pos: 목표 위치 [x, y, z]
        timeout_s: 타임아웃 (초)
        trials: IK sampling 시도 횟수
        max_configs: 최대 configuration 수

    Returns:
        obs, reward, done
    """
    _move_to_log("========== [sawyer_move_to] START ==========")

    arm = _get_arm(task)
    target_pos = np.asarray(target_pos, dtype=float)


    current_quat = arm.get_tip().get_quaternion()
    current_pos = arm.get_tip().get_position()
    current_joint_pos = arm.get_joint_positions()

    _move_to_log(f"[sawyer_move_to] current_pos: {current_pos}")
    _move_to_log(f"[sawyer_move_to] target_pos: {target_pos}")
    _move_to_log(f"[sawyer_move_to] distance: {np.linalg.norm(target_pos - current_pos):.4f}m")
    _move_to_log(f"[sawyer_move_to] current_quat: {current_quat}")
    _move_to_log(f"[sawyer_move_to] current_joint_pos: {current_joint_pos}")


    try:
        ik_joint_pos = arm.solve_ik_via_jacobian(target_pos, quaternion=current_quat)
        _move_to_log(f"[sawyer_move_to] IK via Jacobian succeeded: {ik_joint_pos}")
    except IKError as e:
        _move_to_log(f"[sawyer_move_to] IK via Jacobian failed: {e}")

        try:
            ik_joint_pos = arm.solve_ik_via_jacobian(target_pos)
            _move_to_log(f"[sawyer_move_to] IK via Jacobian (no quat) succeeded: {ik_joint_pos}")
        except IKError as e2:
            _move_to_log(f"[sawyer_move_to] IK via Jacobian (no quat) also failed: {e2}")


    path = None


    try:
        _move_to_log("[sawyer_move_to] Trying linear path...")
        path = arm.get_linear_path(
            target_pos,
            quaternion=current_quat,
            steps=50,
            ignore_collisions=False
        )
        _move_to_log("[sawyer_move_to] Linear path found.")
    except (ConfigurationPathError, RuntimeError) as e:
        _move_to_log(f"[sawyer_move_to] Linear path failed: {e}")


    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying nonlinear path...")
            path = arm.get_nonlinear_path(
                target_pos,
                quaternion=current_quat,
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
            _move_to_log("[sawyer_move_to] Nonlinear path found.")
        except (ConfigurationPathError, RuntimeError) as e:
            _move_to_log(f"[sawyer_move_to] Nonlinear path failed: {e}")


    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying linear path (ignore collisions)...")
            path = arm.get_linear_path(
                target_pos,
                quaternion=current_quat,
                steps=50,
                ignore_collisions=True
            )
            _move_to_log("[sawyer_move_to] Linear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            _move_to_log(f"[sawyer_move_to] Linear path (ignore collisions) failed: {e}")


    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying nonlinear path (ignore collisions, more trials)...")
            path = arm.get_nonlinear_path(
                target_pos,
                quaternion=current_quat,
                ignore_collisions=True,
                trials=trials * 3,
                max_configs=max_configs * 2,
                algorithm=Algos.RRTConnect
            )
            _move_to_log("[sawyer_move_to] Nonlinear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            _move_to_log(f"[sawyer_move_to] Nonlinear path (ignore collisions) failed: {e}")


    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying linear path with different orientation...")

            down_quat = [0.0, 0.707, 0.0, 0.707]
            path = arm.get_linear_path(
                target_pos,
                quaternion=down_quat,
                ignore_collisions=True,
                steps=50
            )
            _move_to_log("[sawyer_move_to] Linear path (different orientation) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            _move_to_log(f"[sawyer_move_to] Linear path (different orientation) failed: {e}")


    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying nonlinear path with different orientation...")
            down_quat = [0.0, 0.707, 0.0, 0.707]
            path = arm.get_nonlinear_path(
                target_pos,
                quaternion=down_quat,
                ignore_collisions=True,
                trials=trials * 5,
                max_configs=max_configs * 3,
                algorithm=Algos.RRTConnect
            )
            _move_to_log("[sawyer_move_to] Nonlinear path (different orientation) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            _move_to_log(f"[sawyer_move_to] Nonlinear path (no quaternion) failed: {e}")



    if path is None:
        try:
            _move_to_log("[sawyer_move_to] Trying multi-waypoint path planning approach...")
            scene = _get_scene(env)


            start_pos = np.array(current_pos)
            end_pos = np.array(target_pos)
            distance = np.linalg.norm(end_pos - start_pos)
            _move_to_log(f"[sawyer_move_to] Distance to target: {distance:.3f}")






            if distance > 0.05:

                n_waypoints = max(2, int(np.ceil(distance / 0.10)))
                _move_to_log(f"[sawyer_move_to] Creating {n_waypoints} waypoints (segments ~{distance/n_waypoints:.2f}m each)...")

                success = True

                for i in range(1, n_waypoints + 1):
                    alpha = i / n_waypoints
                    waypoint_pos = start_pos + alpha * (end_pos - start_pos)

                    _move_to_log(f"[sawyer_move_to] Segment {i}/{n_waypoints}: current={arm.get_tip().get_position()} -> waypoint={waypoint_pos}")


                    segment_path = None
                    current_quat = arm.get_tip().get_quaternion()


                    try:
                        segment_path = arm.get_linear_path(
                            waypoint_pos.tolist(),
                            quaternion=current_quat,
                            ignore_collisions=False
                        )
                        _move_to_log(f"[sawyer_move_to]   Segment {i}: Linear path found ({len(segment_path)} configs)")
                    except (ConfigurationPathError, RuntimeError) as e:
                        _move_to_log(f"[sawyer_move_to]   Segment {i}: Linear path failed: {e}")


                    if segment_path is None:
                        try:
                            segment_path = arm.get_linear_path(
                                waypoint_pos.tolist(),
                                quaternion=current_quat,
                                ignore_collisions=True
                            )
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Linear path (ignore collisions) found ({len(segment_path)} configs)")
                        except (ConfigurationPathError, RuntimeError) as e:
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Linear path (ignore collisions) failed: {e}")


                    if segment_path is None:
                        try:
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Trying nonlinear path...")
                            segment_path = arm.get_path(
                                waypoint_pos.tolist(),
                                quaternion=current_quat,
                                ignore_collisions=True,
                                trials=300,
                                max_configs=60,
                                max_time_ms=10000
                            )
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Nonlinear path found ({len(segment_path)} configs)")
                        except (ConfigurationPathError, RuntimeError) as e:
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Nonlinear path failed: {e}")


                    if segment_path is None:
                        try:
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Trying nonlinear (no quaternion)...")
                            segment_path = arm.get_path(
                                waypoint_pos.tolist(),
                                ignore_collisions=True,
                                trials=300,
                                max_configs=60,
                                max_time_ms=10000
                            )
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Nonlinear (no quaternion) found ({len(segment_path)} configs)")
                        except (ConfigurationPathError, RuntimeError) as e:
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Nonlinear (no quaternion) failed: {e}")


                    if segment_path is None:
                        _move_to_log(f"[sawyer_move_to]   Segment {i}: Trying micro-segment approach...")
                        micro_target = waypoint_pos
                        micro_completed = 0
                        max_micro_attempts = 50

                        for _ in range(max_micro_attempts):
                            current_pos_now = np.array(arm.get_tip().get_position())
                            remaining_dist = np.linalg.norm(micro_target - current_pos_now)

                            if remaining_dist < 0.01:
                                _move_to_log(f"[sawyer_move_to]     Micro: Close enough to target ({remaining_dist:.3f}m)")
                                micro_completed += 1
                                break


                            step_sizes = [0.02, 0.01, 0.005]
                            micro_path = None
                            used_step = None

                            for step_size in step_sizes:
                                if step_size > remaining_dist:
                                    continue

                                direction = (micro_target - current_pos_now) / remaining_dist
                                micro_waypoint = current_pos_now + direction * step_size
                                micro_quat = arm.get_tip().get_quaternion()


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


                            if micro_path is None and remaining_dist > 0.02:
                                z_diff = micro_target[2] - current_pos_now[2]
                                if abs(z_diff) > 0.005:

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
                                        _move_to_log(f"[sawyer_move_to]     Micro: Using z-only step")
                                    except (ConfigurationPathError, RuntimeError):
                                        pass


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


                            if micro_path is None:
                                _move_to_log(f"[sawyer_move_to]     Micro: Trying direct joint interpolation...")
                                for step_size in [0.01, 0.005, 0.002]:
                                    if step_size > remaining_dist:
                                        continue
                                    direction = (micro_target - current_pos_now) / remaining_dist
                                    micro_waypoint = current_pos_now + direction * step_size


                                    try:
                                        target_joints = arm.solve_ik_via_jacobian(
                                            micro_waypoint.tolist(),
                                            quaternion=micro_quat
                                        )

                                        current_joints = np.array(arm.get_joint_positions())
                                        target_joints = np.array(target_joints)


                                        n_interp = 10
                                        for interp_i in range(1, n_interp + 1):
                                            alpha = interp_i / n_interp
                                            interp_joints = current_joints + alpha * (target_joints - current_joints)
                                            arm.set_joint_positions(interp_joints.tolist())
                                            scene.step()

                                        used_step = step_size
                                        micro_path = "direct"
                                        _move_to_log(f"[sawyer_move_to]     Micro: Direct joint interpolation succeeded for {step_size}m")
                                        break
                                    except IKError:
                                        pass

                            if micro_path is not None:
                                if micro_path != "direct":

                                    micro_path.visualize()
                                    done_micro = False
                                    while not done_micro:
                                        done_micro = micro_path.step()
                                        scene.step()
                                        if _check_success(task):
                                            obs = task.get_observation()
                                            _move_to_log(f"[sawyer_move_to]   Task succeeded during micro-segment!")
                                            return obs, 1.0, True
                                else:

                                    if _check_success(task):
                                        obs = task.get_observation()
                                        _move_to_log(f"[sawyer_move_to]   Task succeeded during direct interpolation!")
                                        return obs, 1.0, True
                                micro_completed += 1
                                new_dist = np.linalg.norm(micro_target - np.array(arm.get_tip().get_position()))
                                _move_to_log(f"[sawyer_move_to]     Micro {micro_completed}: OK step={used_step:.3f}m (remaining: {new_dist:.3f}m)")
                            else:

                                _move_to_log(f"[sawyer_move_to]     Micro: All step sizes failed at remaining dist {remaining_dist:.3f}m")
                                _move_to_log(f"[sawyer_move_to]     DEBUG: current_pos={current_pos_now}")
                                _move_to_log(f"[sawyer_move_to]     DEBUG: micro_target={micro_target}")
                                _move_to_log(f"[sawyer_move_to]     DEBUG: final_target={end_pos}")
                                joint_positions = arm.get_joint_positions()
                                _move_to_log(f"[sawyer_move_to]     DEBUG: current_joints={joint_positions}")
                                _move_to_log(f"[sawyer_move_to]     DEBUG: joint_limits:")
                                for j_idx, joint in enumerate(arm.joints):
                                    interval = joint.get_joint_interval()
                                    pos = joint.get_joint_position()
                                    at_limit = ""
                                    if interval[0] and (pos <= interval[1][0] + 0.01 or pos >= interval[1][1] - 0.01):
                                        at_limit = " <-- NEAR LIMIT!"
                                    _move_to_log(f"[sawyer_move_to]       Joint {j_idx}: pos={pos:.4f}, limits={interval}{at_limit}")

                                break


                        final_pos = np.array(arm.get_tip().get_position())
                        final_dist_to_target = np.linalg.norm(end_pos - final_pos)
                        _move_to_log(f"[sawyer_move_to]   Segment {i}: Completed {micro_completed} micro-steps, dist to final target: {final_dist_to_target:.3f}m")

                        if micro_completed > 0:
                            segment_path = True
                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Partial progress made, continuing...")

                    if segment_path is not None:
                        if segment_path is not True:

                            segment_path.visualize()
                            done_moving = False
                            while not done_moving:
                                done_moving = segment_path.step()
                                scene.step()


                                if _check_success(task):
                                    obs = task.get_observation()
                                    _move_to_log(f"[sawyer_move_to]   Task succeeded during segment {i}!")
                                    return obs, 1.0, True


                            _move_to_log(f"[sawyer_move_to]   Segment {i}: Executed successfully")


                        if _check_success(task):
                            obs = task.get_observation()
                            _move_to_log(f"[sawyer_move_to]   Task succeeded after segment {i}!")
                            return obs, 1.0, True
                    else:
                        _move_to_log(f"[sawyer_move_to]   Segment {i}: Failed - could not find path")
                        success = False
                        break

                if success:
                    _move_to_log("[sawyer_move_to] Multi-waypoint path planning succeeded!")
                    obs = task.get_observation()
                    return obs, 0.0, False
                else:

                    final_pos = np.array(arm.get_tip().get_position())
                    final_dist = np.linalg.norm(end_pos - final_pos)
                    initial_dist = np.linalg.norm(end_pos - start_pos)
                    progress = 1.0 - (final_dist / initial_dist) if initial_dist > 0 else 0.0

                    _move_to_log(f"[sawyer_move_to] Multi-waypoint partial: moved {progress*100:.1f}% of distance")
                    _move_to_log(f"[sawyer_move_to]   Initial dist: {initial_dist:.3f}m, Final dist: {final_dist:.3f}m")


                    if progress > 0.5 or final_dist < 0.10:
                        _move_to_log(f"[sawyer_move_to] Accepting partial progress - close enough to target")
                        obs = task.get_observation()
                        return obs, 0.0, False
                    else:
                        _move_to_log("[sawyer_move_to] Multi-waypoint path planning failed (insufficient progress)")
            else:

                _move_to_log(f"[sawyer_move_to] Distance very short ({distance:.4f}m), trying micro-step approach...")


                try:

                    target_joint_pos = None
                    try:
                        target_joint_pos = arm.solve_ik_via_jacobian(target_pos, quaternion=current_quat)
                        _move_to_log(f"[sawyer_move_to] Target IK found: {target_joint_pos}")
                    except IKError:

                        try:
                            target_joint_pos = arm.solve_ik_via_jacobian(target_pos)
                            _move_to_log(f"[sawyer_move_to] Target IK (no quat) found: {target_joint_pos}")
                        except IKError as e:
                            _move_to_log(f"[sawyer_move_to] Could not solve IK for target: {e}")

                    if target_joint_pos is not None:

                        current_joints = np.array(arm.get_joint_positions())
                        target_joints = np.array(target_joint_pos)
                        n_steps = 20

                        for step in range(1, n_steps + 1):
                            alpha = step / n_steps
                            intermediate_joints = current_joints + alpha * (target_joints - current_joints)
                            arm.set_joint_positions(intermediate_joints.tolist())
                            scene.step()


                            if _check_success(task):
                                obs = task.get_observation()
                                _move_to_log(f"[sawyer_move_to] Task succeeded during micro-step!")
                                return obs, 1.0, True

                        _move_to_log("[sawyer_move_to] Micro-step approach succeeded!")
                        obs = task.get_observation()
                        return obs, 0.0, False
                    else:
                        _move_to_log("[sawyer_move_to] Micro-step approach failed (no IK solution)")
                except Exception as e:
                    _move_to_log(f"[sawyer_move_to] Micro-step approach error: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            _move_to_log(f"[sawyer_move_to] Multi-waypoint strategy error: {e}")
            import traceback
            traceback.print_exc()


    if path is None:
        obs = task.get_observation()
        fd_error = FeedbackWithError(
            env=env, task=task, skill_type="sawyer_move_to",
            attempted_action=target_pos, object_positions={},
            robot_pos=obs.gripper_pose,
            waypoints=None, waypoint_index=-1, step_index=-1,
            original_robot_pos=current_pos,
            error_message=f"All path planning strategies failed"
        )
        raise PathOutOfWorkspace(fd_error)


    obs, reward, done = _execute_path(env, task, path, "sawyer_move_to", timeout_s, target_pos=target_pos)

    _move_to_log(f"[sawyer_move_to] Done. reward={reward}, done={done}")
    return obs, reward, done


def sawyer_align_gripper(
    env, task,
    approach_direction='down',
    reference_quat=None,
    yaw_mode='parallel',
    timeout_s=60.0,
    trials=300,
    max_configs=10
):
    """
    그리퍼 방향을 특정 approach direction으로 정렬 (현재 position 유지)

    Args:
        env: RLBench environment
        task: RLBench task
        approach_direction: 'down', 'up', 'left', 'right', 'front', 'back'
        reference_quat: 참조 quaternion (객체의 orientation, yaw 추출용)
        yaw_mode: 'parallel' (객체와 평행) or 'perpendicular' (객체와 수직)
        timeout_s: 타임아웃 (초)
        trials: IK sampling 시도 횟수
        max_configs: 최대 configuration 수

    Returns:
        obs, reward, done
    """
    print("========== [sawyer_align_gripper] START ==========")

    arm = _get_arm(task)
    scene = _get_scene(env)


    current_pos = np.array(arm.get_tip().get_position())
    current_quat = arm.get_tip().get_quaternion()
    
    if reference_quat is None and approach_direction is "front":
        yaw_mode = 'parallel'
    if reference_quat is None and approach_direction is "down":
        yaw_mode = 'perpendicular'


    target_quat = _compute_align_quaternion(approach_direction, reference_quat, yaw_mode)

    print(f"[sawyer_align_gripper] current_pos: {current_pos}")
    print(f"[sawyer_align_gripper] current_quat: {current_quat}")
    print(f"[sawyer_align_gripper] approach_direction: {approach_direction}")
    print(f"[sawyer_align_gripper] yaw_mode: {yaw_mode}")
    print(f"[sawyer_align_gripper] target_quat: {target_quat}")

    path = None


    try:
        print("[sawyer_align_gripper] Trying linear path...")
        path = arm.get_linear_path(
            current_pos.tolist(),
            quaternion=target_quat,
            steps=50,
            ignore_collisions=False
        )
        print("[sawyer_align_gripper] Linear path found.")
    except (ConfigurationPathError, RuntimeError) as e:
        print(f"[sawyer_align_gripper] Linear path failed: {e}")


    if path is None:
        try:
            print("[sawyer_align_gripper] Trying linear path (ignore collisions)...")
            path = arm.get_linear_path(
                current_pos.tolist(),
                quaternion=target_quat,
                steps=50,
                ignore_collisions=True
            )
            print("[sawyer_align_gripper] Linear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[sawyer_align_gripper] Linear path (ignore collisions) failed: {e}")


    if path is None:
        try:
            print("[sawyer_align_gripper] Trying nonlinear path...")
            path = arm.get_nonlinear_path(
                current_pos.tolist(),
                quaternion=target_quat,
                ignore_collisions=False,
                trials=trials,
                max_configs=max_configs,
                algorithm=Algos.RRTConnect
            )
            print("[sawyer_align_gripper] Nonlinear path found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[sawyer_align_gripper] Nonlinear path failed: {e}")


    if path is None:
        try:
            print("[sawyer_align_gripper] Trying nonlinear path (ignore collisions)...")
            path = arm.get_nonlinear_path(
                current_pos.tolist(),
                quaternion=target_quat,
                ignore_collisions=True,
                trials=trials * 2,
                max_configs=max_configs * 2,
                algorithm=Algos.RRTConnect
            )
            print("[sawyer_align_gripper] Nonlinear path (ignore collisions) found.")
        except (ConfigurationPathError, RuntimeError) as e:
            print(f"[sawyer_align_gripper] Nonlinear path (ignore collisions) failed: {e}")


    if path is None:
        print("[sawyer_align_gripper] Trying direct joint interpolation...")
        try:

            target_joints = arm.solve_ik_via_jacobian(
                current_pos.tolist(),
                quaternion=target_quat
            )
            print(f"[sawyer_align_gripper] IK solution found: {target_joints}")


            current_joints = np.array(arm.get_joint_positions())
            target_joints = np.array(target_joints)
            n_interp = 20

            for interp_i in range(1, n_interp + 1):
                alpha = interp_i / n_interp
                interp_joints = current_joints + alpha * (target_joints - current_joints)
                arm.set_joint_positions(interp_joints.tolist())
                scene.step()


                if _check_success(task):
                    obs = task.get_observation()
                    print(f"[sawyer_align_gripper] Task succeeded during interpolation!")
                    return obs, 1.0, True

            print("[sawyer_align_gripper] Direct joint interpolation succeeded.")
            obs = task.get_observation()
            success = _check_success(task)
            reward = 1.0 if success else 0.0
            print(f"[sawyer_align_gripper] Done. reward={reward}, done={success}")
            return obs, reward, success

        except IKError as e:
            print(f"[sawyer_align_gripper] IK failed: {e}")


    if path is None:
        print("[sawyer_align_gripper] Trying incremental quaternion interpolation...")
        from scipy.spatial.transform import Slerp

        try:

            current_quat_arr = np.array(current_quat)
            target_quat_arr = np.array(target_quat)


            rotations = R.from_quat([current_quat_arr, target_quat_arr])
            slerp = Slerp([0, 1], rotations)

            n_steps = 20
            success_steps = 0

            for step_i in range(1, n_steps + 1):
                alpha = step_i / n_steps
                interp_quat = slerp(alpha).as_quat().tolist()


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
                        print(f"[sawyer_align_gripper]   Step {step_i}/{n_steps}: OK (direct IK)")
                    except IKError:
                        print(f"[sawyer_align_gripper]   Step {step_i}/{n_steps}: Failed")
                        break
                else:

                    micro_path.visualize()
                    done_micro = False
                    while not done_micro:
                        done_micro = micro_path.step()
                        scene.step()
                    success_steps += 1
                    print(f"[sawyer_align_gripper]   Step {step_i}/{n_steps}: OK (path)")


                if _check_success(task):
                    obs = task.get_observation()
                    print(f"[sawyer_align_gripper] Task succeeded during step {step_i}!")
                    return obs, 1.0, True

            if success_steps > 0:
                print(f"[sawyer_align_gripper] Incremental interpolation: {success_steps}/{n_steps} steps succeeded")
                obs = task.get_observation()
                success = _check_success(task)
                reward = 1.0 if success else 0.0
                print(f"[sawyer_align_gripper] Done. reward={reward}, done={success}")
                return obs, reward, success

        except Exception as e:
            print(f"[sawyer_align_gripper] Incremental interpolation error: {e}")


    if path is None:
        obs = task.get_observation()
        fd_error = FeedbackWithError(
            env=env, task=task, skill_type="sawyer_align_gripper",
            attempted_action=None, object_positions={},
            robot_pos=obs.gripper_pose,
            waypoints=None, waypoint_index=-1, step_index=-1,
            original_robot_pos=None,
            error_message="All align path planning strategies failed"
        )
        raise PathOutOfWorkspace(fd_error)


    obs, reward, done = _execute_path(env, task, path, "sawyer_align_gripper", timeout_s)

    print(f"[sawyer_align_gripper] Done. reward={reward}, done={done}")
    return obs, reward, done


def sawyer_open_gripper(
    env, task,
    amount=1.0,
    velocity=0.2
):
    """
    그리퍼 열기

    팔 움직임 없이 그리퍼만 조작합니다.

    Args:
        env: RLBench environment
        task: RLBench task
        amount: 열림 정도 (0.0=닫힘, 1.0=완전히 열림)
        velocity: 그리퍼 동작 속도

    Returns:
        obs, reward, done
    """
    print("========== [sawyer_open_gripper] START ==========")
    print(f"[sawyer_open_gripper] amount: {amount}, velocity: {velocity}")

    gripper = _get_gripper(task)
    scene = _get_scene(env)


    amount = float(np.clip(amount, 0.0, 1.0))


    done_gripper = False
    max_steps = 100
    step = 0

    while not done_gripper and step < max_steps:
        done_gripper = gripper.actuate(amount, velocity=velocity)
        scene.step()
        step += 1


        if _check_success(task):
            obs = task.get_observation()
            print(f"[sawyer_open_gripper] Task success during gripper open.")
            return obs, 1.0, True

    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    gripper_state = getattr(obs, 'gripper_open', None)
    print(f"[sawyer_open_gripper] Done. gripper_open={gripper_state}, steps={step}")

    return obs, reward, success


def sawyer_close_gripper(
    env, task,
    amount=0.0,
    velocity=0.2
):
    """
    그리퍼 닫기

    팔 움직임 없이 그리퍼만 조작합니다.
    물체를 잡으면 조기 종료될 수 있습니다.

    Args:
        env: RLBench environment
        task: RLBench task
        amount: 닫힘 정도 (0.0=완전히 닫힘, 1.0=열림)
        velocity: 그리퍼 동작 속도

    Returns:
        obs, reward, done
    """
    print("========== [sawyer_close_gripper] START ==========")
    print(f"[sawyer_close_gripper] amount: {amount}, velocity: {velocity}")
    velocity = 0.2

    gripper = _get_gripper(task)
    scene = _get_scene(env)


    amount = float(np.clip(amount, 0.0, 1.0))


    done_gripper = False
    max_steps = 300
    step = 0
    
    while not done_gripper and step < max_steps:
        done_gripper = gripper.actuate(amount, velocity=velocity)
        print(done_gripper)
        scene.step()
        step += 1


        if _check_success(task):
            obs = task.get_observation()
            print(f"[sawyer_close_gripper] Task success during gripper close.")
            return obs, 1.0, True
    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    gripper_state = getattr(obs, 'gripper_open', None)
    print(f"[sawyer_close_gripper] Done. gripper_open={gripper_state}, steps={step}")

    return obs, reward, success






def sawyer_pick(
    env, task,
    target_object,
    target_pos=None,
    grasp_offset=np.array([0.0, 0.0, 0.0]),
    velocity=0.1
):
    """
    물체를 집는 composite skill

    1. target_pos로 이동 (또는 target_object의 위치)
    2. 그리퍼 닫기
    3. 명시적으로 물체를 gripper에 attach

    Args:
        env: RLBench environment
        task: RLBench task
        target_object: 집을 물체 (PyRep Shape object)
        target_pos: 목표 위치 (None이면 target_object.get_position() 사용)
        grasp_offset: 물체 위치에서의 offset [x, y, z]
        velocity: 그리퍼 닫는 속도

    Returns:
        obs, reward, done
    """
    print("========== [sawyer_pick] START ==========")
    

    if target_pos is None:
        target_pos = np.array(target_object.get_position(), dtype=float)
    else:
        target_pos = np.asarray(target_pos, dtype=float)
    


    grasp_offset = np.array([0.0, 0.0, 0.0])
    velocity = 0.2
    
    if grasp_offset.shape == (3,):
        grasp_pos = target_pos + grasp_offset
    else:
        grasp_pos = target_pos

    print(f"[sawyer_pick] target_pos: {target_pos}")
    print(f"[sawyer_pick] grasp_offset: {grasp_offset}")
    print(f"[sawyer_pick] grasp_pos: {grasp_pos}")


    print(f"[sawyer_pick] Moving to grasp position...")
    obs, reward, done = sawyer_move_to(env, task, grasp_pos)
    if done:
        return obs, reward, done


    print(f"[sawyer_pick] Closing gripper...")
    obs, reward, done = sawyer_close_gripper(env, task, amount=0.0, velocity=velocity)
    if done:
        return obs, reward, done


    scene = _get_scene(env)
    for _ in range(10):
        scene.step()


    print(f"[sawyer_pick] Attempting to grasp object...")
    gripper = _get_gripper(task)
    grasped = gripper.grasp(target_object)

    if grasped:
        print(f"[sawyer_pick] Object successfully grasped and attached to gripper!")
    else:
        print(f"[sawyer_pick] WARNING: Failed to grasp object - not detected by proximity sensor")


    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    print(f"[sawyer_pick] Done. grasped={grasped}, reward={reward}, done={success}")
    return obs, reward, success


def sawyer_place(
    env, task,
    place_pos,
    place_offset=np.array([0.0, 0.0, 0.0])
):
    """
    물체를 놓는 composite skill

    1. place_pos로 이동
    2. 명시적으로 물체를 gripper에서 detach
    3. 그리퍼 열기

    Args:
        env: RLBench environment
        task: RLBench task
        place_pos: 놓을 위치 [x, y, z]
        place_offset: place_pos에서의 offset [x, y, z]

    Returns:
        obs, reward, done
    """
    print("========== [sawyer_place] START ==========")

    place_pos = np.asarray(place_pos, dtype=float)
    place_offset = np.asarray(place_offset, dtype=float)

    if place_offset.shape == (3,):
        final_pos = place_pos + place_offset
    else:
        final_pos = place_pos

    print(f"[sawyer_place] place_pos: {place_pos}")
    print(f"[sawyer_place] place_offset: {place_offset}")
    print(f"[sawyer_place] final_pos: {final_pos}")


    print(f"[sawyer_place] Moving to place position...")
    obs, reward, done = sawyer_move_to(env, task, final_pos)
    if done:
        return obs, reward, done


    print(f"[sawyer_place] Releasing object from gripper...")
    gripper = _get_gripper(task)
    gripper.release()


    scene = _get_scene(env)
    for _ in range(10):
        scene.step()


    print(f"[sawyer_place] Opening gripper...")
    obs, reward, done = sawyer_open_gripper(env, task)
    if done:
        return obs, reward, done


    obs = task.get_observation()
    success = _check_success(task)
    reward = 1.0 if success else 0.0

    print(f"[sawyer_place] Done. reward={reward}, done={success}")
    return obs, reward, success
