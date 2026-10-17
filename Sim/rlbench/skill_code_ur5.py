import numpy as np
import time
import math
import imageio
import signal
from scipy.spatial.transform import Rotation as R

from rlbench.backend.observation import Observation
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import ConfigurationError, ConfigurationPathError, IKError, PyRepError

from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.feedback import FeedbackWithError


class StepTimeoutError(Exception):
    """Raised when task.step() exceeds the allowed timeout."""
    pass


def _timeout_handler(signum, frame):
    raise StepTimeoutError("task.step() timed out")


# =========================================================
# Utils (KEEP) — safe_step / camera / quaternion helpers
# =========================================================

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


def safe_step(env, task, action, skill_type,
              step_index=-1, waypoint_index=-1, waypoints=None, threshold=None, original_pos=None,
              step_timeout_s: int = 30):
    """
    Call task.step(action) and handle exceptions -> raise SkillFailure / PathOutOfWorkspace with Feedback.

    Args:
        step_timeout_s: Timeout in seconds for a single task.step() call (default 30s).
                        Set to 0 or None to disable timeout.
    """
    # Set up signal-based timeout for task.step()
    use_timeout = step_timeout_s and step_timeout_s > 0
    if use_timeout:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(step_timeout_s))

    try:
        obs, reward, done = task.step(action)
    except StepTimeoutError:
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
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
            error_message=f"Single step timed out after {step_timeout_s}s"
        )
        raise SkillFailure(fd_error)
    except ValueError as e:
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
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
        if use_timeout:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
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

    # Cancel the alarm on success
    if use_timeout:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return obs, reward, done


# =========================================================
# Quaternion math for UR5 alignment (robust + compatible intent)
# =========================================================

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


def _yaw_from_quat(q):
    x, y, z, w = q
    t0 = 2.0*(w*z + x*y)
    t1 = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(t0, t1)


_flip_dict = {
    'down':  np.array([1.0, 0.0, 0.0, 0.0], dtype=float),  # 180deg about X
    'right': _quat_from_euler(0.0, math.radians(90), 0.0),
    'left':  _quat_from_euler(0.0, math.radians(-90), 0.0),
    'front': _quat_from_euler(math.radians(-90), 0.0, 0.0),
    'back':  _quat_from_euler(math.radians(90), 0.0, 0.0),
}

_axis_flip_fallback = {
    'z':  _flip_dict['down'],
    '-z': _quat_from_euler(0.0, 0.0, 0.0),  # identity
    'x':  _flip_dict['right'],
    '-x': _flip_dict['left'],
    'y':  _flip_dict['front'],
    '-y': _flip_dict['back'],
}

# Target z-axis directions for each approach direction
_approach_target_z = {
    'down':  np.array([0.0, 0.0, -1.0]),   # gripper z points down
    'z':     np.array([0.0, 0.0, -1.0]),
    '-z':    np.array([0.0, 0.0, 1.0]),    # gripper z points up
    'right': np.array([1.0, 0.0, 0.0]),
    'x':     np.array([1.0, 0.0, 0.0]),
    'left':  np.array([-1.0, 0.0, 0.0]),
    '-x':    np.array([-1.0, 0.0, 0.0]),
    'front': np.array([0.0, 1.0, 0.0]),
    'y':     np.array([0.0, 1.0, 0.0]),
    'back':  np.array([0.0, -1.0, 0.0]),
    '-y':    np.array([0.0, -1.0, 0.0]),
}


def _get_gripper_z_axis(quat: np.ndarray) -> np.ndarray:
    """Get the gripper's z-axis direction in world frame from quaternion (xyzw)."""
    rot = R.from_quat(quat)  # scipy uses xyzw
    z_local = np.array([0.0, 0.0, 1.0])
    return rot.apply(z_local)


def _needs_flip(current_quat: np.ndarray, approach_direction: str, threshold: float = 0.5) -> bool:
    """
    Check if gripper needs to flip to achieve the desired approach direction.
    Returns True if current gripper z-axis is NOT aligned with target direction.
    threshold: dot product threshold (0.5 = 60 degrees, 0.0 = 90 degrees)
    """
    if approach_direction not in _approach_target_z:
        return True  # unknown direction, apply flip by default

    target_z = _approach_target_z[approach_direction]
    current_z = _get_gripper_z_axis(current_quat)

    dot = np.dot(current_z, target_z)
    # If dot > threshold, gripper is already pointing in roughly the right direction
    return dot < threshold


def _get_flip_quat(approach_direction: str) -> np.ndarray:
    if approach_direction in _flip_dict:
        return _flip_dict[approach_direction].copy()
    if approach_direction in _axis_flip_fallback:
        return _axis_flip_fallback[approach_direction].copy()
    raise ValueError(f"Unknown approach_direction: {approach_direction}")


def _get_gripper_state(obs, gripper: float):
    """gripper: 1.0 open, -1.0 close, None -> keep current"""
    # if gripper is None:
    return float(getattr(obs, 'gripper_open', 1.0))
    # return float(gripper)


# =========================================================
# UR5 Skills (NO WRAPPERS)
# =========================================================

def ur5_move_to(
    env, task,
    target_pos: np.ndarray,
    target_quat: np.ndarray = None,   # None -> keep current orientation
    gripper: float = None,            # None -> keep current gripper state
    n_waypoints: int = 10,
    pos_tol: float = 0.01,
    max_steps_per_wp: int = 100,
    timeout_s: float = 30.0,
    log_cam: str = None,
    log_prefix: str = "move"
):
    """
    Straight-line waypoint interpolation to target_pos.
    Keeps current orientation unless target_quat is provided.
    Keeps current gripper state unless gripper is provided.
    """
    print("========== [ur5_move_to] START ==========")
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()

    use_quat = start_quat if target_quat is None else normalize_quaternion(target_quat)
    grip = _get_gripper_state(obs, gripper)

    print(f"[ur5_move_to] start_pos: {start_pos}")
    print(f"[ur5_move_to] start_quat(xyzw): {start_quat}, euler={euler_from_quat(start_quat)}")
    print(f"[ur5_move_to] target_pos: {target_pos}")
    print(f"[ur5_move_to] target_quat: {use_quat}")

    n_waypoints = max(2, int(n_waypoints))
    waypoints = []
    for t in range(n_waypoints):
        a = t / (n_waypoints - 1)
        waypoints.append(start_pos + a * (target_pos - start_pos))

    action = np.zeros(env.action_shape, dtype=float)
    action[-1] = grip
    
    print(f"[ur5_move_to] Target position: {target_pos}, Target quaternion: {use_quat}")
    print(f"[ur5_move_to] Moving through {n_waypoints} waypoints...: \n{waypoints}")

    for i, wp in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps_per_wp:
            # Check timeout BEFORE step execution
            if time.time() - start_time > timeout_s:
                fd_error = FeedbackWithError(
                    env=env, task=task, skill_type="ur5_move_to",
                    attempted_action=action, object_positions={},
                    robot_pos=task.get_observation().gripper_pose,
                    waypoints=waypoints, waypoint_index=i, step_index=step_count,
                    original_robot_pos=original_pos,
                    error_message=f"Timeout: Failed to reach waypoint {i} within {timeout_s} seconds."
                )
                raise SkillFailure(fd_error)

            obs = task.get_observation()
            cur_pos = obs.gripper_pose[:3]
            if np.linalg.norm(wp - cur_pos) < pos_tol:
                print(f"      -> Reached waypoint {i}.")
                break

            action[:3] = wp
            action[3:7] = use_quat

            obs, reward, done = safe_step(
                env, task, action,
                skill_type="ur5_move_to",
                waypoint_index=i,
                step_index=step_count,
                waypoints=waypoints,
                original_pos=original_pos,
                threshold=pos_tol
            )
            step_count += 1

            if done:
                print(f"[ur5_move_to] Task done detected at waypoint {i}.")
                return obs, reward, done

        else:
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="ur5_move_to",
                attempted_action=action, object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=waypoints, waypoint_index=i, step_index=step_count,
                original_robot_pos=original_pos,
                error_message=f"Exceeded max_steps_per_wp({max_steps_per_wp}) at waypoint {i}."
            )
            raise SkillFailure(fd_error)

        if log_cam is not None:
            save_camera_image(obs, log_cam, f'/workspace/rlbench_server/logs/{log_prefix}_step{i}_{log_cam}.png')

    for _ in range(5):
        action[:3] = target_pos
        action[3:7] = use_quat
        obs, reward, done = safe_step(
            env, task, action,
            skill_type="ur5_move_to",
            waypoint_index=-1,
            step_index=-1,
            waypoints=waypoints,
            original_pos=original_pos,
            threshold=pos_tol
        )
        if done:
            return obs, reward, done

    final_gripper_state = float(getattr(obs, 'gripper_open', -999))
    print(f"[ur5_move_to] Done. Final gripper_open: {final_gripper_state}")
    return obs, reward, done


def ur5_grasp_at(
    env, task,
    grasp_pos: np.ndarray,
    grasp_quat: np.ndarray = None,     # None -> keep current orientation
    approach: dict = None,             # {"axis":..., "distance":..., "n_waypoints_approach":..., "n_waypoints_descend":...}
    gripper_close: float = 0.0,
    settle_steps: int = 5,
    close_steps: int = 10,
    pos_tol: float = 0.01,
    max_steps_per_wp: int = 100,
    timeout_s: float = 30.0,
    log_cam: str = "cam_front",
    log_prefix: str = "grasp"
):
    """
    Approach -> descend -> settle -> close.
    Defaults preserve your original pick behavior.
    """
    print("========== [ur5_grasp_at] START ==========")
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()

    use_quat = start_quat if grasp_quat is None else normalize_quaternion(grasp_quat)

    if approach is None:
        approach = dict(axis='z', distance=0.15, n_waypoints_approach=5, n_waypoints_descend=10)

    axis = approach.get('axis', 'z')
    dist = float(approach.get('distance', 0.15))
    n1 = int(approach.get('n_waypoints_approach', 5))
    n2 = int(approach.get('n_waypoints_descend', 10))

    approach_dir_map = {
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
    }
    if axis not in approach_dir_map:
        raise ValueError(f"Unknown approach axis: {axis}")

    approach_pos = grasp_pos + approach_dir_map[axis] * dist

    print(f"[ur5_grasp_at] start_pos: {start_pos}")
    print(f"[ur5_grasp_at] grasp_pos: {grasp_pos}")
    print(f"[ur5_grasp_at] approach_pos: {approach_pos} (axis={axis}, dist={dist})")
    print(f"[ur5_grasp_at] use_quat(xyzw): {use_quat}, euler={euler_from_quat(use_quat)}")

    waypoints = []
    n1 = max(2, n1)
    n2 = max(2, n2)

    for t in range(n1):
        a = t / (n1 - 1)
        waypoints.append(start_pos + a * (approach_pos - start_pos))
    for t in range(n2):
        a = t / (n2 - 1)
        waypoints.append(approach_pos + a * (grasp_pos - approach_pos))

    action = np.zeros(env.action_shape, dtype=float)
    action[-1] = 1.0  # open while approaching

    for i, wp in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps_per_wp:
            # Check timeout BEFORE step execution
            if time.time() - start_time > timeout_s:
                fd_error = FeedbackWithError(
                    env=env, task=task, skill_type="ur5_grasp_at",
                    attempted_action=action, object_positions={},
                    robot_pos=task.get_observation().gripper_pose,
                    waypoints=waypoints, waypoint_index=i, step_index=step_count,
                    original_robot_pos=original_pos,
                    error_message=f"Timeout: Failed to reach waypoint {i} within {timeout_s} seconds."
                )
                raise SkillFailure(fd_error)

            obs = task.get_observation()
            cur_pos = obs.gripper_pose[:3]
            if np.linalg.norm(wp - cur_pos) < pos_tol:
                print(f"      -> Reached waypoint {i}.")
                break

            action[:3] = wp
            action[3:7] = use_quat

            obs, reward, done = safe_step(
                env, task, action,
                skill_type="ur5_grasp_at",
                waypoint_index=i,
                step_index=step_count,
                waypoints=waypoints,
                original_pos=original_pos,
                threshold=pos_tol
            )
            step_count += 1

            if done:
                print(f"[ur5_grasp_at] Task done detected at waypoint {i}.")
                return obs, reward, done

        else:
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="ur5_grasp_at",
                attempted_action=action, object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=waypoints, waypoint_index=i, step_index=step_count,
                original_robot_pos=original_pos,
                error_message=f"Exceeded max_steps_per_wp({max_steps_per_wp}) at waypoint {i}."
            )
            raise SkillFailure(fd_error)

        if log_cam is not None:
            save_camera_image(obs, log_cam, f'/workspace/rlbench_server/logs/{log_prefix}_step{i}_{log_cam}.png')

    print(f"[ur5_grasp_at] Starting settle phase ({settle_steps} steps)...")
    for settle_step in range(max(0, int(settle_steps))):
        action[:3] = grasp_pos
        action[3:7] = use_quat
        action[-1] = 1.0
        obs, reward, done = safe_step(
            env, task, action,
            skill_type="ur5_grasp_at",
            waypoint_index=-1,
            step_index=-1,
            waypoints=waypoints,
            original_pos=original_pos,
            threshold=pos_tol
        )
        print(f"      -> Settle step {settle_step + 1}/{settle_steps} done.")
        if done:
            print(f"[ur5_grasp_at] Task done during settle phase at step {settle_step + 1}.")
            return obs, reward, done

    print(f"[ur5_grasp_at] Settle phase complete. Starting close phase ({close_steps} steps)...")
    action[-1] = float(gripper_close)
    success_close_count = 0
    for close_step in range(max(1, int(close_steps))):
        action[:3] = grasp_pos
        action[3:7] = use_quat
        try:
            obs, reward, done = safe_step(
                env, task, action,
                skill_type="ur5_grasp_at",
                waypoint_index=-1,
                step_index=-1,
                waypoints=waypoints,
                original_pos=original_pos,
                threshold=pos_tol,
                step_timeout_s=5  # Reduce timeout to 5s for close steps
            )
            print(f"      -> Close step {close_step + 1}/{close_steps} done. gripper_open={getattr(obs, 'gripper_open', 'N/A')}")
            success_close_count += 1
            if done:
                print(f"[ur5_grasp_at] Task done during close phase at step {close_step + 1}.")
                return obs, reward, done
        except (SkillFailure, PathOutOfWorkspace) as e:
            # If gripper can't close further due to object contact, that's OK
            print(f"      -> Close step {close_step + 1}/{close_steps} failed (object blocking): {e.feedback.error_message if hasattr(e, 'feedback') else str(e)}")
            # If we had at least some successful closes, break. Otherwise continue trying
            if success_close_count >= 3:
                print(f"[ur5_grasp_at] Gripper closed {success_close_count} times, object blocking further close. Continuing...")
                break
            else:
                print(f"[ur5_grasp_at] Only {success_close_count} successful closes, continuing to try...")
                continue

    final_gripper_state = float(getattr(obs, 'gripper_open', -999))
    print(f"[ur5_grasp_at] Done. Final gripper_open: {final_gripper_state}")
    return obs, reward, done


def ur5_release_at(
    env, task,
    place_pos: np.ndarray,
    place_quat: np.ndarray = None,
    approach: dict = None,
    gripper_open: float = 1.0,
    settle_steps: int = 5,
    open_steps: int = 10,
    pos_tol: float = 0.01,
    max_steps_per_wp: int = 100,
    timeout_s: float = 30.0,
    log_cam: str = "cam_front",
    log_prefix: str = "release"
):
    """
    Approach -> descend -> settle -> open.
    Defaults preserve your original place behavior.
    """
    print("========== [ur5_release_at] START ==========")
    obs = task.get_observation()
    start_pos = obs.gripper_pose[:3]
    start_quat = normalize_quaternion(obs.gripper_pose[3:7])
    original_pos = obs.gripper_pose.copy()

    use_quat = start_quat if place_quat is None else normalize_quaternion(place_quat)

    if approach is None:
        approach = dict(axis='z', distance=0.15, n_waypoints_approach=5, n_waypoints_descend=10)

    axis = approach.get('axis', 'z')
    dist = float(approach.get('distance', 0.15))
    n1 = int(approach.get('n_waypoints_approach', 5))
    n2 = int(approach.get('n_waypoints_descend', 10))

    approach_dir_map = {
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1]),
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
    }
    if axis not in approach_dir_map:
        raise ValueError(f"Unknown approach axis: {axis}")

    # For release, approach from above (same as grasp) - move to approach_pos first, then descend to place_pos
    # E.g., if axis='z' and dist=0.1, approach_pos = place_pos + [0,0,0.1] (above the place location)
    approach_pos = place_pos + approach_dir_map[axis] * dist

    # Calculate waypoint count based on distance (max 2cm per waypoint)
    max_wp_spacing = 0.02  # 2cm
    dist_to_approach = np.linalg.norm(approach_pos - start_pos)
    dist_to_place = np.linalg.norm(place_pos - approach_pos)

    n1 = max(n1, int(np.ceil(dist_to_approach / max_wp_spacing)) + 1)
    n2 = max(n2, int(np.ceil(dist_to_place / max_wp_spacing)) + 1)

    print(f"[ur5_release_at] start_pos: {start_pos}")
    print(f"[ur5_release_at] place_pos: {place_pos}")
    print(f"[ur5_release_at] approach_pos: {approach_pos} (axis={axis}, dist={dist})")
    print(f"[ur5_release_at] dist_to_approach: {dist_to_approach:.3f}m, n1: {n1}")
    print(f"[ur5_release_at] dist_to_place: {dist_to_place:.3f}m, n2: {n2}")
    print(f"[ur5_release_at] use_quat(xyzw): {use_quat}, euler={euler_from_quat(use_quat)}")

    waypoints = []

    for t in range(n1):
        a = t / (n1 - 1)
        waypoints.append(start_pos + a * (approach_pos - start_pos))
    for t in range(n2):
        a = t / (n2 - 1)
        waypoints.append(approach_pos + a * (place_pos - approach_pos))

    action = np.zeros(env.action_shape, dtype=float)
    # Keep current gripper state while moving - use gripper_open directly as action value
    # gripper_open: 1.0 = open, 0.0 = closed. Action: 1.0 = open, -1.0 = closed
    current_gripper_state = float(getattr(obs, 'gripper_open', 1.0))
    # Convert gripper_open (0-1) to action (-1 to 1): action = 2 * gripper_open - 1
    gripper_action = 2.0 * current_gripper_state - 1.0
    action[-1] = gripper_action

    print(f"[ur5_release_at] current_gripper_state: {current_gripper_state}, gripper_action: {gripper_action}")
    print("[ur5_relase_at] waypoints:\n", waypoints)
    
    for i, wp in enumerate(waypoints):
        step_count = 0
        start_time = time.time()
        while step_count < max_steps_per_wp:
            # Check timeout BEFORE step execution
            if time.time() - start_time > timeout_s:
                fd_error = FeedbackWithError(
                    env=env, task=task, skill_type="ur5_release_at",
                    attempted_action=action, object_positions={},
                    robot_pos=task.get_observation().gripper_pose,
                    waypoints=waypoints, waypoint_index=i, step_index=step_count,
                    original_robot_pos=original_pos,
                    error_message=f"Timeout: Failed to reach waypoint {i} within {timeout_s} seconds."
                )
                raise SkillFailure(fd_error)

            obs = task.get_observation()
            cur_pos = obs.gripper_pose[:3]
            if np.linalg.norm(wp - cur_pos) < pos_tol:
                print(f"      -> Reached waypoint {i}.")
                break

            action[:3] = wp
            action[3:7] = use_quat

            obs, reward, done = safe_step(
                env, task, action,
                skill_type="ur5_release_at",
                waypoint_index=i,
                step_index=step_count,
                waypoints=waypoints,
                original_pos=original_pos,
                threshold=pos_tol
            )
            step_count += 1

            # preserve your original place behavior on done
            if done:
                print(f"[ur5_release_at] Task done detected at waypoint {i}.")
                return obs, reward, done

        else:
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="ur5_release_at",
                attempted_action=action, object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=waypoints, waypoint_index=i, step_index=step_count,
                original_robot_pos=original_pos,
                error_message=f"Exceeded max_steps_per_wp({max_steps_per_wp}) at waypoint {i}."
            )
            raise SkillFailure(fd_error)

        if log_cam is not None:
            save_camera_image(obs, log_cam, f'/workspace/rlbench_server/logs/{log_prefix}_step{i}_{log_cam}.png')

    print(f"[ur5_release_at] Starting settle phase ({settle_steps} steps)...")
    for settle_step in range(max(0, int(settle_steps))):
        action[:3] = place_pos
        action[3:7] = use_quat
        # Keep current gripper state during settle (don't force close)
        action[-1] = gripper_action
        obs, reward, done = safe_step(
            env, task, action,
            skill_type="ur5_release_at",
            waypoint_index=-1,
            step_index=-1,
            waypoints=waypoints,
            original_pos=original_pos,
            threshold=pos_tol
        )
        print(f"      -> Settle step {settle_step + 1}/{settle_steps} done.")
        if done:
            print(f"[ur5_release_at] Task done during settle phase at step {settle_step + 1}.")
            return obs, reward, done

    print(f"[ur5_release_at] Settle phase complete. Starting open phase ({open_steps} steps)...")
    action[-1] = float(gripper_open)
    for open_step in range(max(1, int(open_steps))):
        action[:3] = place_pos
        action[3:7] = use_quat
        try:
            obs, reward, done = safe_step(
                env, task, action,
                skill_type="ur5_release_at",
                waypoint_index=-1,
                step_index=-1,
                waypoints=waypoints,
                original_pos=original_pos,
                threshold=pos_tol
            )
            print(f"      -> Open step {open_step + 1}/{open_steps} done. Gripper state: {getattr(obs, 'gripper_open', 'N/A')}")
            if done:
                print(f"[ur5_release_at] Task done during open phase at step {open_step + 1}.")
                return obs, reward, done
        except (SkillFailure, PathOutOfWorkspace) as e:
            # If gripper can't open further due to object contact, that's OK - just continue
            print(f"      -> Open step {open_step + 1}/{open_steps} failed (object blocking): {e.feedback.error_message if hasattr(e, 'feedback') else str(e)}")
            print(f"[ur5_release_at] Gripper blocked by object at open step {open_step + 1}, continuing...")
            break

    final_gripper_state = getattr(obs, 'gripper_open', 1.0)
    print(f"[ur5_release_at] Final gripper state: {final_gripper_state}")
    if final_gripper_state != 1.0:
        print(f"[ur5_release_at] WARNING: Gripper did not fully open (state={final_gripper_state}), but continuing anyway...")

    print("[ur5_release_at] Done.")
    return obs, reward, done


def _get_arm(task):
    """Get robot arm from task object."""
    if hasattr(task, '_robot'):
        return task._robot.arm
    elif hasattr(task, '_task') and hasattr(task._task, 'robot'):
        return task._task.robot.arm
    return None


def _select_best_config_for_neutral(configs: np.ndarray, neutral_joints: np.ndarray) -> np.ndarray:
    """
    Select the joint configuration closest to neutral pose.

    Args:
        configs: Array of shape (N, num_joints) with candidate configurations
        neutral_joints: Target neutral joint values

    Returns:
        Best configuration (closest to neutral)
    """
    if len(configs) == 1:
        return configs[0]

    # Score each config by distance to neutral (weighted by joint importance)
    # Joints closer to base typically have more impact on overall pose
    weights = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.3])  # UR5 has 6 joints
    if len(neutral_joints) != len(weights):
        weights = np.ones(len(neutral_joints))

    best_config = configs[0]
    best_score = float('inf')

    for config in configs:
        # Angular distance with joint wrapping consideration
        diff = np.abs(config - neutral_joints)
        # Handle circular joints (wrap around 2*pi)
        diff = np.minimum(diff, 2*np.pi - diff)
        score = np.sum(weights[:len(diff)] * diff)

        if score < best_score:
            best_score = score
            best_config = config

    return best_config


# UR5 neutral/home joint configuration (in radians)
# These values represent a "ready" pose where arm is not twisted
UR5_NEUTRAL_JOINTS = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])


def ur5_align_gripper(
    env, task,
    reference_quat: np.ndarray,
    approach_direction: str = "down",          # 'down','right','left','front','back' (also 'x','-x','y','-y','z','-z')
    yaw_mode: str = "parallel",               # 'parallel' or 'perpendicular'
    yaw_offset_rad: float = 0.0,
    pos_hold: np.ndarray = None,        # None -> hold current position
    gripper: float = None,              # None -> keep current gripper
    ori_tol: float = 1e-3,
    max_steps: int = 100,
    timeout_s: float = 30.0
):
    """
    Align gripper orientation using reference_quat's yaw + mode + approach flip.
    Keeps position fixed while rotating.
    """
    print("========== [ur5_align_gripper] START ==========")

    obs = task.get_observation()
    original_pos = obs.gripper_pose.copy()
    current_quat = normalize_quaternion(obs.gripper_pose[3:7])

    pos = obs.gripper_pose[:3] if pos_hold is None else np.asarray(pos_hold, dtype=float)
    grip = _get_gripper_state(obs, gripper)

    ref_q = normalize_quaternion(reference_quat)
    yaw = _yaw_from_quat(ref_q)

    yaw_extra = 0.0 if (yaw_mode == "parallel") else (math.pi / 2.0)
    yaw_extra += float(yaw_offset_rad)

    yaw_quat = _quat_from_euler(0.0, 0.0, yaw + yaw_extra)

    # Check if flip is needed based on current gripper orientation
    needs_flip = _needs_flip(current_quat, str(approach_direction))
    print(f"[ur5_align_gripper] current gripper z-axis: {_get_gripper_z_axis(current_quat)}")
    print(f"[ur5_align_gripper] needs_flip: {needs_flip}")

    if needs_flip:
        flip_quat = _get_flip_quat(str(approach_direction))
        target_quat = normalize_quaternion(_quat_mul(yaw_quat, flip_quat))
    else:
        # No flip needed - keep current pitch/roll, just adjust yaw
        # Extract current pitch from gripper orientation
        current_euler = R.from_quat(current_quat).as_euler('xyz', degrees=False)
        current_pitch = current_euler[1]  # Y-axis rotation (pitch)

        # Build target: yaw from reference + current pitch + 180 roll (to point down)
        target_euler = np.array([math.pi, current_pitch, yaw + yaw_extra])
        target_quat = normalize_quaternion(R.from_euler('xyz', target_euler).as_quat())

    print(f"[ur5_align_gripper] pos_hold: {pos}")
    print(f"[ur5_align_gripper] yaw_mode: {yaw_mode}, yaw_offset_rad: {yaw_offset_rad}")
    print(f"[ur5_align_gripper] approach_direction: {approach_direction}")
    print(f"[ur5_align_gripper] target_quat(xyzw): {target_quat}, euler={euler_from_quat(target_quat)}")

    action = np.zeros(env.action_shape, dtype=float)
    action[-1] = grip

    start_time = time.time()
    step = 0
    while True:
        # Check timeout BEFORE step execution
        if time.time() - start_time > timeout_s:
            print(f"[ur5_align_gripper] Timeout after {timeout_s}s at step {step}.")
            return obs, 0.0, False

        obs = task.get_observation()
        curr_quat = normalize_quaternion(obs.gripper_pose[3:7])

        if angle_diff(curr_quat, target_quat) < ori_tol:
            break
        if step >= max_steps:
            break

        action[:3] = pos
        action[3:7] = target_quat

        obs, reward, done = safe_step(
            env, task, action,
            skill_type="ur5_align_gripper",
            waypoint_index=-1,
            step_index=step,
            waypoints=None,
            original_pos=original_pos,
            threshold=None
        )

        if done:
            if reward >= 1.0:
                return obs, reward, done
            fd_error = FeedbackWithError(
                env=env, task=task, skill_type="ur5_align_gripper",
                attempted_action=action, object_positions={},
                robot_pos=obs.gripper_pose,
                waypoints=None, waypoint_index=-1, step_index=step,
                original_robot_pos=original_pos,
                error_message=f"Task ended with failure at step {step}."
            )
            raise SkillFailure(fd_error)

        step += 1

    for _ in range(5):
        action[:3] = pos
        action[3:7] = target_quat
        obs, reward, done = safe_step(
            env, task, action,
            skill_type="ur5_align_gripper",
            waypoint_index=-1,
            step_index=-1,
            waypoints=None,
            original_pos=original_pos,
            threshold=None
        )
        if done:
            break

    print("[ur5_align_gripper] Done.")
    return obs, reward, done

# =========================================================
# UR5 Skills (0103 added)
# =========================================================

def close_ur5_ee(
    env, task,
    gripper_close: float = 0.0,
    velocity: float = 0.2
):
    """
    Close the gripper directly without moving the arm.

    Args:
        env: Environment instance
        task: Task instance
        gripper_close: Gripper close amount (0.0 = fully closed, 1.0 = fully open)
        velocity: Gripper actuation velocity

    Returns:
        obs, reward, done
    """
    print("========== [close_ur5_ee] START ==========")
    print(f"[close_ur5_ee] gripper_close: {gripper_close}, velocity: {velocity}")

    if gripper_close < 0.0:
        gripper_close = 0.0
    if gripper_close > 1.0:
        gripper_close = 1.0

    # Access gripper directly
    if hasattr(task, '_robot'):
        gripper = task._robot.gripper
    elif hasattr(task, '_task') and hasattr(task._task, 'robot'):
        gripper = task._task.robot.gripper
    else:
        raise RuntimeError("Cannot access robot gripper")

    # Actuate gripper until done
    done_gripper = False
    while not done_gripper:
        done_gripper = gripper.actuate(gripper_close, velocity=velocity)
        env._scene.step()

    obs = task.get_observation()
    _, reward = task._task.success()
    done = reward == 1.0

    print(f"[close_ur5_ee] Done. gripper_open={getattr(obs, 'gripper_open', 'N/A')}")
    return obs, reward, done


def open_ur5_ee(
    env, task,
    gripper_open: float = 1.0,
    velocity: float = 0.2
):
    """
    Open the gripper directly without moving the arm.

    Args:
        env: Environment instance
        task: Task instance
        gripper_open: Gripper open amount (0.0 = fully closed, 1.0 = fully open)
        velocity: Gripper actuation velocity

    Returns:
        obs, reward, done
    """
    print("========== [open_ur5_ee] START ==========")
    print(f"[open_ur5_ee] gripper_open: {gripper_open}, velocity: {velocity}")

    # Access gripper directly
    if hasattr(task, '_robot'):
        gripper = task._robot.gripper
    elif hasattr(task, '_task') and hasattr(task._task, 'robot'):
        gripper = task._task.robot.gripper
    else:
        raise RuntimeError("Cannot access robot gripper")

    # Actuate gripper until done
    done_gripper = False
    while not done_gripper:
        done_gripper = gripper.actuate(gripper_open, velocity=velocity)
        env._scene.step()

    obs = task.get_observation()
    _, reward = task._task.success()
    done = reward == 1.0

    print(f"[open_ur5_ee] Done. gripper_open={getattr(obs, 'gripper_open', 'N/A')}")
    return obs, reward, done