"""
Primitive Skill APIs for Genesis Simulator.

This module provides low-level robot manipulation primitives that can be
composed to perform complex tasks.
"""

import numpy as np
import torch

from src.common.constants import FLOOR_AABB


BASE_ANGLE_OFFSET = {
    "down": 0,
    "left": 0,
    "right": 0,
    "front": np.pi / 2,
}


def normalize_quaternion(q):
    """Normalize a quaternion."""
    return q / np.linalg.norm(q)


def direction_to_quat(direction, angle=0.0):
    """
    Convert a direction vector to a quaternion.

    Args:
        direction: Target direction vector (3D).
        angle: Rotation angle around the direction axis.

    Returns:
        Quaternion [w, x, y, z].
    """
    dir_vec = np.asarray(direction, dtype=float)
    n = np.linalg.norm(dir_vec)
    if n < 1e-8:
        raise ValueError("direction must be non-zero")
    dir_vec /= n

    z = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(z, dir_vec), -1.0, 1.0))

    eps = 1e-6
    if abs(dot - 1.0) < eps:
        base = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    elif abs(dot + 1.0) < eps:
        base = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    else:
        axis = np.cross(z, dir_vec)
        axis /= np.linalg.norm(axis)
        half = np.arccos(dot) * 0.5
        s = np.sin(half)
        base = np.array(
            [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float
        )

    if angle != 0.0:
        h = -angle * 0.5
        s = np.sin(h)
        wrist = np.array(
            [np.cos(h), dir_vec[0] * s, dir_vec[1] * s, dir_vec[2] * s], dtype=float
        )
        w1, x1, y1, z1 = wrist
        w2, x2, y2, z2 = base
        base = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    base /= np.linalg.norm(base)
    return base


def get_gripper_offset(ee_type, pointing_to="down"):
    """
    Get the gripper offset based on end-effector type and pointing direction.

    Args:
        ee_type: "gripper", "robotiq85", or "suction".
        pointing_to: "down", "left", or "right".

    Returns:
        Offset vector [x, y, z].
    """
    if ee_type == "gripper":
        pos_offset = np.array([0.0035, 0, 0.115])
    elif ee_type == "robotiq85":



        pos_offset = np.array([0.0, 0, 0.265])
    else:
        pos_offset = np.array([0.0, 0.0, 0.215])

    if pointing_to == "left":
        pos_offset = np.array([pos_offset[0], pos_offset[2], pos_offset[1]])
    elif pointing_to == "right":
        pos_offset = np.array([pos_offset[0], -pos_offset[2], pos_offset[1]])

    return pos_offset


def move_gripper_to(env, obj_name, pointing_to="down", depth=0.01):
    """
    Move the gripper to a target object.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the target object.
        pointing_to: Approach direction ("down", "left", "right").
        depth: How deep to position the gripper relative to object surface.

    Returns:
        True if successful, False otherwise.
    """
    if obj_name not in env.env.scene_objects:
        error_msg = f"Object '{obj_name}' not found in scene"
        print(error_msg)
        raise RuntimeError(error_msg)

    if not _feasibility_check(env, obj_name):
        error_msg = f"Object '{obj_name}' is not reachable."
        print(error_msg)
        raise RuntimeError(error_msg)

    if env.ee_type == "suction":
        depth = 0.0
        if "handle" in obj_name:
            depth = -0.01

    direction = {
        "down": np.array([0.0, 0.0, -1.0]),
        "left": np.array([0.0, -1.0, 0.0]),
        "right": np.array([0.0, 1.0, 0.0]),
    }[pointing_to]

    obj_aabb = env.get_obj_bbox(obj_name)
    obj_offset = {
        "down": (obj_aabb[1] - obj_aabb[0])[2] / 2.0,
        "left": (obj_aabb[1] - obj_aabb[0])[1] / 2.0,
        "right": (obj_aabb[1] - obj_aabb[0])[1] / 2.0,
    }[pointing_to]

    obj_pos = env.get_obj_pos(obj_name)
    target_pos = obj_pos + (depth - obj_offset) * direction

    if env.ee_type == "gripper":
        obj_angle = env.get_obj_angle(obj_name)
        target = env.gripper_state["angle"]
        candidates = [obj_angle + n * np.pi for n in range(2)]

        def angle_diff(a, b):
            return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

        angle = min(candidates, key=lambda x: angle_diff(x, target))
    else:
        angle = 0.0

    move_to_position(env, target_pos, pointing_to, angle=angle)
    return True


def move_to_position(env, pos, pointing_to="down", lift_clearance=0.12, angle=0.0):
    """
    Move the end-effector to a specific position.

    Args:
        env: LMPWrapper environment instance.
        pos: Target position [x, y, z].
        pointing_to: Direction the gripper should point ("down", "left", "right").
        lift_clearance: Height to lift before moving horizontally.
        angle: Rotation angle around the pointing direction.
    """
    end_effector = env.env.franka.get_link(env.ee_name)

    if env.ee_type == "gripper":
        candidates = [angle + n * np.pi for n in range(2)]

        def angle_diff(a, b):
            return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

        angle = min(
            candidates,
            key=lambda x: angle_diff(env.gripper_state["angle"], x),
        )

    prev_pointing_to = env.env.scene_objects["gripper"].pointing_to
    prev_direction = {
        "down": np.array([0, 0, -1]),
        "left": np.array([0, -1, 0]),
        "right": np.array([0, 1, 0]),
    }[prev_pointing_to]
    prev_quat = direction_to_quat(prev_direction, angle=env.gripper_state["angle"])

    curr_direction = {
        "down": np.array([0, 0, -1]),
        "left": np.array([0, -1, 0]),
        "right": np.array([0, 1, 0]),
    }[pointing_to]
    curr_quat = direction_to_quat(curr_direction, angle=angle)

    env.gripper_state["angle"] = angle
    env.env.scene_objects["gripper"].pointing_to = pointing_to

    pos_offset = get_gripper_offset(env.ee_type, pointing_to)
    target_pos = pos + pos_offset
    current_pos = end_effector.get_pos().cpu().numpy()

    if prev_pointing_to == pointing_to:
        disp_direction = (target_pos - current_pos) / np.linalg.norm(
            target_pos - current_pos
        )
        error = np.linalg.norm(np.cross(prev_direction, disp_direction))
        if error < 0.05:
            lift_clearance = 0.0

    ascend_pos = current_pos - prev_direction * lift_clearance
    descend_pos = target_pos - curr_direction * lift_clearance

    waypoints = [ascend_pos, descend_pos, target_pos]
    quaternions = [prev_quat, curr_quat, curr_quat]

    if len(env.trajectory) == 1:
        waypoints.pop(0)
        quaternions.pop(0)

    for i, (waypoint, quat) in enumerate(zip(waypoints, quaternions)):
        qpos = env.env.franka.inverse_kinematics(
            link=end_effector, pos=waypoint, quat=quat
        )

        if i == len(waypoints) - 2:

            holding_object = env._grasp["active"] or env._welded["active"]
            use_path_planning = (
                env.ee_type == "gripper"
                and not holding_object
            )

            if use_path_planning:
                path, is_valid = env.franka.plan_path(
                    qpos_goal=qpos, num_waypoints=200, return_valid_mask=True
                )
            else:
                is_valid = None

            if use_path_planning and is_valid is not None and is_valid.all():
                for wp in path:
                    env.franka.control_dofs_position(wp)

                    if env.ee_type == "gripper":
                        if not env.env.scene_objects["gripper"].gripper_open:
                            env.env.franka.control_dofs_force(
                                np.array([-0.5, -0.5]), [7, 8]
                            )
                        else:
                            env.env.franka.control_dofs_position(
                                [0.05, 0.05], [7, 8]
                            )

                    env.step()

                for _ in range(50):
                    env.step()
            else:
                dist = np.linalg.norm(waypoints[i] - waypoints[i + 1])
                steps = max(int(dist * 150), 80)
                _execute_joint_motion(env, qpos[:7], steps=steps)
        else:
            _execute_joint_motion(env, qpos[:7], steps=120)

            for _ in range(50):
                env.step()

    env.trajectory.append(env.env.capture_obs())


def move_parallel(env, move_dir, offset, pointing_to="down"):
    """
    Move the gripper in a direction parallel to the workspace.

    Args:
        env: LMPWrapper environment instance.
        move_dir: Direction to move ("left", "right", "front", "back", "up", "down").
        offset: Distance to move.
        pointing_to: Direction the gripper should point.
    """
    if move_dir not in ["left", "right", "front", "back", "up", "down"]:
        raise ValueError(
            "Invalid direction. Choose from 'left', 'right', 'front', 'back', 'up', 'down' "
        )
    if offset <= 0:
        raise ValueError("Offset must be a positive value.")

    if move_dir == "left":
        offset *= -1

    end_effector = env.franka.get_link(env.ee_name)
    direction = {
        "down": np.array([0, 0, -1]),
        "left": np.array([0, -1, 0]),
        "right": np.array([0, 1, 0]),
    }[pointing_to]
    quat = direction_to_quat(direction, angle=env.gripper_state["angle"])
    env.env.scene_objects["gripper"].pointing_to = pointing_to

    current_pos = end_effector.get_pos().cpu().numpy()
    if move_dir == "front":
        waypoint = np.array(
            [current_pos[0] + offset, current_pos[1], current_pos[2]]
        )
    elif move_dir == "back":
        waypoint = np.array(
            [current_pos[0] - offset, current_pos[1], current_pos[2]]
        )
    elif move_dir == "up":
        waypoint = np.array(
            [current_pos[0], current_pos[1], current_pos[2] + offset]
        )
    elif move_dir == "down":
        waypoint = np.array(
            [current_pos[0], current_pos[1], current_pos[2] - offset]
        )
    else:
        waypoint = np.array(
            [current_pos[0], current_pos[1] + offset, current_pos[2]]
        )

    qpos = env.env.franka.inverse_kinematics(
        link=end_effector, pos=waypoint, quat=quat
    )
    _execute_joint_motion(env, qpos[:7])

    env.trajectory.append(env.env.capture_obs())


def rotate_gripper(env, angle, steps=80):
    """
    Rotate the gripper around its pointing axis (wrist rotation).

    Args:
        env: LMPWrapper environment instance.
        angle: Rotation angle delta in degrees (relative to current angle).
        steps: Number of interpolation steps for smooth motion.
    """





    angle_delta = np.radians(angle)


    env.gripper_state["angle"] += angle_delta


    current_qpos = env.env.franka.get_qpos().cpu().numpy()[:7].astype(np.float32)
    goal_qpos = current_qpos.copy()
    goal_qpos[6] -= angle_delta


    _execute_joint_motion(env, goal_qpos, steps=steps)

    env.trajectory.append(env.env.capture_obs())


def open_gripper(env):
    """
    Open the gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_strict and env.ee_type != "gripper":
        print("[env] open_gripper called but end-effector is not gripper.")
        return

    src_qpos = env.env.franka.get_qpos().cpu().numpy()[-2:]
    tgt_qpos = np.array([0.05, 0.05])
    step = (tgt_qpos - src_qpos) / 50

    if env._welded["active"]:
        rigid = env.env.scene.sim.rigid_solver
        link_object = env._welded["obj_link_idx"]
        link_franka = env.franka.get_link(env.ee_name).idx
        rigid.delete_weld_constraint(link_object, link_franka)
        env._welded.update(
            dict(
                active=False,
                object=None,
                obj_link_idx=None,
            )
        )

    for i in range(50):
        if env.ee_type == "gripper":
            env.env.franka.control_dofs_position(src_qpos + step * i, [7, 8])
        env.step()

    env.env.scene_objects["gripper"].gripper_open = True
    env.trajectory.append(env.env.capture_obs())


def close_gripper(env):
    """
    Close the gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_strict and env.ee_type != "gripper":
        print("[env] close_gripper called but end-effector is not gripper.")
        return

    if env.ee_type == "gripper":
        env.env.franka.control_dofs_force(np.array([-0.5, -0.5]), [7, 8])
    for _ in range(50):
        env.step()

    if not env._grasp["active"] and not env._welded["active"]:
        for obj_name in env.env.scene_objects:
            if obj_name in ["gripper", "floor"] or "handle" in obj_name:
                continue

            scene_obj = env.env.scene_objects[obj_name]

            if env.obj_in_gripper(obj_name):
                rigid = env.env.scene.sim.rigid_solver

                if not hasattr(scene_obj, "links"):
                    continue

                link_object = scene_obj.links[0].idx
                link_franka = env.franka.get_link(env.ee_name).idx

                rigid.add_weld_constraint(link_object, link_franka)
                env._welded.update(
                    dict(
                        active=True,
                        object=obj_name,
                        obj_link_idx=link_object,
                    )
                )
                break

    env.env.scene_objects["gripper"].gripper_open = False
    env.trajectory.append(env.env.capture_obs())


def grasp_handle(env, handle_name):
    """
    Grasp a handle object.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle to grasp.

    Returns:
        True if successful, False otherwise.
    """
    if env.ee_strict and env.ee_type != "gripper":
        print("[env] grasp_handle called but end-effector is not gripper.")
        return False

    if env._grasp["active"] and env._grasp["object"] == handle_name:
        close_gripper(env)
        return True

    if "handle" not in handle_name or handle_name not in env.env.scene_objects:
        return False

    if env._welded["active"]:
        return False

    handle = env.env.scene_objects[handle_name]
    ee = env.franka.get_link(env.ee_name)

    handle_pos = env.get_obj_pos(handle_name)
    gripper_pos = ee.get_pos().cpu().numpy() - get_gripper_offset(
        env.ee_type, env.env.scene_objects["gripper"].pointing_to
    )

    displacement = gripper_pos - handle_pos
    distance = np.linalg.norm(displacement)

    if distance > 0.1:
        return False

    env._grasp.update(
        dict(
            active=True,
            object=handle_name,
            obj_link_idx=handle.idx,
        )
    )
    close_gripper(env)
    return True


def release_handle(env):
    """
    Release a grasped handle.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_strict and env.ee_type != "gripper":
        print("[env] release_handle called but end-effector is not gripper.")
        return

    env._grasp.update(
        dict(
            active=False,
            object=None,
            obj_link_idx=None,
        )
    )
    open_gripper(env)


def pick(env, obj_name, pointing_to="down"):
    """
    Pick up an object.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to pick.
        pointing_to: Approach direction.

    Returns:
        True if successful, False otherwise.
    """
    if not move_gripper_to(env, obj_name, pointing_to=pointing_to):
        return False
    close_gripper(env)
    return True


def place(env, obj_name, pointing_to="down"):
    """
    Place an object at a target location.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the target object/location.
        pointing_to: Approach direction.

    Returns:
        True if successful, False otherwise.
    """
    if not move_gripper_to(env, obj_name, pointing_to=pointing_to):
        return False
    open_gripper(env)
    return True



def open_robotiq85(env):
    """
    Open the Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "robotiq85":
        print("[env] open_robotiq85 called but end-effector is not robotiq85.")
        return

    if env._welded["active"]:
        rigid = env.env.scene.sim.rigid_solver
        link_object = env._welded["obj_link_idx"]
        link_franka = env.franka.get_link(env.ee_name).idx
        rigid.delete_weld_constraint(link_object, link_franka)
        env._welded.update(
            dict(
                active=False,
                object=None,
                obj_link_idx=None,
            )
        )


    src_qpos = env.env.franka.get_qpos().cpu().numpy()[7]
    tgt_qpos = 0.7
    steps = 50

    for i in range(steps):
        alpha = i / steps
        interp = (1 - alpha) * src_qpos + alpha * tgt_qpos
        env.env.franka.control_dofs_position([interp], [7])
        env.step()

    env.env.scene_objects["gripper"].gripper_open = True
    env.trajectory.append(env.env.capture_obs())


def close_robotiq85(env):
    """
    Close the Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "robotiq85":
        print("[env] close_robotiq85 called but end-effector is not robotiq85.")
        return



    env.env.franka.control_dofs_force(np.array([-50.0]), [7])
    for _ in range(50):
        env.step()

    if not env._grasp["active"] and not env._welded["active"]:
        for obj_name in env.env.scene_objects:
            if obj_name in ["gripper", "floor"] or "handle" in obj_name:
                continue

            scene_obj = env.env.scene_objects[obj_name]


            in_gripper = env.obj_in_gripper(obj_name)
            print(f"[DEBUG close_robotiq85] Checking {obj_name}: in_gripper={in_gripper}")

            if in_gripper:
                rigid = env.env.scene.sim.rigid_solver

                if not hasattr(scene_obj, "links"):
                    print(f"[DEBUG close_robotiq85] {obj_name} has no links attribute, skipping...")
                    continue

                link_object = scene_obj.links[0].idx
                link_franka = env.franka.get_link(env.ee_name).idx

                rigid.add_weld_constraint(link_object, link_franka)
                env._welded.update(
                    dict(
                        active=True,
                        object=obj_name,
                        obj_link_idx=link_object,
                    )
                )
                print(f"[DEBUG close_robotiq85] Welded {obj_name} to gripper!")
                break

    env.env.scene_objects["gripper"].gripper_open = False
    env.trajectory.append(env.env.capture_obs())


def grasp_handle_robotiq85(env, handle_name):
    """
    Grasp a handle with Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle to grasp.

    Returns:
        True if successful, False otherwise.
    """
    if env.ee_type != "robotiq85":
        print("[env] grasp_handle_robotiq85 called but end-effector is not robotiq85.")
        return False

    if env._grasp["active"] and env._grasp["object"] == handle_name:
        close_robotiq85(env)
        return True

    if "handle" not in handle_name or handle_name not in env.env.scene_objects:
        return False

    if env._welded["active"]:
        return False

    handle = env.env.scene_objects[handle_name]
    ee = env.franka.get_link(env.ee_name)

    handle_pos = env.get_obj_pos(handle_name)
    gripper_pos = ee.get_pos().cpu().numpy() - get_gripper_offset(
        env.ee_type, env.env.scene_objects["gripper"].pointing_to
    )

    displacement = gripper_pos - handle_pos
    distance = np.linalg.norm(displacement)

    if distance > 0.1:
        return False

    env._grasp.update(
        dict(
            active=True,
            object=handle_name,
            obj_link_idx=handle.idx,
        )
    )
    close_robotiq85(env)
    return True


def release_handle_robotiq85(env):
    """
    Release a grasped handle with Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "robotiq85":
        print("[env] release_handle_robotiq85 called but end-effector is not robotiq85.")
        return

    env._grasp.update(
        dict(
            active=False,
            object=None,
            obj_link_idx=None,
        )
    )
    open_robotiq85(env)


def pick_robotiq85(env, obj_name, pointing_to="down"):
    """
    Pick up an object with Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to pick.
        pointing_to: Approach direction.

    Returns:
        True if successful, False otherwise.
    """
    if env.ee_type != "robotiq85":
        print("[env] pick_robotiq85 called but end-effector is not robotiq85.")
        return False

    if not move_gripper_to(env, obj_name, pointing_to=pointing_to):
        return False
    close_robotiq85(env)
    return True


def place_robotiq85(env, obj_name, pointing_to="down"):
    """
    Place an object with Robotiq85 gripper.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the target object/location.
        pointing_to: Approach direction.

    Returns:
        True if successful, False otherwise.
    """
    if env.ee_type != "robotiq85":
        print("[env] place_robotiq85 called but end-effector is not robotiq85.")
        return False

    if not move_gripper_to(env, obj_name, pointing_to=pointing_to):
        return False
    open_robotiq85(env)
    return True



def activate_vacuum(env):
    """
    Activate the vacuum gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "suction":
        print("[env] activate_vacuum called but end-effector is not suction.")
        return

    strict = env.ee_strict
    env.ee_strict = False
    close_gripper(env)
    env.ee_strict = strict


def deactivate_vacuum(env):
    """
    Deactivate the vacuum gripper.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "suction":
        print("[env] deactivate_vacuum called but end-effector is not suction.")
        return

    strict = env.ee_strict
    env.ee_strict = False
    open_gripper(env)
    env.ee_strict = strict


def attach_vacuum_handle(env, handle_name):
    """
    Attach the vacuum gripper to a handle.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle.

    Returns:
        True if successful, False otherwise.
    """
    if env.ee_type != "suction":
        print("[env] attach_vacuum_handle called but end-effector is not suction.")
        return False

    strict = env.ee_strict
    env.ee_strict = False
    result = grasp_handle(env, handle_name)
    env.ee_strict = strict
    return result


def detach_vacuum_handle(env):
    """
    Detach the vacuum gripper from a handle.

    Args:
        env: LMPWrapper environment instance.
    """
    if env.ee_type != "suction":
        print("[env] detach_vacuum_handle called but end-effector is not suction.")
        return

    strict = env.ee_strict
    env.ee_strict = False
    release_handle(env)
    env.ee_strict = strict



def _execute_joint_motion(env, goal_qpos, steps=80):
    """Execute smooth joint motion to goal position."""
    if torch.is_tensor(goal_qpos):
        goal_qpos = goal_qpos.detach().cpu().numpy()
    goal_qpos = np.asarray(goal_qpos, dtype=np.float32)

    current_qpos = env.env.franka.get_qpos().cpu().numpy()[:7].astype(np.float32)
    lin = np.linspace(0, 1, steps)
    alphas = np.where(lin < 0.5, 2 * lin**2, 1 - 2 * (1 - lin) ** 2)

    for alpha in alphas:
        interp = (1 - alpha) * current_qpos + alpha * goal_qpos
        env.env.franka.control_dofs_position(interp, range(7))

        if (
            env.ee_type == "gripper"
            and not env.env.scene_objects["gripper"].gripper_open
        ):
            env.env.franka.control_dofs_force(np.array([-0.5, -0.5]), [7, 8])

        env.step()

    env.env.franka.control_dofs_position(goal_qpos, range(7))
    for _ in range(50):
        env.step()


def _feasibility_check(env, obj_name):
    """Check if an object is reachable."""

    if any(
        [
            obj_name in ["apple", "orange", "lemon"],
            "cube" in obj_name,
            "cylinder" in obj_name,
        ]
    ):
        floor_low, floor_high = map(np.asarray, FLOOR_AABB)
        ol, oh = map(np.asarray, env.get_obj_bbox(obj_name))
        if (
            (ol[0] < floor_low[0])
            or (oh[0] > floor_high[0])
            or (ol[1] < floor_low[1])
            or (oh[1] > floor_high[1])
        ):
            return False


    if "cube" in obj_name or "cylinder" in obj_name:
        target_bbox = env.get_obj_bbox(obj_name)
        if target_bbox is not None:
            tl, th = map(np.asarray, target_bbox)

            tol_xy = 0.005
            tol_z = 0.004

            for other_name, other in env.env.scene_objects.items():
                if other_name == obj_name:
                    continue
                if not ("cube" in other_name or "cylinder" in other_name):
                    continue

                other_bbox = env.get_obj_bbox(other_name)
                if other_bbox is None:
                    continue

                ol, oh = map(np.asarray, other_bbox)

                overlap_x = (tl[0] - tol_xy) < oh[0] and (th[0] + tol_xy) > ol[0]
                overlap_y = (tl[1] - tol_xy) < oh[1] and (th[1] + tol_xy) > ol[1]
                if not (overlap_x and overlap_y):
                    continue

                if ol[2] >= th[2] - tol_z and ol[2] > tl[2] + tol_z:
                    return False


    if obj_name in ["bottom_drawer_handle"]:
        top_drawer = env.env.scene_objects.get("top_drawer")
        if top_drawer is not None:
            top_bbox = env.get_obj_bbox("top_drawer")
            handle_bbox = env.get_obj_bbox(obj_name)

            if top_bbox is not None and handle_bbox is not None:
                tl, th = map(np.asarray, top_bbox)
                hl, hh = map(np.asarray, handle_bbox)

                overlap_x = (hl[0] - 0.005) < th[0] and (hh[0] + 0.005) > tl[0]
                overlap_y = (hl[1] - 0.005) < th[1] and (hh[1] + 0.005) > tl[1]
                overlaps_xy = overlap_x and overlap_y

                if overlaps_xy:
                    return False
    return True