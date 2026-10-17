"""
Subtask success/failure check functions for Put in Hinge Box task.
"""

import numpy as np


def check_pick_success(env, obj_name):
    """
    Check if pick subtask succeeded.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was picked.

    Returns:
        True if object is in gripper and gripper is closed.
    """
    in_gripper = env.obj_in_gripper(obj_name)
    gripper_closed = not env.gripper_is_open()
    return in_gripper and gripper_closed


def check_place_success(env, obj_name, target_name="hinge_body", margin=0.02):
    """
    Check if place subtask succeeded (object is inside hinge box).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was placed.
        target_name: Name of the target container.
        margin: Margin for bbox comparison.

    Returns:
        True if object bbox is inside hinge box bbox.
    """

    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False


    box_bbox = env.get_obj_bbox(target_name)
    obj_bbox = env.get_obj_bbox(obj_name)


    box_min = box_bbox[0] - np.array([margin, margin, margin])
    box_max = box_bbox[1] + np.array([margin, margin, 0.05])

    obj_min, obj_max = obj_bbox[0], obj_bbox[1]

    in_box = (
        np.all(obj_min >= box_min) and
        np.all(obj_max <= box_max)
    )
    return in_box


def check_open_hinge_success(env, open_threshold=0.3):
    """
    Check if open hinge lid subtask succeeded.

    Args:
        env: LMPWrapper environment instance.
        open_threshold: Minimum DOF position to consider lid open.

    Returns:
        True if hinge lid DOF position is above threshold.
    """
    hinge_lid = env.env.scene_objects.get("hinge_lid")

    if hinge_lid is None:
        return False

    dof_pos = hinge_lid.entity.get_dofs_position()[0].item()
    return dof_pos >= open_threshold


def check_close_hinge_success(env, close_threshold=0.2):
    """
    Check if close hinge lid subtask succeeded.

    Args:
        env: LMPWrapper environment instance.
        close_threshold: Maximum DOF position to consider lid closed (near -0.785).

    Returns:
        True if hinge lid DOF position is near closed state.
    """
    hinge_lid = env.env.scene_objects.get("hinge_lid")

    if hinge_lid is None:
        return False

    dof_pos = hinge_lid.entity.get_dofs_position()[0].item()

    return abs(dof_pos + 0.785) < close_threshold


def check_move_obstacle_success(env, obj_name):
    """
    Check if move obstacle (off lid) subtask succeeded.
    Object should be off the hinge lid area.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the obstacle object.

    Returns:
        True if object is not on the hinge lid.
    """

    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False


    hinge_lid = env.env.scene_objects.get("hinge_lid")
    if hinge_lid is None:
        return True

    lid_bbox = env.get_obj_bbox("hinge_lid")
    obj_pos = env.get_obj_pos(obj_name)


    not_on_lid = (
        obj_pos[0] < lid_bbox[0][0] - 0.02 or
        obj_pos[0] > lid_bbox[1][0] + 0.02 or
        obj_pos[1] < lid_bbox[0][1] - 0.02 or
        obj_pos[1] > lid_bbox[1][1] + 0.02
    )
    return not_on_lid
