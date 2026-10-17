"""
Subtask success/failure check functions for Put in Prismatic Drawer task.
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


def check_place_success(env, obj_name, target_name="bottom_drawer"):
    """
    Check if place subtask succeeded (object is inside drawer).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was placed.
        target_name: Name of the target drawer.

    Returns:
        True if object bbox is inside drawer bbox.
    """

    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False


    drawer_bbox = env.get_obj_bbox(target_name)
    obj_bbox = env.get_obj_bbox(obj_name)

    obj_min, obj_max = obj_bbox[0], obj_bbox[1]
    drawer_min, drawer_max = drawer_bbox[0], drawer_bbox[1]


    margin = np.array([0.02, 0.02, 0.03])
    in_drawer = (
        np.all(obj_min >= drawer_min - margin) and
        np.all(obj_max <= drawer_max + margin)
    )
    return in_drawer


def check_open_drawer_success(env, drawer_type="bottom", open_threshold=0.08):
    """
    Check if open drawer subtask succeeded.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom" drawer.
        open_threshold: Minimum DOF position to consider drawer open.

    Returns:
        True if drawer DOF position is above threshold.
    """
    drawer_name = f"{drawer_type}_drawer"
    drawer = env.env.scene_objects.get(drawer_name)

    if drawer is None:
        return False

    dof_idx = 1 if drawer_type == "top" else 0
    dof_pos = drawer.entity.get_dofs_position()[dof_idx].item()

    return dof_pos >= open_threshold


def check_close_drawer_success(env, drawer_type="bottom", close_threshold=0.08):
    """
    Check if close drawer subtask succeeded.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom" drawer.
        close_threshold: Maximum DOF position to consider drawer closed.

    Returns:
        True if drawer DOF position is below threshold.
    """
    drawer_name = f"{drawer_type}_drawer"
    drawer = env.env.scene_objects.get(drawer_name)

    if drawer is None:
        return False

    dof_idx = 1 if drawer_type == "top" else 0
    dof_pos = drawer.entity.get_dofs_position()[dof_idx].item()

    return abs(dof_pos) < close_threshold
