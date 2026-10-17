"""
Subtask success/failure check functions for Pick and Place task.
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


def check_place_success(env, obj_name, target_name="toy_box", margin=0.02):
    """
    Check if place subtask succeeded (object is inside target container).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was placed.
        target_name: Name of the target container.
        margin: Margin for bbox comparison.

    Returns:
        True if object bbox is inside target bbox.
    """
    # Object should not be in gripper
    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False

    # Check if object is inside target container
    target_bbox = env.get_obj_bbox(target_name)
    obj_bbox = env.get_obj_bbox(obj_name)

    # Expand target bbox with margin for height
    target_min = target_bbox[0] - np.array([margin, margin, 0.01])
    target_max = target_bbox[1] + np.array([margin, margin, 0.25])

    obj_min, obj_max = obj_bbox[0], obj_bbox[1]

    in_target = (
        np.all(obj_min >= target_min) and
        np.all(obj_max <= target_max)
    )
    return in_target


def check_unstack_success(env, obj_name):
    """
    Check if unstack subtask succeeded.
    Object should be picked up (in gripper) or placed on floor.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was unstacked.

    Returns:
        True if object is in gripper or on floor level.
    """
    # If in gripper, unstack is in progress (success)
    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return True

    # Check if object is now at floor level (not stacked)
    obj_bbox = env.get_obj_bbox(obj_name)
    obj_bottom_z = obj_bbox[0][2]

    # Object should be near floor level (z ~ 0.5)
    return obj_bottom_z < 0.52
