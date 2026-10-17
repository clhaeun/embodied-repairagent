"""
Subtask success/failure check functions for Insert Slot task.
"""

import numpy as np


def check_remove_from_slot_success(env, obj_name="blocking_object"):
    """
    Check if remove_from_slot subtask succeeded.
    Object should be picked up (in gripper).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the blocking object.

    Returns:
        True if object is in gripper and gripper is closed.
    """
    in_gripper = env.obj_in_gripper(obj_name)
    gripper_closed = not env.gripper_is_open()
    return in_gripper and gripper_closed


def check_place_beside_slot_success(env, obj_name="blocking_object", slot_name="auxiliary_slot"):
    """
    Check if place_beside_slot subtask succeeded.
    Object should be placed beside the slot (not inside it).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was placed.
        slot_name: Name of the slot it was placed beside.

    Returns:
        True if object is beside slot and not in gripper.
    """
    # Object should not be in gripper
    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False

    # Get slot and object positions
    slot_bbox = env.get_obj_bbox(slot_name)
    obj_pos = env.get_obj_pos(obj_name)

    # Object should be near slot's xy but not inside it
    slot_center = (slot_bbox[0] + slot_bbox[1]) / 2

    # Check if object is near the slot area (within reasonable distance)
    dist_xy = np.linalg.norm(obj_pos[:2] - slot_center[:2])
    near_slot = dist_xy < 0.15

    # Check that object is not inside the slot
    obj_bbox = env.get_obj_bbox(obj_name)
    margin = 0.01
    inside_slot_xy = (
        obj_bbox[0][0] > slot_bbox[0][0] + margin and
        obj_bbox[1][0] < slot_bbox[1][0] - margin and
        obj_bbox[0][1] > slot_bbox[0][1] + margin and
        obj_bbox[1][1] < slot_bbox[1][1] - margin
    )

    return near_slot and not inside_slot_xy


def check_insert_into_slot_success(env, obj_name="target_object", slot_name="target_slot"):
    """
    Check if insert_into_slot subtask succeeded.
    Object should be inserted into the slot.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the target object.
        slot_name: Name of the target slot.

    Returns:
        True if object is properly inserted into slot.
    """
    # Object should not be in gripper
    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False

    # Get bboxes
    target_bbox = env.get_obj_bbox(obj_name)
    slot_bbox = env.get_obj_bbox(slot_name)

    # Use negative margin to allow slight overflow (more lenient)
    margin = -0.01

    # Check if target's xy AABB is within slot's xy AABB
    target_in_slot_xy = (
        target_bbox[0][0] > slot_bbox[0][0] + margin and
        target_bbox[1][0] < slot_bbox[1][0] - margin and
        target_bbox[0][1] > slot_bbox[0][1] + margin and
        target_bbox[1][1] < slot_bbox[1][1] - margin
    )

    # Check if target's bottom Z is near or below slot's middle Z (inserted)
    slot_mid_z = (slot_bbox[0][2] + slot_bbox[1][2]) / 2
    target_inserted_z = target_bbox[0][2] < slot_mid_z + 0.02

    return target_in_slot_xy and target_inserted_z
