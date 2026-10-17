"""
Subtask success/failure check functions for Pick and Place 2 task.
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


def check_place_success(env, obj_name, target_name="tray"):
    """
    Check if place subtask succeeded (object is inside tray).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object that was placed.
        target_name: Name of the target tray.

    Returns:
        True if object bbox is inside tray area.
    """

    if env.obj_in_gripper(obj_name) and not env.gripper_is_open():
        return False


    tray_bbox = env.get_obj_bbox(target_name)
    center = (tray_bbox[0] + tray_bbox[1]) / 2
    half_size = (tray_bbox[1] - tray_bbox[0]) / 2 * 0.8

    tray_min = center - half_size
    tray_max = center + half_size


    obj_bbox = env.get_obj_bbox(obj_name)


    xy_ok = np.all(
        (tray_min[:2] - 0.04 < obj_bbox[0][:2]) &
        (obj_bbox[1][:2] < tray_max[:2] + 0.04)
    )
    z_ok = tray_bbox[0][2] - 0.02 < obj_bbox[0][2] < tray_bbox[1][2] + 0.02

    return xy_ok and z_ok


def check_pick_and_place_success(env, obj_name, target_name="tray"):
    """
    Check if combined pick_and_place subtask succeeded.
    This is a convenience function that checks if the object ended up in the tray.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object.
        target_name: Name of the target tray.

    Returns:
        True if object is in tray and not in gripper.
    """
    return check_place_success(env, obj_name, target_name)


def check_lid_open(env, lid_name="box_lid"):
    """
    Check if the box lid is open by checking handle height AND gripper state.

    Args:
        env: LMPWrapper environment instance.
        lid_name: Name of the lid entity.

    Returns:
        True if lid is open (handle is raised above body AND gripper has released).
    """

    handle_bbox = env.get_obj_bbox("box_handle")
    body_bbox = env.get_obj_bbox("box_body")

    if handle_bbox is None or body_bbox is None:
        return False


    handle_bottom_z = handle_bbox[0][2]
    body_top_z = body_bbox[1][2]


    gripper_released = not env._grasp.get("active", False)

    print(f"[CheckLidOpen] handle_bottom_z={handle_bottom_z:.3f}, body_top_z={body_top_z:.3f}, gripper_released={gripper_released}")




    height_ok = handle_bottom_z > body_top_z + 0.15
    return height_ok and gripper_released
