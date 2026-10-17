"""
Open Box Lid Subtask.

Opens the lid of a box by grasping and lifting the handle.
Used by pick_place_2 variants.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    grasp_handle,
    release_handle,
)


def open_box_lid(env, handle_name="box_handle", lid_name="box_lid"):
    """
    Open the box lid.

    Args:
        env: LMPWrapper environment instance.
        handle_name: Name of the handle to grasp.
        lid_name: Name of the lid entity (for state checking).

    Returns:
        True if successful, False otherwise.
    """
    print(f"[OpenBoxLid] Opening box lid")


    success = move_gripper_to(env, handle_name, pointing_to="down")
    if not success:
        print(f"[OpenBoxLid] Failed to move to handle")
        return False


    grasp_handle(env, handle_name)

    handle_pos = env.get_obj_pos(handle_name)
    

    for _ in range(3):
        move_parallel(env, "up", 0.09, pointing_to="down")
        move_parallel(env, "right", 0.09, pointing_to="down")


    release_handle(env)


    move_parallel(env, "up", 0.05, pointing_to="down")
    move_parallel(env, "left", 0.05, pointing_to="down")
    
    print(f"[OpenBoxLid] Successfully opened box lid")

    return True