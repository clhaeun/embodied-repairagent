"""
Open Drawer Subtask.

Opens a prismatic drawer by grasping and pulling the handle.
"""

from skill_code import (
    move_gripper_to,
    move_parallel,
    grasp_handle,
    release_handle,
)


def open_drawer(env, drawer_type="bottom"):
    """
    Open a prismatic drawer.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom" drawer to open.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[OpenDrawer] Opening {drawer_type} drawer")


    handle_name = "top_drawer_handle" if drawer_type == "top" else "bottom_drawer_handle"


    success = move_gripper_to(env, handle_name, pointing_to="right")
    if not success:
        print(f"[OpenDrawer] Failed to move to handle")
        return False


    grasp_handle(env, handle_name)


    move_parallel(env, "left", 0.12, pointing_to="right")


    release_handle(env)


    move_parallel(env, "left", 0.05, pointing_to="right")

    print(f"[OpenDrawer] Successfully opened {drawer_type} drawer")
    return True