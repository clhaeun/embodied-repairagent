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

    # Get handle name based on drawer type
    handle_name = "top_drawer_handle" if drawer_type == "top" else "bottom_drawer_handle"

    # Move to handle (approach from right side)
    success = move_gripper_to(env, handle_name, pointing_to="right")
    if not success:
        print(f"[OpenDrawer] Failed to move to handle")
        return False

    # Grasp the handle
    grasp_handle(env, handle_name)

    # Pull to open (left direction)
    move_parallel(env, "left", 0.12, pointing_to="right")

    # Release handle
    release_handle(env)

    # Move back to clear
    move_parallel(env, "left", 0.05, pointing_to="right")

    print(f"[OpenDrawer] Successfully opened {drawer_type} drawer")
    return True