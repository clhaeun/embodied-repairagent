"""
Precondition checks for subtasks.

These functions check if a subtask needs to be executed or can be skipped.
Returns True if the goal state is already achieved (skip subtask),
False if subtask needs to run, None if state cannot be determined.
"""




def is_drawer_already_closed(env, drawer_type="bottom", threshold=0.05):
    """
    Check if drawer is already closed.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom".
        threshold: Position threshold for closed state.

    Returns:
        True if already closed, False if needs closing, None if unknown.
    """
    if drawer_type == "top":
        drawer_name = "top_drawer"
        dof_idx = 1
    else:
        drawer_name = "bottom_drawer"
        dof_idx = 0

    drawer = env.env.scene_objects.get(drawer_name)
    if drawer is None:
        return None

    dof_pos = drawer.entity.get_dofs_position()[dof_idx].item()
    return dof_pos <= threshold


def is_drawer_already_open(env, drawer_type="bottom", threshold=0.08):
    """
    Check if drawer is already open.

    Args:
        env: LMPWrapper environment instance.
        drawer_type: "top" or "bottom".
        threshold: Position threshold for open state.

    Returns:
        True if already open, False if needs opening, None if unknown.
    """
    if drawer_type == "top":
        drawer_name = "top_drawer"
        dof_idx = 1
    else:
        drawer_name = "bottom_drawer"
        dof_idx = 0

    drawer = env.env.scene_objects.get(drawer_name)
    if drawer is None:
        return None

    dof_pos = drawer.entity.get_dofs_position()[dof_idx].item()
    return dof_pos >= threshold




def is_hinge_lid_already_open(env, lid_name="hinge_lid", threshold=0.3):
    """
    Check if hinge lid is already open.

    Args:
        env: LMPWrapper environment instance.
        lid_name: Name of the lid entity.
        threshold: Angle threshold for open state.

    Returns:
        True if already open, False if needs opening, None if unknown.
    """
    hinge_lid = env.env.scene_objects.get(lid_name)
    if hinge_lid is None:
        return None

    dof_pos = hinge_lid.entity.get_dofs_position()[0].item()
    return dof_pos >= threshold


def is_hinge_lid_already_closed(env, lid_name="hinge_lid", threshold=0.1):
    """
    Check if hinge lid is already closed.

    Args:
        env: LMPWrapper environment instance.
        lid_name: Name of the lid entity.
        threshold: Angle threshold for closed state.

    Returns:
        True if already closed, False if needs closing, None if unknown.
    """
    hinge_lid = env.env.scene_objects.get(lid_name)
    if hinge_lid is None:
        return None

    dof_pos = hinge_lid.entity.get_dofs_position()[0].item()
    return dof_pos < threshold




def is_obj_visible(env, obj_name):
    """
    Check if object is visible.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object.

    Returns:
        True if visible, False if not visible.
    """
    return env.is_obj_visible(obj_name)


def does_obj_exist(env, obj_name):
    """
    Check if object exists (has a position).

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object.

    Returns:
        True if exists, False if not found.
    """
    return env.get_obj_pos(obj_name) is not None


