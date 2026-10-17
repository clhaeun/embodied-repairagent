"""
Subtask success/failure check functions for each task.

Each module provides check functions to verify if a subtask succeeded.
Also provides SUBTASK_CHECK_MAP for automatic subtask-to-check function mapping.
"""

from tasks.checks.pick_place import (
    check_pick_success as pick_place_check_pick,
    check_place_success as pick_place_check_place,
    check_unstack_success,
)

from tasks.checks.sweep import (
    check_sweep_success,
    check_relocate_success,
)

from tasks.checks.put_in_prismatic import (
    check_pick_success as prismatic_check_pick,
    check_place_success as prismatic_check_place,
    check_open_drawer_success,
    check_close_drawer_success,
)

from tasks.checks.put_in_hinge import (
    check_pick_success as hinge_check_pick,
    check_place_success as hinge_check_place,
    check_open_hinge_success,
    check_close_hinge_success,
    check_move_obstacle_success,
)

from tasks.checks.pick_place_2 import (
    check_pick_success as pick_place_2_check_pick,
    check_place_success as pick_place_2_check_place,
    check_pick_and_place_success,
)

from tasks.checks.insert_slot import (
    check_remove_from_slot_success,
    check_place_beside_slot_success,
    check_insert_into_slot_success,
)

from tasks.checks.pour_liquid import (
    check_grasp_container_success,
    check_pour_success,
    check_release_container_success,
)















SUBTASK_CHECK_MAP = {



    "pick_place": {
        "pick_single": pick_place_check_pick,
        "place_single": pick_place_check_place,
        "unstack_single": check_unstack_success,
    },




    "sweep": {
        "sweep_single": check_sweep_success,
        "relocate_single": check_relocate_success,
    },




    "put_in_prismatic": {
        "pick_single": prismatic_check_pick,
        "place_single": prismatic_check_place,
        "open_drawer": check_open_drawer_success,
        "close_drawer": check_close_drawer_success,
    },




    "put_in_hinge": {
        "pick_single": hinge_check_pick,
        "place_single": hinge_check_place,
        "place_at_position": check_move_obstacle_success,
        "open_hinge_lid": check_open_hinge_success,
        "close_hinge_lid": check_close_hinge_success,
    },




    "pick_place_2": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_ball": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_cube": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_cylinder": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_inverse_ball": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_inverse_cube": {
        "pick_and_place_object": check_pick_and_place_success,
    },
    "pick_place_inverse_cylinder": {
        "pick_and_place_object": check_pick_and_place_success,
    },




    "insert_slot": {
        "remove_from_slot": check_remove_from_slot_success,
        "place_beside_slot": check_place_beside_slot_success,
        "insert_into_slot": check_insert_into_slot_success,
    },




    "pour_liquid": {
        "grasp_container_vacuum": check_grasp_container_success,
        "pour_into_container": check_pour_success,
        "release_container_vacuum": check_release_container_success,
    },
}


def get_check_function(task_name: str, subtask_name: str):
    """
    Get the check function for a given task and subtask.

    Args:
        task_name: Name of the task (e.g., "pick_place", "sweep").
        subtask_name: Name of the subtask (e.g., "pick_single", "place_single").

    Returns:
        Check function if found, None otherwise.
    """
    task_checks = SUBTASK_CHECK_MAP.get(task_name, {})
    return task_checks.get(subtask_name)


__all__ = [

    "pick_place_check_pick",
    "pick_place_check_place",
    "check_unstack_success",

    "check_sweep_success",
    "check_relocate_success",

    "prismatic_check_pick",
    "prismatic_check_place",
    "check_open_drawer_success",
    "check_close_drawer_success",

    "hinge_check_pick",
    "hinge_check_place",
    "check_open_hinge_success",
    "check_close_hinge_success",
    "check_move_obstacle_success",

    "pick_place_2_check_pick",
    "pick_place_2_check_place",
    "check_pick_and_place_success",

    "check_remove_from_slot_success",
    "check_place_beside_slot_success",
    "check_insert_into_slot_success",

    "check_grasp_container_success",
    "check_pour_success",
    "check_release_container_success",

    "SUBTASK_CHECK_MAP",
    "get_check_function",
]
