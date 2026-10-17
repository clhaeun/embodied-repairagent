"""
Subtask modules for Panda robot manipulation.

Each subtask handles a single object operation with primitive API calls.
"""

from subtasks_panda.pick_single import pick_single
from subtasks_panda.place_single import place_single
from subtasks_panda.place_at_position import place_at_position
from subtasks_panda.unstack_single import unstack_single
from subtasks_panda.sweep_single import sweep_single
from subtasks_panda.relocate_single import relocate_single
from subtasks_panda.open_hinge_lid import open_hinge_lid
from subtasks_panda.close_hinge_lid import close_hinge_lid
from subtasks_panda.open_box_lid import open_box_lid
from subtasks_panda.open_drawer import open_drawer
from subtasks_panda.close_drawer import close_drawer
from subtasks_panda.pick_and_place_object import pick_and_place_object
from subtasks_panda.remove_from_slot import remove_from_slot
from subtasks_panda.place_beside_slot import place_beside_slot
from subtasks_panda.insert_into_slot import insert_into_slot
from subtasks_panda.grasp_container import grasp_container
from subtasks_panda.pour_into_container import pour_into_container
from subtasks_panda.release_container import release_container


__all__ = [
    "pick_single",
    "place_single",
    "place_at_position",
    "unstack_single",
    "sweep_single",
    "relocate_single",
    "open_hinge_lid",
    "close_hinge_lid",
    "open_box_lid",
    "open_drawer",
    "close_drawer",
    # pick_place_2 variants
    "pick_and_place_object",
    # insert_slot
    "remove_from_slot",
    "place_beside_slot",
    "insert_into_slot",
    # pour_liquid
    "grasp_container",
    "pour_into_container",
    "release_container",
]
