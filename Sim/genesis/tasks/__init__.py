"""
Task execution modules for Genesis Simulator.

Each task module contains a run_task function that defines the sequence
of primitive skills to execute for completing the task.
"""

from .pick_place import run_task as run_pick_place
from .put_in_hinge import run_task as run_put_in_hinge
from .put_in_prismatic import run_task as run_put_in_prismatic
from .sweep import run_task as run_sweep
from .insert_slot import run_task as run_insert_slot
from .pour_liquid import run_task as run_pour_liquid

# Pick Place 2 variants (BasePickPlaceTask)
# All variants use the same run_task function - it handles all ball/cube/cylinder/inverse variants
from .pick_place_2 import run_task as run_pick_place_2

__all__ = [
    # Original tasks
    "run_pick_place",
    "run_put_in_hinge",
    "run_put_in_prismatic",
    "run_sweep",
    # New tasks
    "run_insert_slot",
    "run_pour_liquid",
    # Pick Place 2 (handles all ball/cube/cylinder/inverse variants)
    "run_pick_place_2",
]
