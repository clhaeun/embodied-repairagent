"""
Dynamic subtasks module loader.

This module dynamically exports functions from either subtasks_panda or subtasks_suction
based on the TARGET_ROBOT environment variable or module-level configuration.

Usage:
    # Set target robot before importing
    import subtasks
    subtasks.set_target_robot("suction")  # or "panda"

    # Then import functions
    from subtasks import pick_single, place_single

    # Or in tasks files:
    from subtasks import pick_single, place_single
"""

import sys
import importlib
from typing import Optional

# Default target robot
_target_robot: str = "panda"

# Module references
_subtasks_module = None


def set_target_robot(robot: str) -> None:
    """
    Set the target robot type and reload subtask functions.

    Args:
        robot: Robot type - "panda", "suction", or "robotiq"
    """
    global _target_robot, _subtasks_module

    if "robotiq" in robot:
        robot = "robotiq"
        
    if robot not in ("panda", "suction", "robotiq"):
        raise ValueError(f"Invalid robot type: {robot}. Must be 'panda', 'suction', or 'robotiq'")

    _target_robot = robot
    _reload_subtasks()


def get_target_robot() -> str:
    """Get the current target robot type."""
    return _target_robot


def _reload_subtasks() -> None:
    """Reload subtask functions from the appropriate module."""
    global _subtasks_module

    module_name = f"subtasks_{_target_robot}"

    try:
        _subtasks_module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Failed to import {module_name}: {e}")

    # Update this module's namespace with functions from the target module
    current_module = sys.modules[__name__]

    # Get all exported names from the target module
    if hasattr(_subtasks_module, '__all__'):
        export_names = _subtasks_module.__all__
    else:
        export_names = [name for name in dir(_subtasks_module) if not name.startswith('_')]

    # Export functions to this module's namespace
    for name in export_names:
        if hasattr(_subtasks_module, name):
            setattr(current_module, name, getattr(_subtasks_module, name))


# Initialize with default robot (panda)
_reload_subtasks()

# Define __all__ dynamically
def _get_all():
    if _subtasks_module and hasattr(_subtasks_module, '__all__'):
        return list(_subtasks_module.__all__) + ['set_target_robot', 'get_target_robot']
    return ['set_target_robot', 'get_target_robot']

__all__ = _get_all()
