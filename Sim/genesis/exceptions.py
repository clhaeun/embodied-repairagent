"""
Custom exceptions for Genesis task execution.

Provides SubtaskFailure for subtask-level failure handling and repair.
"""

from typing import Any, Dict, Optional


class SubtaskFailure(Exception):
    """
    Exception raised when a subtask check fails.

    This exception is used to trigger subtask-level repair logic in genesis_loop.py.
    When raised, the loop can attempt to repair and re-execute the failed subtask.

    Attributes:
        subtask_name: Name of the failed subtask (e.g., "pick_single", "place_single").
        obj_name: Name of the object involved in the subtask (if applicable).
        target_name: Name of the target (if applicable).
        context: Additional context information for repair.
        message: Human-readable error message.
    """

    def __init__(
        self,
        subtask_name: str,
        obj_name: Optional[str] = None,
        target_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ):
        self.subtask_name = subtask_name
        self.obj_name = obj_name
        self.target_name = target_name
        self.context = context or {}


        if message is None:
            parts = [f"Subtask '{subtask_name}' failed"]
            if obj_name:
                parts.append(f"for object '{obj_name}'")
            if target_name:
                parts.append(f"with target '{target_name}'")
            message = " ".join(parts)

        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "subtask_name": self.subtask_name,
            "obj_name": self.obj_name,
            "target_name": self.target_name,
            "context": self.context,
            "message": self.message,
        }


class SubtaskNotfinished(Exception):
    """
    Exception raised when a subtask check fails.

    This exception is used to trigger subtask-level revise logic in genesis_loop.py.
    When raised, the loop can attempt to revise and re-execute the failed subtask.

    Attributes:
        subtask_name: Name of the failed subtask (e.g., "pick_single", "place_single").
        obj_name: Name of the object involved in the subtask (if applicable).
        target_name: Name of the target (if applicable).
        context: Additional context information for revise.
        message: Human-readable error message.
    """

    def __init__(
        self,
        subtask_name: str,
        obj_name: Optional[str] = None,
        target_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
    ):
        self.subtask_name = subtask_name
        self.obj_name = obj_name
        self.target_name = target_name
        self.context = context or {}


        if message is None:
            parts = [f"Subtask '{subtask_name}' not finished"]
            if obj_name:
                parts.append(f"for object '{obj_name}'")
            if target_name:
                parts.append(f"with target '{target_name}'")
            message = " ".join(parts)

        self.message = message
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "subtask_name": self.subtask_name,
            "obj_name": self.obj_name,
            "target_name": self.target_name,
            "context": self.context,
            "message": self.message,
        }


class SubtaskSkip(Exception):
    """
    Exception raised to skip the current subtask after max repair attempts.

    This exception signals that repair attempts have been exhausted and
    the task should continue with the next subtask.

    Attributes:
        subtask_name: Name of the subtask to skip.
        obj_name: Name of the object involved (if applicable).
        repair_attempts: Number of repair attempts made.
        message: Human-readable message.
        result: SubtaskResult object containing execution details.
    """

    def __init__(
        self,
        subtask_name: str,
        obj_name: Optional[str] = None,
        repair_attempts: int = 0,
        message: Optional[str] = None,
        result: Optional[Any] = None,
    ):
        self.subtask_name = subtask_name
        self.obj_name = obj_name
        self.repair_attempts = repair_attempts
        self.result = result

        if message is None:
            message = (
                f"Skipping subtask '{subtask_name}' after {repair_attempts} repair attempts"
            )
            if obj_name:
                message += f" for object '{obj_name}'"

        self.message = message
        super().__init__(self.message)
