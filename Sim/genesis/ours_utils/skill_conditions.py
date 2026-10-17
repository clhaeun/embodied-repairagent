"""
Skill condition specifications for Genesis simulator.

This module defines the preconditions, effects, and postconditions for primitive skills
in the Genesis environment. It is adapted from the RLBench version to match Genesis APIs.

Genesis skills include:
- move_gripper_to: Move gripper to target object
- move_to_position: Move to specific 3D position
- move_parallel: Move parallel to workspace
- rotate_gripper: Rotate gripper wrist
- open_gripper / close_gripper: Gripper control
- pick / place: Pick and place objects
- grasp_handle / release_handle: Handle manipulation
- activate_vacuum / deactivate_vacuum: Suction gripper control
- Robotiq85 variants: open_robotiq85, close_robotiq85, etc.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union


# =============================================================================
# Condition Spec Dataclasses
# =============================================================================

@dataclass(frozen=True)
class GripperOpen:
    """Precondition: gripper must be open."""
    pass


@dataclass(frozen=True)
class GripperClosed:
    """Precondition/Postcondition: gripper must be closed."""
    pass


@dataclass(frozen=True)
class HoldingObject:
    """Precondition: gripper must be holding an object (welded)."""
    pass


@dataclass(frozen=True)
class NotHolding:
    """Precondition: gripper must not be holding any object."""
    pass


@dataclass(frozen=True)
class NearTarget:
    """Precondition: gripper must be within distance of target."""
    max_distance: float = 0.35


@dataclass(frozen=True)
class AtTarget:
    """Postcondition: gripper should be at target position."""
    tolerance: float = 0.05


@dataclass(frozen=True)
class TargetReachable:
    """Precondition: target should be reachable (soft check)."""
    max_distance: float = 0.5  # Warning threshold, not hard fail


@dataclass(frozen=True)
class NotAtTarget:
    """Precondition: gripper must NOT already be at target (for move)."""
    min_distance: float = 0.01


@dataclass(frozen=True)
class ObjectExists:
    """Precondition: object must exist in scene."""
    pass


@dataclass(frozen=True)
class ObjectVisible:
    """Precondition: object must be visible."""
    pass


@dataclass(frozen=True)
class ObjectGrasped:
    """Postcondition: object should be grasped (welded) after skill."""
    pass


@dataclass(frozen=True)
class ObjectReleased:
    """Postcondition: object should be released after skill."""
    pass


@dataclass(frozen=True)
class ObjectAtTarget:
    """Postcondition: object should be at target position after release."""
    tolerance: float = 0.1


@dataclass(frozen=True)
class IsHandle:
    """Precondition: object must be a handle."""
    pass


@dataclass(frozen=True)
class IsSuctionGripper:
    """Precondition: must be using suction end-effector."""
    pass


@dataclass(frozen=True)
class IsParallelGripper:
    """Precondition: must be using parallel gripper end-effector."""
    pass


@dataclass(frozen=True)
class IsRobotiq85Gripper:
    """Precondition: must be using Robotiq85 end-effector."""
    pass


# Union type for all conditions
ConditionSpec = Union[
    GripperOpen, GripperClosed, HoldingObject, NotHolding,
    NearTarget, AtTarget, TargetReachable, NotAtTarget,
    ObjectExists, ObjectVisible, ObjectGrasped, ObjectReleased, ObjectAtTarget,
    IsHandle, IsSuctionGripper, IsParallelGripper, IsRobotiq85Gripper
]


# =============================================================================
# Effect Spec Dataclasses
# =============================================================================

@dataclass(frozen=True)
class MoveToTarget:
    """Effect: gripper moves to target position."""
    pass


@dataclass(frozen=True)
class MoveToObject:
    """Effect: gripper moves to target object position."""
    pass


@dataclass(frozen=True)
class MoveParallel:
    """Effect: gripper moves parallel to current position."""
    pass


@dataclass(frozen=True)
class RotateWrist:
    """Effect: gripper rotates around pointing axis."""
    pass


@dataclass(frozen=True)
class CloseGripper:
    """Effect: gripper closes."""
    pass


@dataclass(frozen=True)
class OpenGripper:
    """Effect: gripper opens."""
    pass


@dataclass(frozen=True)
class GraspObject:
    """Effect: gripper grasps target object (close + weld)."""
    pass


@dataclass(frozen=True)
class ReleaseObject:
    """Effect: gripper releases held object (open + unweld)."""
    pass


@dataclass(frozen=True)
class GraspHandle:
    """Effect: gripper grasps a handle."""
    pass


@dataclass(frozen=True)
class ReleaseHandle:
    """Effect: gripper releases a handle."""
    pass


@dataclass(frozen=True)
class ActivateVacuum:
    """Effect: activate suction/vacuum."""
    pass


@dataclass(frozen=True)
class DeactivateVacuum:
    """Effect: deactivate suction/vacuum."""
    pass


# Union type for all effects
EffectSpec = Union[
    MoveToTarget, MoveToObject, MoveParallel, RotateWrist,
    CloseGripper, OpenGripper, GraspObject, ReleaseObject,
    GraspHandle, ReleaseHandle, ActivateVacuum, DeactivateVacuum
]


# =============================================================================
# Skill Spec Dataclass
# =============================================================================

@dataclass
class SkillSpec:
    """Complete specification for a primitive skill."""
    name: str
    preconditions: List[ConditionSpec] = field(default_factory=list)
    effects: List[EffectSpec] = field(default_factory=list)
    postconditions: List[ConditionSpec] = field(default_factory=list)
    transition: Optional[str] = None  # "picked" or "released" or None
    ee_type: Optional[str] = None  # "gripper", "suction", "robotiq85", or None (any)


# =============================================================================
# SKILL_SPECS: Single Source of Truth for Genesis Skills
# =============================================================================

SKILL_SPECS: Dict[str, SkillSpec] = {
    # =========================
    # Movement skills
    # =========================
    "move_gripper_to": SkillSpec(
        name="move_gripper_to",
        preconditions=[
            ObjectExists(),
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToObject(),
        ],
        postconditions=[
            AtTarget(tolerance=0.05),
        ],
        transition=None,
    ),

    "move_to_position": SkillSpec(
        name="move_to_position",
        preconditions=[
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToTarget(),
        ],
        postconditions=[
            AtTarget(tolerance=0.05),
        ],
        transition=None,
    ),

    "move_parallel": SkillSpec(
        name="move_parallel",
        preconditions=[],
        effects=[
            MoveParallel(),
        ],
        postconditions=[],
        transition=None,
    ),

    "rotate_gripper": SkillSpec(
        name="rotate_gripper",
        preconditions=[
            IsParallelGripper(),
        ],
        effects=[
            RotateWrist(),
        ],
        postconditions=[],
        transition=None,
        ee_type="gripper",
    ),

    # =========================
    # Gripper control skills (parallel gripper)
    # NOTE: ee_type=None because these can be called from other ee types
    # when ee_strict=False (e.g., activate_vacuum calls close_gripper internally)
    # =========================
    "open_gripper": SkillSpec(
        name="open_gripper",
        preconditions=[],
        effects=[
            OpenGripper(),
            ReleaseObject(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition="released",
        ee_type=None,  # Can be called from any ee_type when ee_strict=False
    ),

    "close_gripper": SkillSpec(
        name="close_gripper",
        preconditions=[
            GripperOpen(),
        ],
        effects=[
            CloseGripper(),
            GraspObject(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition="picked",
        ee_type=None,  # Can be called from any ee_type when ee_strict=False
    ),

    # =========================
    # Pick/Place composite skills
    # =========================
    "pick": SkillSpec(
        name="pick",
        preconditions=[
            ObjectExists(),
            GripperOpen(),
            NotHolding(),
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToObject(),
            GraspObject(),
        ],
        postconditions=[
            GripperClosed(),
            ObjectGrasped(),
        ],
        transition="picked",
        ee_type="gripper",
    ),

    "place": SkillSpec(
        name="place",
        preconditions=[
            ObjectExists(),
            GripperClosed(),
            HoldingObject(),
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToObject(),
            ReleaseObject(),
        ],
        postconditions=[
            GripperOpen(),
            ObjectReleased(),
        ],
        transition="released",
        ee_type="gripper",
    ),

    # =========================
    # Handle manipulation skills
    # =========================
    "grasp_handle": SkillSpec(
        name="grasp_handle",
        preconditions=[
            IsHandle(),
            ObjectExists(),
            GripperOpen(),
            NearTarget(max_distance=0.1),
            NotHolding(),
        ],
        effects=[
            GraspHandle(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition=None,
        ee_type="gripper",
    ),

    "release_handle": SkillSpec(
        name="release_handle",
        preconditions=[],
        effects=[
            ReleaseHandle(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition=None,
        ee_type="gripper",
    ),

    # =========================
    # Suction/Vacuum gripper skills
    # =========================
    "activate_vacuum": SkillSpec(
        name="activate_vacuum",
        preconditions=[
            IsSuctionGripper(),
            GripperOpen(),
        ],
        effects=[
            ActivateVacuum(),
            CloseGripper(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition="picked",
        ee_type="suction",
    ),

    "deactivate_vacuum": SkillSpec(
        name="deactivate_vacuum",
        preconditions=[
            IsSuctionGripper(),
        ],
        effects=[
            DeactivateVacuum(),
            OpenGripper(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition="released",
        ee_type="suction",
    ),

    "attach_vacuum_handle": SkillSpec(
        name="attach_vacuum_handle",
        preconditions=[
            IsSuctionGripper(),
            IsHandle(),
            ObjectExists(),
            GripperOpen(),
            NearTarget(max_distance=0.1),
        ],
        effects=[
            GraspHandle(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition=None,
        ee_type="suction",
    ),

    "detach_vacuum_handle": SkillSpec(
        name="detach_vacuum_handle",
        preconditions=[
            IsSuctionGripper(),
        ],
        effects=[
            ReleaseHandle(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition=None,
        ee_type="suction",
    ),

    # =========================
    # Robotiq85 gripper skills
    # =========================
    "open_robotiq85": SkillSpec(
        name="open_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
        ],
        effects=[
            OpenGripper(),
            ReleaseObject(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition="released",
        ee_type="robotiq85",
    ),

    "close_robotiq85": SkillSpec(
        name="close_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
            GripperOpen(),
        ],
        effects=[
            CloseGripper(),
            GraspObject(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition="picked",
        ee_type="robotiq85",
    ),

    "pick_robotiq85": SkillSpec(
        name="pick_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
            ObjectExists(),
            GripperOpen(),
            NotHolding(),
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToObject(),
            GraspObject(),
        ],
        postconditions=[
            GripperClosed(),
            ObjectGrasped(),
        ],
        transition="picked",
        ee_type="robotiq85",
    ),

    "place_robotiq85": SkillSpec(
        name="place_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
            ObjectExists(),
            GripperClosed(),
            HoldingObject(),
            TargetReachable(max_distance=0.8),
        ],
        effects=[
            MoveToObject(),
            ReleaseObject(),
        ],
        postconditions=[
            GripperOpen(),
            ObjectReleased(),
        ],
        transition="released",
        ee_type="robotiq85",
    ),

    "grasp_handle_robotiq85": SkillSpec(
        name="grasp_handle_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
            IsHandle(),
            ObjectExists(),
            GripperOpen(),
            NearTarget(max_distance=0.1),
            NotHolding(),
        ],
        effects=[
            GraspHandle(),
        ],
        postconditions=[
            GripperClosed(),
        ],
        transition=None,
        ee_type="robotiq85",
    ),

    "release_handle_robotiq85": SkillSpec(
        name="release_handle_robotiq85",
        preconditions=[
            IsRobotiq85Gripper(),
        ],
        effects=[
            ReleaseHandle(),
        ],
        postconditions=[
            GripperOpen(),
        ],
        transition=None,
        ee_type="robotiq85",
    ),
}


def get_skill_spec(skill_name: str) -> Optional[SkillSpec]:
    """Get SkillSpec for a skill by name. Returns None if not found."""
    return SKILL_SPECS.get(skill_name.lower())


def get_skills_for_ee_type(ee_type: str) -> List[str]:
    """Get list of skill names available for a given end-effector type."""
    result = []
    for name, spec in SKILL_SPECS.items():
        if spec.ee_type is None or spec.ee_type == ee_type:
            result.append(name)
    return result


# =============================================================================
# Helper functions for Genesis environment
# =============================================================================

def get_gripper_state(env) -> Dict[str, Any]:
    """
    Extract gripper state from Genesis LMPWrapper environment.

    Args:
        env: LMPWrapper instance

    Returns:
        Dict with gripper_open, held_object, position, etc.
    """
    gripper_open = env.env.scene_objects["gripper"].gripper_open
    pointing_to = env.env.scene_objects["gripper"].pointing_to

    # Get gripper position
    end_effector = env.env.franka.get_link(env.ee_name)
    position = end_effector.get_pos().cpu().numpy()

    # Check if holding object (welded)
    held_object = None
    if env._welded["active"]:
        held_object = env._welded["object"]

    return {
        "gripper_open": gripper_open,
        "pointing_to": pointing_to,
        "position": position,
        "held_object": held_object,
        "ee_type": env.ee_type,
        "grasp_active": env._grasp["active"],
        "grasp_object": env._grasp["object"],
    }


def get_distance_to_object(env, obj_name: str) -> float:
    """
    Compute distance from gripper to object.

    Args:
        env: LMPWrapper instance
        obj_name: Name of target object

    Returns:
        Distance in meters
    """
    if obj_name not in env.env.scene_objects:
        return float('inf')

    gripper_state = get_gripper_state(env)
    gripper_pos = gripper_state["position"]

    # Account for gripper offset
    from skill_code import get_gripper_offset
    pos_offset = get_gripper_offset(env.ee_type, gripper_state["pointing_to"])
    gripper_tip_pos = gripper_pos - pos_offset

    obj_pos = env.get_obj_pos(obj_name)

    return float(np.linalg.norm(gripper_tip_pos - obj_pos))


def check_object_exists(env, obj_name: str) -> bool:
    """Check if an object exists in the scene."""
    return obj_name in env.env.scene_objects


def check_is_handle(obj_name: str) -> bool:
    """Check if object name indicates it's a handle."""
    return "handle" in obj_name.lower()