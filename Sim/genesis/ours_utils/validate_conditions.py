"""
Validation module for primitive skill preconditions and postconditions in Genesis.

This module provides:
1. Utilities to check if conditions can actually be validated at runtime
2. Enhanced condition checking with proper validation logic
3. Projected state tracking for symbolic validation
4. Support for all Genesis end-effector types (gripper, suction, robotiq85)

Adapted from RLBench version to work with Genesis LMPWrapper API.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field

from ours_utils.skill_conditions import (
    ConditionSpec, EffectSpec, SkillSpec, SKILL_SPECS, get_skill_spec,
    # Condition specs
    GripperOpen, GripperClosed, HoldingObject, NotHolding,
    NearTarget, AtTarget, TargetReachable, NotAtTarget,
    ObjectExists, ObjectVisible, ObjectGrasped, ObjectReleased, ObjectAtTarget,
    IsHandle, IsSuctionGripper, IsParallelGripper, IsRobotiq85Gripper,
    # Effect specs
    MoveToTarget, MoveToObject, MoveParallel, RotateWrist,
    CloseGripper, OpenGripper, GraspObject, ReleaseObject,
    GraspHandle, ReleaseHandle, ActivateVacuum, DeactivateVacuum,
    # Helpers
    get_gripper_state, get_distance_to_object, check_object_exists, check_is_handle,
)


@dataclass
class ValidationResult:
    """Result of a condition validation."""
    is_valid: bool
    can_be_validated: bool  # Whether this condition can actually be checked
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Helper functions for extracting robot state from Genesis
# =============================================================================

def get_gripper_openness(env) -> Tuple[bool, bool]:
    """
    Check if gripper is open from Genesis environment.

    Returns:
        (is_open, is_available): gripper open state and whether it was available
    """
    if env is None:
        return True, False

    try:
        gripper_open = env.env.scene_objects["gripper"].gripper_open
        return gripper_open, True
    except Exception:
        return True, False


def get_gripper_position(env) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract gripper position from Genesis environment.

    Returns:
        (position, is_available): 3D position and whether it was available
    """
    if env is None:
        return None, False

    try:
        end_effector = env.env.franka.get_link(env.ee_name)
        position = end_effector.get_pos().cpu().numpy()
        return position, True
    except Exception:
        return None, False


def get_gripper_tip_position(env) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract gripper tip position (accounting for gripper offset).

    Returns:
        (position, is_available): 3D tip position and whether it was available
    """
    if env is None:
        return None, False

    try:
        from skill_code import get_gripper_offset

        end_effector = env.env.franka.get_link(env.ee_name)
        position = end_effector.get_pos().cpu().numpy()
        pointing_to = env.env.scene_objects["gripper"].pointing_to
        offset = get_gripper_offset(env.ee_type, pointing_to)
        tip_pos = position - offset
        return tip_pos, True
    except Exception:
        return None, False


def get_held_object(env) -> Tuple[Optional[str], bool]:
    """
    Get the name of the object currently held by the gripper.

    Returns:
        (object_name, is_available): held object name and whether check was available
    """
    if env is None:
        return None, False

    try:
        if env._welded["active"]:
            return env._welded["object"], True
        return None, True
    except Exception:
        return None, False


def get_ee_type(env) -> str:
    """Get the end-effector type."""
    if env is None:
        return "gripper"
    return getattr(env, "ee_type", "gripper")


def compute_distance(pos1: Optional[np.ndarray], pos2: Optional[np.ndarray]) -> Tuple[float, bool]:
    """
    Compute distance between two positions.

    Returns:
        (distance, is_valid): distance and whether computation was valid
    """
    if pos1 is None or pos2 is None:
        return float('inf'), False

    try:
        return float(np.linalg.norm(np.array(pos1) - np.array(pos2))), True
    except Exception:
        return float('inf'), False


# =============================================================================
# Condition Context for Validation
# =============================================================================

@dataclass
class ConditionContext:
    """Context for evaluating conditions against projected state."""
    gripper_open: bool
    gripper_position: Optional[np.ndarray]
    held_object: Optional[str]
    target_pos: Optional[np.ndarray]
    target_obj: Optional[str]
    objects: Dict[str, Any]  # Object states
    ee_type: str = "gripper"
    grasp_active: bool = False
    flags: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> "ConditionContext":
        """Create context from Genesis LMPWrapper environment."""
        gripper_open, _ = get_gripper_openness(env)
        gripper_pos, _ = get_gripper_tip_position(env)
        held_object, _ = get_held_object(env)
        ee_type = get_ee_type(env)

        # Get object states
        objects = {}
        if env is not None:
            for obj_name in env.env.scene_objects:
                if obj_name == "gripper":
                    continue
                try:
                    pos = env.get_obj_pos(obj_name)
                    objects[obj_name] = {"position": pos, "exists": True}
                except Exception:
                    objects[obj_name] = {"position": None, "exists": True}

        grasp_active = False
        if env is not None:
            grasp_active = env._grasp.get("active", False)

        return cls(
            gripper_open=gripper_open,
            gripper_position=gripper_pos,
            held_object=held_object,
            target_pos=target_pos,
            target_obj=target_obj,
            objects=objects,
            ee_type=ee_type,
            grasp_active=grasp_active,
        )

    @classmethod
    def from_projected_state(
        cls,
        projected_state: "ProjectedState",
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> "ConditionContext":
        """Create context from a ProjectedState."""
        return cls(
            gripper_open=projected_state.gripper.is_open,
            gripper_position=projected_state.gripper.position,
            held_object=projected_state.gripper.held_object,
            target_pos=target_pos,
            target_obj=target_obj,
            objects={k: {"position": v.position, "exists": True}
                     for k, v in projected_state.objects.items()},
            ee_type=projected_state.ee_type,
            grasp_active=projected_state.grasp_active,
            flags=getattr(projected_state, 'flags', {}),
        )


# =============================================================================
# Spec-Driven Condition Validators
# =============================================================================

def validate_condition(
    condition: ConditionSpec,
    ctx: ConditionContext,
    skill_name: str = "",
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate a single condition against context.

    Args:
        condition: The ConditionSpec to validate
        ctx: Context containing current state
        skill_name: Name of skill (for error messages)

    Returns:
        (success, violation_message, warning_message)
        - success: True if condition is met
        - violation_message: Non-None if hard failure
        - warning_message: Non-None if soft warning
    """
    # Gripper state conditions
    if isinstance(condition, GripperOpen):
        if not ctx.gripper_open:
            return False, (
                f"{skill_name}: gripper must be open but is closed "
                f"(holding: {ctx.held_object})"
            ), None
        return True, None, None

    elif isinstance(condition, GripperClosed):
        if ctx.gripper_open:
            return False, (
                f"{skill_name}: gripper must be closed but is open"
            ), None
        return True, None, None

    elif isinstance(condition, HoldingObject):
        if ctx.held_object is None:
            return False, (
                f"{skill_name}: must be holding an object but gripper is empty"
            ), None
        return True, None, None

    elif isinstance(condition, NotHolding):
        if ctx.held_object is not None:
            return False, (
                f"{skill_name}: must not be holding anything but holding '{ctx.held_object}'"
            ), None
        return True, None, None

    # Distance/position conditions
    elif isinstance(condition, NearTarget):
        if ctx.target_pos is None and ctx.target_obj is None:
            return True, None, f"{skill_name}: no target specified for NearTarget check"
        if ctx.gripper_position is not None:
            target = ctx.target_pos
            if target is None and ctx.target_obj and ctx.target_obj in ctx.objects:
                target = ctx.objects[ctx.target_obj].get("position")
            if target is not None:
                dist = float(np.linalg.norm(target - ctx.gripper_position))
                if dist > condition.max_distance:
                    return False, (
                        f"{skill_name}: gripper too far from target "
                        f"({dist:.3f}m > {condition.max_distance}m)"
                    ), None
        return True, None, None

    elif isinstance(condition, AtTarget):
        if ctx.target_pos is None and ctx.target_obj is None:
            return True, None, f"{skill_name}: no target for AtTarget check"
        if ctx.gripper_position is not None:
            target = ctx.target_pos
            if target is None and ctx.target_obj and ctx.target_obj in ctx.objects:
                target = ctx.objects[ctx.target_obj].get("position")
            if target is not None:
                dist = float(np.linalg.norm(target - ctx.gripper_position))
                if dist > condition.tolerance:
                    return False, (
                        f"{skill_name}: not at target "
                        f"({dist:.3f}m > {condition.tolerance}m tolerance)"
                    ), None
        return True, None, None

    elif isinstance(condition, TargetReachable):
        if ctx.target_pos is None and ctx.target_obj is None:
            return True, None, f"{skill_name}: no target for reachability check"
        if ctx.gripper_position is not None:
            target = ctx.target_pos
            if target is None and ctx.target_obj and ctx.target_obj in ctx.objects:
                target = ctx.objects[ctx.target_obj].get("position")
            if target is not None:
                dist = float(np.linalg.norm(target - ctx.gripper_position))
                if dist > condition.max_distance:
                    return True, None, (
                        f"{skill_name}: target may be far "
                        f"({dist:.3f}m > {condition.max_distance}m threshold)"
                    )
        return True, None, None

    elif isinstance(condition, NotAtTarget):
        if ctx.target_pos is None and ctx.target_obj is None:
            return True, None, None
        if ctx.gripper_position is not None:
            target = ctx.target_pos
            if target is None and ctx.target_obj and ctx.target_obj in ctx.objects:
                target = ctx.objects[ctx.target_obj].get("position")
            if target is not None:
                dist = float(np.linalg.norm(target - ctx.gripper_position))
                if dist < condition.min_distance:
                    return False, (
                        f"{skill_name}: already at target "
                        f"(distance {dist:.4f}m < {condition.min_distance}m)"
                    ), None
        return True, None, None

    # Object conditions
    elif isinstance(condition, ObjectExists):
        if ctx.target_obj is None:
            return True, None, f"{skill_name}: no target object specified"
        if ctx.target_obj not in ctx.objects:
            return False, (
                f"{skill_name}: object '{ctx.target_obj}' does not exist in scene"
            ), None
        return True, None, None

    elif isinstance(condition, ObjectVisible):
        if ctx.target_obj is None:
            return True, None, None
        if ctx.target_obj not in ctx.objects:
            return False, (
                f"{skill_name}: object '{ctx.target_obj}' is not visible"
            ), None
        return True, None, None

    elif isinstance(condition, ObjectGrasped):
        if ctx.held_object is None:
            return False, (
                f"{skill_name}: object should be grasped but gripper is empty"
            ), None
        return True, None, None

    elif isinstance(condition, ObjectReleased):
        if ctx.held_object is not None:
            return False, (
                f"{skill_name}: object should be released but still holding '{ctx.held_object}'"
            ), None
        return True, None, None

    elif isinstance(condition, ObjectAtTarget):
        if ctx.target_pos is None or ctx.target_obj is None:
            return True, None, None
        if ctx.target_obj in ctx.objects:
            obj_pos = ctx.objects[ctx.target_obj].get("position")
            if obj_pos is not None:
                dist = float(np.linalg.norm(ctx.target_pos - obj_pos))
                if dist > condition.tolerance:
                    return True, None, (
                        f"{skill_name}: object '{ctx.target_obj}' may not be at target "
                        f"({dist:.3f}m > {condition.tolerance}m)"
                    )
        return True, None, None

    elif isinstance(condition, IsHandle):
        if ctx.target_obj is None:
            return False, f"{skill_name}: no target object specified for handle check", None
        if not check_is_handle(ctx.target_obj):
            return False, (
                f"{skill_name}: '{ctx.target_obj}' is not a handle"
            ), None
        return True, None, None

    # End-effector type conditions
    elif isinstance(condition, IsSuctionGripper):
        if ctx.ee_type != "suction":
            return False, (
                f"{skill_name}: requires suction gripper but using '{ctx.ee_type}'"
            ), None
        return True, None, None

    elif isinstance(condition, IsParallelGripper):
        if ctx.ee_type != "gripper":
            return False, (
                f"{skill_name}: requires parallel gripper but using '{ctx.ee_type}'"
            ), None
        return True, None, None

    elif isinstance(condition, IsRobotiq85Gripper):
        if ctx.ee_type != "robotiq85":
            return False, (
                f"{skill_name}: requires Robotiq85 gripper but using '{ctx.ee_type}'"
            ), None
        return True, None, None

    # Unknown condition type - pass through
    return True, None, f"Unknown condition type: {type(condition).__name__}"


def validate_conditions_list(
    conditions: List[ConditionSpec],
    ctx: ConditionContext,
    skill_name: str = "",
) -> Dict[str, Any]:
    """
    Validate a list of conditions.

    Returns:
        Dict with success, violations, warnings
    """
    violations = []
    warnings = []

    for cond in conditions:
        success, violation, warning = validate_condition(cond, ctx, skill_name)
        if not success and violation:
            violations.append(violation)
        if warning:
            warnings.append(warning)

    return {
        "success": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
    }


# =============================================================================
# Projected State Classes for Symbolic Validation
# =============================================================================

@dataclass
class ObjectState:
    """Symbolic state of an object in the scene."""
    name: str
    position: Optional[np.ndarray] = None
    is_grasped: bool = False
    grasped_by: Optional[str] = None

    def copy(self) -> "ObjectState":
        return ObjectState(
            name=self.name,
            position=self.position.copy() if self.position is not None else None,
            is_grasped=self.is_grasped,
            grasped_by=self.grasped_by,
        )


@dataclass
class GripperState:
    """Symbolic state of the robot gripper."""
    position: Optional[np.ndarray] = None
    is_open: bool = True
    held_object: Optional[str] = None
    pointing_to: str = "down"
    angle: float = 0.0

    def copy(self) -> "GripperState":
        return GripperState(
            position=self.position.copy() if self.position is not None else None,
            is_open=self.is_open,
            held_object=self.held_object,
            pointing_to=self.pointing_to,
            angle=self.angle,
        )


@dataclass
class ProjectedState:
    """
    Projected world state for symbolic validation.
    Composed of confirmed effects (from observations) + predicted effects (symbolic).
    """
    gripper: GripperState
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    ee_type: str = "gripper"
    grasp_active: bool = False
    last_confirmed_index: int = -1
    flags: Dict[str, bool] = field(default_factory=dict)

    def copy(self) -> "ProjectedState":
        return ProjectedState(
            gripper=self.gripper.copy(),
            objects={k: v.copy() for k, v in self.objects.items()},
            ee_type=self.ee_type,
            grasp_active=self.grasp_active,
            last_confirmed_index=self.last_confirmed_index,
            flags=dict(self.flags),
        )


# =============================================================================
# Spec-Driven Effect Appliers
# =============================================================================

def apply_effect(
    effect: EffectSpec,
    state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> None:
    """
    Apply a single effect to projected state (mutates state).

    Args:
        effect: The EffectSpec to apply
        state: ProjectedState to mutate
        target_pos: Target position (if applicable)
        target_obj: Target object name (if applicable)
    """
    if isinstance(effect, MoveToTarget):
        if target_pos is not None:
            state.gripper.position = np.array(target_pos).copy()
            # If holding object, move it too
            if state.gripper.held_object:
                held = state.gripper.held_object
                if held in state.objects:
                    state.objects[held].position = np.array(target_pos).copy()

    elif isinstance(effect, MoveToObject):
        if target_obj and target_obj in state.objects:
            obj_pos = state.objects[target_obj].position
            if obj_pos is not None:
                state.gripper.position = np.array(obj_pos).copy()
        elif target_pos is not None:
            state.gripper.position = np.array(target_pos).copy()
        # If holding object, move it too
        if state.gripper.held_object:
            held = state.gripper.held_object
            if held in state.objects and state.gripper.position is not None:
                state.objects[held].position = state.gripper.position.copy()

    elif isinstance(effect, MoveParallel):
        # Position change handled by parameters, not tracked symbolically
        pass

    elif isinstance(effect, RotateWrist):
        # Rotation doesn't affect symbolic state
        pass

    elif isinstance(effect, CloseGripper):
        state.gripper.is_open = False

    elif isinstance(effect, OpenGripper):
        state.gripper.is_open = True

    elif isinstance(effect, GraspObject):
        state.gripper.is_open = False
        state.gripper.held_object = target_obj
        if target_obj and target_obj in state.objects:
            state.objects[target_obj].is_grasped = True
            state.objects[target_obj].grasped_by = "gripper"

    elif isinstance(effect, ReleaseObject):
        released = state.gripper.held_object
        state.gripper.is_open = True
        state.gripper.held_object = None
        if released and released in state.objects:
            state.objects[released].is_grasped = False
            state.objects[released].grasped_by = None
            # Object stays at release position
            if state.gripper.position is not None:
                state.objects[released].position = state.gripper.position.copy()

    elif isinstance(effect, GraspHandle):
        state.gripper.is_open = False
        state.grasp_active = True

    elif isinstance(effect, ReleaseHandle):
        state.gripper.is_open = True
        state.grasp_active = False

    elif isinstance(effect, ActivateVacuum):
        state.gripper.is_open = False

    elif isinstance(effect, DeactivateVacuum):
        state.gripper.is_open = True


def apply_effects_list(
    effects: List[EffectSpec],
    state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> None:
    """Apply a list of effects to projected state."""
    for effect in effects:
        apply_effect(effect, state, target_pos, target_obj)


# =============================================================================
# Spec-Driven Validation Engine
# =============================================================================

def validate_skill_preconditions(
    skill_name: str,
    env,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate preconditions for a skill against current environment state.

    Args:
        skill_name: Name of the skill
        env: LMPWrapper environment instance
        target_pos: Target position
        target_obj: Target object name

    Returns:
        Dict with success, violations, warnings
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return {
            "success": True,
            "violations": [],
            "warnings": [f"Unknown skill: {skill_name}, no preconditions defined"],
        }

    ctx = ConditionContext.from_env(env, target_pos, target_obj)
    return validate_conditions_list(spec.preconditions, ctx, skill_name)


def validate_skill_preconditions_projected(
    skill_name: str,
    projected_state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate preconditions using projected state (spec-driven).

    Args:
        skill_name: Name of the skill
        projected_state: Current projected state
        target_pos: Target position
        target_obj: Target object name

    Returns:
        Dict with success, violations, warnings
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return {
            "success": True,
            "violations": [],
            "warnings": [f"Unknown skill: {skill_name}, no preconditions defined"],
        }

    ctx = ConditionContext.from_projected_state(projected_state, target_pos, target_obj)
    return validate_conditions_list(spec.preconditions, ctx, skill_name)


def validate_skill_postconditions(
    skill_name: str,
    env,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate postconditions for a skill against current environment state.
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return {
            "success": True,
            "violations": [],
            "warnings": [f"Unknown skill: {skill_name}, no postconditions defined"],
        }

    ctx = ConditionContext.from_env(env, target_pos, target_obj)
    return validate_conditions_list(spec.postconditions, ctx, skill_name)


def validate_skill_postconditions_projected(
    skill_name: str,
    projected_state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validate postconditions using projected state.
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return {
            "success": True,
            "violations": [],
            "warnings": [f"Unknown skill: {skill_name}, no postconditions defined"],
        }

    ctx = ConditionContext.from_projected_state(projected_state, target_pos, target_obj)
    return validate_conditions_list(spec.postconditions, ctx, skill_name)


def apply_skill_effects(
    skill_name: str,
    state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> None:
    """
    Apply skill effects to projected state (spec-driven).
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return

    apply_effects_list(spec.effects, state, target_pos, target_obj)


def validate_skill_full(
    skill_name: str,
    projected_state: ProjectedState,
    target_pos: Optional[np.ndarray] = None,
    target_obj: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full validation sequence:
    1. Check preconditions
    2. Apply effects (assume success)
    3. Check postconditions

    Returns combined result.
    """
    spec = get_skill_spec(skill_name)
    if spec is None:
        return {
            "success": True,
            "precondition_success": True,
            "postcondition_success": True,
            "violations": [],
            "precondition_violations": [],
            "postcondition_violations": [],
            "warnings": [f"Unknown skill: {skill_name}"],
            "transition": None,
        }

    # Step 1: Check preconditions
    ctx_pre = ConditionContext.from_projected_state(projected_state, target_pos, target_obj)
    pre_result = validate_conditions_list(spec.preconditions, ctx_pre, skill_name)

    # Step 2: Apply effects (mutates projected_state)
    apply_effects_list(spec.effects, projected_state, target_pos, target_obj)

    # Step 3: Check postconditions
    ctx_post = ConditionContext.from_projected_state(projected_state, target_pos, target_obj)
    post_result = validate_conditions_list(spec.postconditions, ctx_post, skill_name)

    # Combine results
    all_violations = pre_result["violations"] + post_result["violations"]
    all_warnings = pre_result["warnings"] + post_result["warnings"]

    return {
        "success": len(all_violations) == 0,
        "precondition_success": pre_result["success"],
        "postcondition_success": post_result["success"],
        "violations": all_violations,
        "precondition_violations": pre_result["violations"],
        "postcondition_violations": post_result["violations"],
        "warnings": all_warnings,
        "transition": spec.transition,
        "projected_gripper_pos": projected_state.gripper.position.tolist() if projected_state.gripper.position is not None else None,
        "projected_held_object": projected_state.gripper.held_object,
        "projected_gripper_open": projected_state.gripper.is_open,
    }


# =============================================================================
# Projected State Tracker
# =============================================================================

class ProjectedStateTracker:
    """
    Tracks and projects world state for validation of future statements.

    Usage:
        tracker = ProjectedStateTracker()
        tracker.initialize_from_env(env)

        # After each executed statement
        tracker.confirm_state(env, statement_index)

        # To validate future statements
        for stmt in future_statements:
            projected = tracker.project_effect(skill_name, target_pos, target_obj)
            validation = tracker.validate_preconditions(skill_name, projected)
    """

    def __init__(self):
        self.confirmed_state: Optional[ProjectedState] = None
        self.projected_state: Optional[ProjectedState] = None
        self.execution_history: List[Dict[str, Any]] = []

    def initialize_from_env(self, env) -> ProjectedState:
        """
        Initialize state from Genesis LMPWrapper environment.

        Args:
            env: LMPWrapper instance
        """
        # Get gripper state
        gripper_pos, _ = get_gripper_tip_position(env)
        gripper_open, _ = get_gripper_openness(env)
        held_object, _ = get_held_object(env)
        ee_type = get_ee_type(env)

        pointing_to = "down"
        angle = 0.0
        try:
            pointing_to = env.env.scene_objects["gripper"].pointing_to
            angle = env.gripper_state.get("angle", 0.0)
        except Exception:
            pass

        gripper = GripperState(
            position=gripper_pos,
            is_open=gripper_open,
            held_object=held_object,
            pointing_to=pointing_to,
            angle=angle,
        )

        # Get object states
        objects = {}
        if env is not None:
            for obj_name in env.env.scene_objects:
                if obj_name == "gripper":
                    continue
                try:
                    pos = env.get_obj_pos(obj_name)
                    is_grasped = obj_name == held_object
                    objects[obj_name] = ObjectState(
                        name=obj_name,
                        position=pos,
                        is_grasped=is_grasped,
                        grasped_by="gripper" if is_grasped else None,
                    )
                except Exception:
                    objects[obj_name] = ObjectState(name=obj_name)

        grasp_active = False
        try:
            grasp_active = env._grasp.get("active", False)
        except Exception:
            pass

        self.confirmed_state = ProjectedState(
            gripper=gripper,
            objects=objects,
            ee_type=ee_type,
            grasp_active=grasp_active,
            last_confirmed_index=-1,
        )
        self.projected_state = self.confirmed_state.copy()

        return self.confirmed_state

    def confirm_state(
        self,
        env,
        statement_index: int,
        skill_name: Optional[str] = None,
        target_obj: Optional[str] = None,
    ) -> None:
        """
        Confirm state after a statement has been executed.
        Updates confirmed_state from actual environment.
        """
        if self.confirmed_state is None:
            return

        # Update gripper state
        gripper_pos, pos_avail = get_gripper_tip_position(env)
        gripper_open, _ = get_gripper_openness(env)
        held_object, _ = get_held_object(env)

        if pos_avail:
            self.confirmed_state.gripper.position = gripper_pos
        self.confirmed_state.gripper.is_open = gripper_open
        self.confirmed_state.gripper.held_object = held_object

        try:
            self.confirmed_state.gripper.pointing_to = env.env.scene_objects["gripper"].pointing_to
            self.confirmed_state.gripper.angle = env.gripper_state.get("angle", 0.0)
            self.confirmed_state.grasp_active = env._grasp.get("active", False)
        except Exception:
            pass

        # Update object states
        for obj_name in self.confirmed_state.objects:
            try:
                pos = env.get_obj_pos(obj_name)
                self.confirmed_state.objects[obj_name].position = pos
                is_grasped = obj_name == held_object
                self.confirmed_state.objects[obj_name].is_grasped = is_grasped
                self.confirmed_state.objects[obj_name].grasped_by = "gripper" if is_grasped else None
            except Exception:
                pass

        self.confirmed_state.last_confirmed_index = statement_index

        # Record in history
        self.execution_history.append({
            "index": statement_index,
            "skill_name": skill_name,
            "gripper_pos": gripper_pos.tolist() if pos_avail else None,
            "gripper_open": gripper_open,
            "held_object": held_object,
        })

        # Reset projected state to confirmed state
        self.projected_state = self.confirmed_state.copy()

    def project_effect(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> ProjectedState:
        """
        Project the effect of executing a skill on current projected state.
        Returns new projected state WITHOUT modifying current state.
        """
        if self.projected_state is None:
            raise ValueError("State not initialized")

        new_state = self.projected_state.copy()
        apply_skill_effects(skill_name, new_state, target_pos, target_obj)
        return new_state

    def apply_projection(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> None:
        """
        Apply skill effect to projected state (mutates self.projected_state).
        """
        self.projected_state = self.project_effect(skill_name, target_pos, target_obj)

    def reset_projection(self) -> None:
        """Reset projected state to last confirmed state."""
        if self.confirmed_state:
            self.projected_state = self.confirmed_state.copy()

    def validate_preconditions_on_projected(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate preconditions for a skill against projected state.
        """
        if self.projected_state is None:
            return {"success": False, "violations": ["State not initialized"], "warnings": []}

        result = validate_skill_preconditions_projected(
            skill_name, self.projected_state, target_pos, target_obj
        )

        # Add projected state info
        gripper = self.projected_state.gripper
        result["projected_gripper_pos"] = gripper.position.tolist() if gripper.position is not None else None
        result["projected_held_object"] = gripper.held_object
        result["projected_gripper_open"] = gripper.is_open

        return result

    def validate_postconditions_on_projected(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate postconditions for a skill against projected state.
        """
        if self.projected_state is None:
            return {"success": False, "violations": ["State not initialized"], "warnings": []}

        result = validate_skill_postconditions_projected(
            skill_name, self.projected_state, target_pos, target_obj
        )

        gripper = self.projected_state.gripper
        result["projected_gripper_pos"] = gripper.position.tolist() if gripper.position is not None else None
        result["projected_held_object"] = gripper.held_object
        result["projected_gripper_open"] = gripper.is_open

        return result

    def validate_statement_full(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full validation sequence for a statement.
        """
        if self.projected_state is None:
            return {
                "success": False,
                "precondition_success": False,
                "postcondition_success": False,
                "violations": ["State not initialized"],
                "precondition_violations": ["State not initialized"],
                "postcondition_violations": [],
                "warnings": [],
            }

        return validate_skill_full(
            skill_name, self.projected_state, target_pos, target_obj
        )


# =============================================================================
# Runtime Condition Validator Wrapper
# =============================================================================

class ConditionValidator:
    """
    Wrapper class for validating conditions during skill execution.

    Example usage:
        validator = ConditionValidator(env)

        # Before executing pick
        pre_result = validator.check_precondition("pick", target_obj="cube")
        if not pre_result["success"]:
            print(f"Precondition failed: {pre_result['violations']}")

        # Execute skill...

        # After executing pick
        post_result = validator.check_postcondition("pick", target_obj="cube")
        if not post_result["success"]:
            print(f"Postcondition failed: {post_result['violations']}")
    """

    def __init__(self, env, strict: bool = False, log_warnings: bool = True):
        """
        Args:
            env: LMPWrapper environment instance
            strict: If True, fail on unvalidatable conditions
            log_warnings: If True, print warnings during validation
        """
        self.env = env
        self.strict = strict
        self.log_warnings = log_warnings
        self.history: List[Dict] = []

    def check_precondition(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check preconditions for a skill."""
        result = validate_skill_preconditions(
            skill_name, self.env, target_pos, target_obj
        )

        if self.log_warnings and result["warnings"]:
            for warn in result["warnings"]:
                print(f"[WARN] {skill_name} precondition: {warn}")

        self.history.append({
            "type": "precondition",
            "skill": skill_name,
            "result": result
        })

        return result

    def check_postcondition(
        self,
        skill_name: str,
        target_pos: Optional[np.ndarray] = None,
        target_obj: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Check postconditions for a skill."""
        result = validate_skill_postconditions(
            skill_name, self.env, target_pos, target_obj
        )

        if self.log_warnings and result["warnings"]:
            for warn in result["warnings"]:
                print(f"[WARN] {skill_name} postcondition: {warn}")

        self.history.append({
            "type": "postcondition",
            "skill": skill_name,
            "result": result
        })

        return result

    def get_history(self) -> List[Dict]:
        """Get validation history."""
        return self.history

    def clear_history(self):
        """Clear validation history."""
        self.history = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from validation history."""
        if not self.history:
            return {"total": 0}

        total = len(self.history)
        passed = sum(1 for h in self.history if h["result"]["success"])
        failed = total - passed

        by_skill = {}
        for h in self.history:
            skill = h["skill"]
            if skill not in by_skill:
                by_skill[skill] = {"passed": 0, "failed": 0}
            if h["result"]["success"]:
                by_skill[skill]["passed"] += 1
            else:
                by_skill[skill]["failed"] += 1

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "by_skill": by_skill
        }


# =============================================================================
# Batch Invalid Statement Info
# =============================================================================

# =============================================================================
# Basic Skill Validation (without full ProjectedStateTracker)
# =============================================================================

def basic_validate_skill_call(
    skill_name: str,
    call_node,  # ast.Call
    scene_info: Dict[str, Any],
) -> List[str]:
    """
    Basic validation of a skill call without full ProjectedStateTracker.

    Performs simple checks like:
    - Object referenced in call exists in scene
    - Gripper state is appropriate for the skill
    - Position values are reasonable

    Args:
        skill_name: Name of the skill being called
        call_node: AST Call node for the skill call
        scene_info: Current scene information (includes gripper and objects)

    Returns:
        List of violation messages (empty if valid)
    """
    import ast
    from ours_utils.ast_helpers import extract_position_from_call

    violations = []
    skill_lower = skill_name.lower()

    # Get list of object names in scene
    scene_objects = set()
    if "objects" in scene_info:
        for obj in scene_info["objects"]:
            if isinstance(obj, dict) and "name" in obj:
                scene_objects.add(obj["name"])

    # Get gripper state for validation
    gripper = scene_info.get("gripper", {})
    gripper_is_open = gripper.get("is_open", True)
    held_object = gripper.get("held_object")

    # =================================================================
    # Gripper State Validation
    # =================================================================

    # Skills that require gripper to be open (grasp/pick skills)
    grasp_skills = {"pick", "pick_robotiq85", "grasp_handle", "grasp_handle_robotiq85", "close_gripper", "close_robotiq85"}
    if skill_lower in grasp_skills:
        if not gripper_is_open:
            violations.append(
                f"{skill_name}: gripper must be open but is currently closed"
            )
        if held_object is not None:
            violations.append(
                f"{skill_name}: gripper must be empty but is holding '{held_object}'"
            )

    # Skills that require gripper to be closed / holding object (place/release skills)
    release_skills = {"place", "place_robotiq85", "release_handle", "release_handle_robotiq85", "open_gripper", "open_robotiq85", "deactivate_vacuum"}
    if skill_lower in release_skills:
        # For place skills, should be holding something
        if skill_lower in {"place", "place_robotiq85"}:
            if gripper_is_open:
                violations.append(
                    f"{skill_name}: gripper must be closed (holding object) but is open"
                )
            if held_object is None:
                violations.append(
                    f"{skill_name}: must be holding an object to place"
                )

    # =================================================================
    # Surface Property Validation (for place skills)
    # =================================================================
    place_skills = {"place", "place_robotiq85", "deactivate_vacuum", "detach_vacuum_handle"}
    if skill_lower in place_skills:
        # Check if placing on a surface with slippery edges
        place_pos = extract_position_from_call(call_node)

        for obj in scene_info.get("objects", []):
            surface_info = obj.get("surface_info")
            if not surface_info:
                continue

            # Check if surface has slippery edges
            edges_info = surface_info.get("edges", {})
            if edges_info.get("slippery", False):
                obj_bbox = obj.get("bbox")
                if obj_bbox and place_pos:
                    # Check if place position is on the edge (not center)
                    bbox_min, bbox_max = obj_bbox[0], obj_bbox[1]
                    center = [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)]
                    half_size = [(bbox_max[i] - bbox_min[i]) / 2 for i in range(3)]

                    # Center region is ~50% of the surface
                    center_threshold = 0.5
                    x, y, z = place_pos
                    in_center_x = abs(x - center[0]) < half_size[0] * center_threshold
                    in_center_y = abs(y - center[1]) < half_size[1] * center_threshold

                    if not (in_center_x and in_center_y):
                        note = surface_info.get("note", "")
                        violations.append(
                            f"{skill_name}: placing at edge of '{obj['name']}' which has slippery edges (friction={edges_info.get('friction', 'low')}). {note}"
                        )

    # =================================================================
    # Object Reference Validation
    # =================================================================

    # Extract string arguments from call (potential object names)
    for arg in call_node.args[1:]:  # Skip first arg (env)
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            obj_ref = arg.value
            # Check if this looks like an object reference
            if obj_ref and not obj_ref.startswith(("_", "#")):
                if scene_objects and obj_ref not in scene_objects:
                    # Only report if we have scene objects and this isn't one
                    violations.append(
                        f"Object '{obj_ref}' referenced in {skill_name} may not exist in scene"
                    )

    # Check keyword arguments
    for kw in call_node.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            obj_ref = kw.value.value
            if obj_ref and not obj_ref.startswith(("_", "#")):
                if "name" in kw.arg.lower() or "obj" in kw.arg.lower():
                    if scene_objects and obj_ref not in scene_objects:
                        violations.append(
                            f"Object '{obj_ref}' in {kw.arg} may not exist in scene"
                        )

    return violations


# =============================================================================
# Batch Invalid Statement Info
# =============================================================================

@dataclass
class InvalidStatementInfo:
    """Information about an invalid future statement."""
    line_number: int
    statement_code: str
    skill_name: str
    violations: List[str]
    warnings: List[str]
    projected_state_summary: Dict[str, Any]


def create_invalid_statements_payload(
    invalid_statements: List[InvalidStatementInfo]
) -> Dict[str, Any]:
    """
    Create JSON payload for batch invalid statement repair request.

    Args:
        invalid_statements: List of InvalidStatementInfo

    Returns:
        Dict ready to be sent to remote_llm
    """
    statements_data = []
    for stmt_info in invalid_statements:
        statements_data.append({
            "line_number": stmt_info.line_number,
            "statement_code": stmt_info.statement_code,
            "skill_name": stmt_info.skill_name,
            "violations": stmt_info.violations,
            "warnings": stmt_info.warnings,
            "projected_state": stmt_info.projected_state_summary,
        })

    return {
        "invalid_statements": statements_data,
        "count": len(statements_data),
    }
