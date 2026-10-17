"""
Genesis ours_utils - Validation utilities for skill conditions.

This module provides validation tools for primitive skills in the Genesis simulator,
adapted from the RLBench ours_utils module.

Main components:
- skill_conditions: Skill specification definitions (preconditions, effects, postconditions)
- validate_conditions: Runtime validation and projected state tracking
"""

from ours_utils.skill_conditions import (
    # Condition specs
    ConditionSpec,
    GripperOpen, GripperClosed, HoldingObject, NotHolding,
    NearTarget, AtTarget, TargetReachable, NotAtTarget,
    ObjectExists, ObjectVisible, ObjectGrasped, ObjectReleased, ObjectAtTarget,
    IsHandle, IsSuctionGripper, IsParallelGripper, IsRobotiq85Gripper,
    # Effect specs
    EffectSpec,
    MoveToTarget, MoveToObject, MoveParallel, RotateWrist,
    CloseGripper, OpenGripper, GraspObject, ReleaseObject,
    GraspHandle, ReleaseHandle, ActivateVacuum, DeactivateVacuum,
    # Skill spec
    SkillSpec,
    SKILL_SPECS,
    get_skill_spec,
    get_skills_for_ee_type,
    # Helpers
    get_gripper_state,
    get_distance_to_object,
    check_object_exists,
    check_is_handle,
)

from ours_utils.ast_helpers import (
    # AST helpers
    GENESIS_PRIMITIVE_SKILLS,
    SkillCallInfo,
    extract_all_skill_calls,
    extract_subtask_statements,
    get_statement_code,
    is_primitive_skill,
    extract_skill_call_info,
    # Projected state management (local simplified version)
    ProjectedGripperState,
    ProjectedObjectState,
    ProjectedState as SimpleProjectedState,
    create_projected_state,
    projected_state_to_scene_info,
    apply_skill_effect,
    extract_position_from_call,
    extract_target_object_from_call,
    update_surface_observation,
)

from ours_utils.validate_conditions import (
    # Validation result
    ValidationResult,
    # Context
    ConditionContext,
    # State classes
    ObjectState,
    GripperState,
    ProjectedState,
    # Validators
    validate_condition,
    validate_conditions_list,
    validate_skill_preconditions,
    validate_skill_preconditions_projected,
    validate_skill_postconditions,
    validate_skill_postconditions_projected,
    apply_skill_effects,
    validate_skill_full,
    # Tracker
    ProjectedStateTracker,
    # Runtime validator
    ConditionValidator,
    # Basic validation
    basic_validate_skill_call,
    # Batch info
    InvalidStatementInfo,
    create_invalid_statements_payload,
)

__all__ = [
    # AST helpers
    "GENESIS_PRIMITIVE_SKILLS",
    "SkillCallInfo",
    "extract_all_skill_calls",
    "extract_subtask_statements",
    "get_statement_code",
    "is_primitive_skill",
    "extract_skill_call_info",
    # Projected state (simplified)
    "ProjectedGripperState",
    "ProjectedObjectState",
    "SimpleProjectedState",
    "create_projected_state",
    "projected_state_to_scene_info",
    "apply_skill_effect",
    "extract_position_from_call",
    "extract_target_object_from_call",
    "update_surface_observation",
    # Condition specs
    "ConditionSpec",
    "GripperOpen", "GripperClosed", "HoldingObject", "NotHolding",
    "NearTarget", "AtTarget", "TargetReachable", "NotAtTarget",
    "ObjectExists", "ObjectVisible", "ObjectGrasped", "ObjectReleased", "ObjectAtTarget",
    "IsHandle", "IsSuctionGripper", "IsParallelGripper", "IsRobotiq85Gripper",
    # Effect specs
    "EffectSpec",
    "MoveToTarget", "MoveToObject", "MoveParallel", "RotateWrist",
    "CloseGripper", "OpenGripper", "GraspObject", "ReleaseObject",
    "GraspHandle", "ReleaseHandle", "ActivateVacuum", "DeactivateVacuum",
    # Skill spec
    "SkillSpec",
    "SKILL_SPECS",
    "get_skill_spec",
    "get_skills_for_ee_type",
    # Helpers
    "get_gripper_state",
    "get_distance_to_object",
    "check_object_exists",
    "check_is_handle",
    # Validation
    "ValidationResult",
    "ConditionContext",
    "ObjectState",
    "GripperState",
    "ProjectedState",
    "validate_condition",
    "validate_conditions_list",
    "validate_skill_preconditions",
    "validate_skill_preconditions_projected",
    "validate_skill_postconditions",
    "validate_skill_postconditions_projected",
    "apply_skill_effects",
    "validate_skill_full",
    "ProjectedStateTracker",
    "ConditionValidator",
    "basic_validate_skill_call",
    "InvalidStatementInfo",
    "create_invalid_statements_payload",
]
