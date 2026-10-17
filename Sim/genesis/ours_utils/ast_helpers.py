"""
Background Validation Utilities for 'ours' mode execution.

This module contains AST helpers and projected state management for
background validation of skill calls in subtask code.
"""

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# AST Helpers for Extracting Skill Calls from Subtask Code
# =============================================================================

# Genesis primitive skill names that can be validated
GENESIS_PRIMITIVE_SKILLS = {
    "move_gripper_to", "move_to_position", "move_parallel", "rotate_gripper",
    "open_gripper", "close_gripper", "pick", "place",
    "grasp_handle", "release_handle",
    "activate_vacuum", "deactivate_vacuum",
    "attach_vacuum_handle", "detach_vacuum_handle",
    "open_robotiq85", "close_robotiq85",
    "pick_robotiq85", "place_robotiq85",
    "grasp_handle_robotiq85", "release_handle_robotiq85",
}


class SubtaskFunctionExtractor(ast.NodeVisitor):
    """Extract the body of a subtask function from parsed AST."""

    def __init__(self, func_name: str):
        self.func_name = func_name
        self.function_body: List[ast.stmt] = []
        self.found = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.func_name:
            self.function_body = node.body
            self.found = True
        self.generic_visit(node)


def extract_skill_call_info(stmt: ast.stmt) -> Tuple[Optional[str], Optional[ast.Call]]:
    """
    Extract skill call information from an AST statement.

    Returns:
        (skill_name, call_node) or (None, None) if not a skill call
    """
    # Case 1: Bare expression statement with function call
    # e.g., move_gripper_to(env, "cube")
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name):
                return call.func.id, call
            elif isinstance(call.func, ast.Attribute):
                return call.func.attr, call

    # Case 2: Assignment with function call
    # e.g., result = move_gripper_to(env, "cube")
    if isinstance(stmt, ast.Assign):
        if isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name):
                return call.func.id, call
            elif isinstance(call.func, ast.Attribute):
                return call.func.attr, call

    return None, None


@dataclass
class SkillCallInfo:
    """Information about a skill call extracted from AST."""
    skill_name: str
    call_node: ast.Call
    stmt: ast.stmt
    lineno: int
    in_conditional: bool = False  # True if inside if/while/try
    in_loop: bool = False  # True if inside for/while
    parent_context: Optional[str] = None  # Description of parent context


class SkillCallExtractor(ast.NodeVisitor):
    """
    Recursively extract all skill calls from AST, including those inside
    compound statements (if, for, while, try, with).
    """

    def __init__(self, src_lines: List[str]):
        self.src_lines = src_lines
        self.skill_calls: List[SkillCallInfo] = []
        self._in_conditional = False
        self._in_loop = False
        self._parent_context: Optional[str] = None

    def _extract_call_info(self, stmt: ast.stmt) -> Optional[Tuple[str, ast.Call]]:
        """Extract skill call from a simple statement."""
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name):
                return call.func.id, call
            elif isinstance(call.func, ast.Attribute):
                return call.func.attr, call
        elif isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if isinstance(call.func, ast.Name):
                return call.func.id, call
            elif isinstance(call.func, ast.Attribute):
                return call.func.attr, call
        return None

    def _visit_body(self, body: List[ast.stmt]) -> None:
        """Visit a list of statements."""
        for stmt in body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr) -> None:
        result = self._extract_call_info(node)
        if result:
            skill_name, call_node = result
            if is_primitive_skill(skill_name):
                self.skill_calls.append(SkillCallInfo(
                    skill_name=skill_name,
                    call_node=call_node,
                    stmt=node,
                    lineno=node.lineno,
                    in_conditional=self._in_conditional,
                    in_loop=self._in_loop,
                    parent_context=self._parent_context,
                ))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        result = self._extract_call_info(node)
        if result:
            skill_name, call_node = result
            if is_primitive_skill(skill_name):
                self.skill_calls.append(SkillCallInfo(
                    skill_name=skill_name,
                    call_node=call_node,
                    stmt=node,
                    lineno=node.lineno,
                    in_conditional=self._in_conditional,
                    in_loop=self._in_loop,
                    parent_context=self._parent_context,
                ))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        old_conditional = self._in_conditional
        old_context = self._parent_context
        self._in_conditional = True
        self._parent_context = f"if (line {node.lineno})"

        self._visit_body(node.body)

        if node.orelse:
            self._parent_context = f"else (line {node.lineno})"
            self._visit_body(node.orelse)

        self._in_conditional = old_conditional
        self._parent_context = old_context

    def visit_For(self, node: ast.For) -> None:
        old_loop = self._in_loop
        old_context = self._parent_context
        self._in_loop = True
        self._parent_context = f"for (line {node.lineno})"

        self._visit_body(node.body)

        if node.orelse:
            self._visit_body(node.orelse)

        self._in_loop = old_loop
        self._parent_context = old_context

    def visit_While(self, node: ast.While) -> None:
        old_loop = self._in_loop
        old_conditional = self._in_conditional
        old_context = self._parent_context
        self._in_loop = True
        self._in_conditional = True
        self._parent_context = f"while (line {node.lineno})"

        self._visit_body(node.body)

        if node.orelse:
            self._visit_body(node.orelse)

        self._in_loop = old_loop
        self._in_conditional = old_conditional
        self._parent_context = old_context

    def visit_Try(self, node: ast.Try) -> None:
        old_conditional = self._in_conditional
        old_context = self._parent_context
        self._in_conditional = True
        self._parent_context = f"try (line {node.lineno})"

        self._visit_body(node.body)

        for handler in node.handlers:
            self._parent_context = f"except (line {handler.lineno})"
            self._visit_body(handler.body)

        if node.orelse:
            self._parent_context = f"else (line {node.lineno})"
            self._visit_body(node.orelse)

        if node.finalbody:
            self._parent_context = f"finally (line {node.lineno})"
            self._visit_body(node.finalbody)

        self._in_conditional = old_conditional
        self._parent_context = old_context

    def visit_With(self, node: ast.With) -> None:
        old_context = self._parent_context
        self._parent_context = f"with (line {node.lineno})"

        self._visit_body(node.body)

        self._parent_context = old_context


def extract_all_skill_calls(
    subtask_code: str,
    subtask_name: str,
) -> Tuple[List[SkillCallInfo], List[str]]:
    """
    Extract ALL skill calls from a subtask function, including those
    inside compound statements (if, for, while, try, with).

    Args:
        subtask_code: Full subtask code string
        subtask_name: Name of the subtask function to extract from

    Returns:
        (list of SkillCallInfo, source_lines)
    """
    try:
        tree = ast.parse(subtask_code)
        extractor = SubtaskFunctionExtractor(subtask_name)
        extractor.visit(tree)

        if not extractor.found:
            return [], []

        src_lines = subtask_code.split('\n')

        # Use SkillCallExtractor to recursively find all skill calls
        skill_extractor = SkillCallExtractor(src_lines)
        for stmt in extractor.function_body:
            skill_extractor.visit(stmt)

        return skill_extractor.skill_calls, src_lines

    except SyntaxError as e:
        print(f"[WARN] Failed to parse subtask code: {e}")
        return [], []


def get_statement_code(stmt: ast.stmt, src_lines: List[str]) -> str:
    """
    Get the source code text for an AST statement.

    Args:
        stmt: AST statement node
        src_lines: List of source code lines

    Returns:
        Source code text for the statement
    """
    try:
        start_line = stmt.lineno - 1  # 0-indexed
        end_line = getattr(stmt, 'end_lineno', stmt.lineno) - 1

        if 0 <= start_line < len(src_lines):
            if start_line == end_line:
                return src_lines[start_line].strip()
            else:
                lines = src_lines[start_line:end_line + 1]
                return '\n'.join(line.rstrip() for line in lines)
    except Exception:
        pass

    return f"<line {stmt.lineno}>"


def extract_subtask_statements(
    subtask_code: str,
    subtask_name: str,
) -> Tuple[List[ast.stmt], List[str]]:
    """
    Extract statements from a subtask function.

    Args:
        subtask_code: Full subtask code string
        subtask_name: Name of the subtask function to extract

    Returns:
        (statements, source_lines)
    """
    try:
        tree = ast.parse(subtask_code)
        extractor = SubtaskFunctionExtractor(subtask_name)
        extractor.visit(tree)

        if not extractor.found:
            return [], []

        src_lines = subtask_code.split('\n')
        return extractor.function_body, src_lines

    except SyntaxError as e:
        print(f"[WARN] Failed to parse subtask code: {e}")
        return [], []


def is_primitive_skill(skill_name: str) -> bool:
    """Check if a skill name is a Genesis primitive skill."""
    return skill_name.lower() in {s.lower() for s in GENESIS_PRIMITIVE_SKILLS}


# =============================================================================
# Projected State Management for Background Validation
# =============================================================================
# These functions track the projected gripper/object state as we iterate through
# future statements. This allows us to validate statement N based on the state
# that would exist AFTER statements 1..N-1 have executed.
# =============================================================================

@dataclass
class ProjectedGripperState:
    """Projected state of the gripper for validation."""
    position: Optional[List[float]] = None
    quaternion: Optional[List[float]] = None
    is_open: bool = True
    held_object: Optional[str] = None


@dataclass
class ProjectedObjectState:
    """Projected state of an object for validation."""
    name: str
    position: Optional[List[float]] = None
    bbox: Optional[List[List[float]]] = None  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    is_grasped: bool = False
    grasped_by: Optional[str] = None
    surface_info: Optional[Dict[str, Any]] = None  # e.g., {"center": {...}, "edges": {...}}
    surface_observed: bool = False  # True after gripper moves above this object


@dataclass
class ProjectedState:
    """
    Projected world state for sequential statement validation.

    This is updated after each statement to reflect the expected state
    after that statement executes, allowing accurate validation of
    subsequent statements.
    """
    gripper: ProjectedGripperState
    objects: Dict[str, ProjectedObjectState]


def create_projected_state(scene_info: Dict[str, Any]) -> ProjectedState:
    """
    Create a ProjectedState from scene_info captured from environment.

    Args:
        scene_info: Scene info dict from _get_scene_info()

    Returns:
        ProjectedState initialized from current scene
    """
    gripper_info = scene_info.get("gripper", {})
    gripper = ProjectedGripperState(
        position=gripper_info.get("position"),
        quaternion=gripper_info.get("quaternion"),
        is_open=gripper_info.get("is_open", True),
        held_object=gripper_info.get("held_object"),
    )

    objects: Dict[str, ProjectedObjectState] = {}
    for obj in scene_info.get("objects", []):
        if isinstance(obj, dict) and "name" in obj:
            objects[obj["name"]] = ProjectedObjectState(
                name=obj["name"],
                position=obj.get("position"),
                bbox=obj.get("bbox"),
                is_grasped=obj.get("is_grasped", False),
                grasped_by=obj.get("grasped_by"),
                surface_info=obj.get("surface_info"),
                surface_observed=False,  # Not yet observed until gripper moves above
            )

    return ProjectedState(gripper=gripper, objects=objects)


def projected_state_to_scene_info(
    projected: ProjectedState,
    original_scene: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert ProjectedState back to scene_info format for validation functions.

    Args:
        projected: Current projected state
        original_scene: Original scene_info (for fields we don't track)

    Returns:
        scene_info dict compatible with validation functions
    """
    scene_info: Dict[str, Any] = {
        "gripper": {
            "position": projected.gripper.position,
            "quaternion": projected.gripper.quaternion,
            "is_open": projected.gripper.is_open,
            "held_object": projected.gripper.held_object,
        },
        "robot_type": original_scene.get("robot_type", "gripper"),
        "objects": [],
    }

    # Rebuild objects list with projected state
    for obj in original_scene.get("objects", []):
        if isinstance(obj, dict) and "name" in obj:
            obj_name = obj["name"]
            proj_obj = projected.objects.get(obj_name)

            # Only include surface_info if it has been observed (gripper moved above object)
            observed_surface_info = None
            if proj_obj and proj_obj.surface_observed and proj_obj.surface_info:
                observed_surface_info = proj_obj.surface_info

            scene_info["objects"].append({
                "name": obj_name,
                "position": proj_obj.position if proj_obj else obj.get("position"),
                "quaternion": obj.get("quaternion"),
                "bbox": proj_obj.bbox if proj_obj else obj.get("bbox"),
                "visible": obj.get("visible", True),
                "is_grasped": proj_obj.is_grasped if proj_obj else obj.get("is_grasped", False),
                "grasped_by": proj_obj.grasped_by if proj_obj else obj.get("grasped_by"),
                "surface_info": observed_surface_info,
            })

    return scene_info


def extract_target_object_from_call(call_node: ast.Call) -> Optional[str]:
    """
    Extract target object name from a skill call's arguments.

    Args:
        call_node: AST Call node

    Returns:
        Object name if found, None otherwise
    """
    # Check positional args (skip first arg which is usually 'env')
    for arg in call_node.args[1:]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            # First string arg is likely the object name
            return arg.value

    # Check keyword args
    for kw in call_node.keywords:
        if kw.arg in ("obj_name", "object_name", "target_name", "name"):
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                return kw.value.value

    return None


def extract_position_from_call(call_node: ast.Call) -> Optional[List[float]]:
    """
    Extract target position from a move skill call's arguments.

    Handles patterns like:
    - move_to_position(env, [x, y, z], ...)
    - move_to_position(env, pos_variable, ...)  # Can't extract, returns None
    - move_to_position(env, target_pos=[x, y, z], ...)

    Args:
        call_node: AST Call node

    Returns:
        Position as [x, y, z] if found and extractable, None otherwise
    """
    def extract_list_value(node: ast.expr) -> Optional[List[float]]:
        """Extract a list of numbers from an AST node."""
        if isinstance(node, ast.List):
            values = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    values.append(float(elt.value))
                elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                    if isinstance(elt.operand, ast.Constant):
                        values.append(-float(elt.operand.value))
                else:
                    return None  # Non-constant element
            return values if len(values) == 3 else None
        elif isinstance(node, ast.Tuple):
            values = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, (int, float)):
                    values.append(float(elt.value))
                elif isinstance(elt, ast.UnaryOp) and isinstance(elt.op, ast.USub):
                    if isinstance(elt.operand, ast.Constant):
                        values.append(-float(elt.operand.value))
                else:
                    return None
            return values if len(values) == 3 else None
        return None

    # Check positional args (skip first arg which is usually 'env')
    for arg in call_node.args[1:]:
        pos = extract_list_value(arg)
        if pos is not None:
            return pos

    # Check keyword args
    for kw in call_node.keywords:
        if kw.arg in ("pos", "position", "target_pos", "target_position"):
            pos = extract_list_value(kw.value)
            if pos is not None:
                return pos

    return None


def update_surface_observation(state: ProjectedState, gripper_pos: List[float]) -> None:
    """
    Update surface observation status based on gripper position.

    When gripper moves above an object that has surface_info, mark that
    surface as observed. This simulates the agent gaining knowledge about
    the surface properties by being in position to observe/sense them.

    Args:
        state: ProjectedState to update (mutated in place)
        gripper_pos: Current gripper position [x, y, z]
    """
    for obj_name, obj_state in state.objects.items():
        # Skip if no surface_info or already observed
        if not obj_state.surface_info or obj_state.surface_observed:
            continue

        # Check if gripper is above this object
        if obj_state.bbox is not None:
            bbox_min, bbox_max = obj_state.bbox[0], obj_state.bbox[1]

            # Gripper is "above" if x,y within bbox and z higher than object top
            x, y, z = gripper_pos
            if (bbox_min[0] <= x <= bbox_max[0] and
                bbox_min[1] <= y <= bbox_max[1] and
                z >= bbox_max[2]):
                obj_state.surface_observed = True
                # Log surface observation with readable info
                surface_info = obj_state.surface_info
                edges_info = surface_info.get("edges", {})
                center_info = surface_info.get("center", {})
                note = surface_info.get("note", "")
                print(f"[INFO] Surface property observed for '{obj_name}':")
                print(f"  - Center: friction={center_info.get('friction', 'N/A')}, slippery={center_info.get('slippery', False)}")
                print(f"  - Edges: friction={edges_info.get('friction', 'N/A')}, slippery={edges_info.get('slippery', False)}")
                if note:
                    print(f"  - Note: {note}")


def apply_skill_effect(
    state: ProjectedState,
    skill_name: str,
    call_node: ast.Call,
) -> None:
    """
    Apply the symbolic effect of a skill to the projected state.

    This updates the projected state to reflect what would happen AFTER
    the skill executes, allowing accurate validation of subsequent skills.

    Args:
        state: ProjectedState to modify (mutated in place)
        skill_name: Name of the skill
        call_node: AST Call node for extracting arguments
    """
    skill_lower = skill_name.lower()
    target_obj = extract_target_object_from_call(call_node)

    # =========================================================================
    # Grasp/Pick Skills: gripper closes, object becomes held
    # =========================================================================
    grasp_skills = {
        "pick", "pick_robotiq85", "grasp_handle", "grasp_handle_robotiq85",
        "close_gripper", "close_robotiq85", "activate_vacuum",
    }
    if skill_lower in grasp_skills:
        state.gripper.is_open = False
        if target_obj:
            state.gripper.held_object = target_obj
            if target_obj in state.objects:
                state.objects[target_obj].is_grasped = True
                state.objects[target_obj].grasped_by = "gripper"

    # =========================================================================
    # Release/Place Skills: gripper opens, object is released
    # =========================================================================
    release_skills = {
        "place", "place_robotiq85", "release_handle", "release_handle_robotiq85",
        "open_gripper", "open_robotiq85", "deactivate_vacuum",
    }
    if skill_lower in release_skills:
        released_obj = state.gripper.held_object
        state.gripper.is_open = True
        state.gripper.held_object = None

        if released_obj and released_obj in state.objects:
            state.objects[released_obj].is_grasped = False
            state.objects[released_obj].grasped_by = None

    # =========================================================================
    # Move Skills: gripper position changes (and held object moves with it)
    # When gripper moves above an object, its surface becomes observable
    # =========================================================================
    move_skills = {"move_to_position", "move_gripper_to"}
    if skill_lower in move_skills:
        target_pos = extract_position_from_call(call_node)
        if target_pos:
            state.gripper.position = target_pos
            # Check if gripper is now above any object with surface_info
            update_surface_observation(state, target_pos)

    # =========================================================================
    # Vacuum-specific skills
    # =========================================================================
    if skill_lower == "attach_vacuum_handle":
        state.gripper.is_open = False
        if target_obj:
            state.gripper.held_object = target_obj
            if target_obj in state.objects:
                state.objects[target_obj].is_grasped = True
                state.objects[target_obj].grasped_by = "gripper"

    if skill_lower == "detach_vacuum_handle":
        released_obj = state.gripper.held_object
        state.gripper.held_object = None
        if released_obj and released_obj in state.objects:
            state.objects[released_obj].is_grasped = False
            state.objects[released_obj].grasped_by = None
