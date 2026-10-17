"""
Grasp Information Module

Provides functionality to:
1. Detect graspable objects in the scene
2. Analyze bounding box dimensions
3. Generate natural language grasp guidance for LLM consumption
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pyrep.objects.shape import Shape
from scipy.spatial.transform import Rotation as R


# Gripper specifications (in meters)
GRIPPER_SPECS = {
    "panda": {"max_opening": 0.08, "finger_width": 0.02},
    "ur5": {"max_opening": 0.085, "finger_width": 0.02},
    "sawyer": {"max_opening": 0.08, "finger_width": 0.02},
    "jaco": {"max_opening": 0.06, "finger_width": 0.015},
}

# Objects that should NOT be considered graspable (containers, fixtures, sensors, etc.)
NON_GRASPABLE_KEYWORDS = [
    'success', 'target', 'boundary', 'sensor', 'detector', 'goal', 'hoop',
    'bin', 'cupboard', 'grill', 'plane', 'joint', 'button', 'base', 'spawn',
    'waypoint', 'visual', 'wrap', 'topPlate', 'dynamic', 'hole'
]

# Known graspable objects by task (explicitly graspable)
GRASPABLE_OBJECTS_BY_TASK = {
    "BasketballInHoop": ["ball"],
    "PickAndLift": ["pick_and_lift_target"],
    "PickUpCup": ["cup1", "cup2"],
    "PutRubbishInBin": ["rubbish"],
    "PutGroceriesInCupboard": ["chocolate_jello", "strawberry_jello", "soup", "tuna", "spam", "sugar", "coffee", "mustard", "crackers"],
    "PutAllGroceriesInCupboard": ["chocolate_jello", "strawberry_jello", "soup", "tuna", "spam", "sugar", "coffee", "mustard", "crackers"],
    "MeatOnGrill": ["chicken", "steak"],
    "MeatOffGrill": ["chicken", "steak"],
    "UnplugCharger": ["charger"],
    "TakeUsbOutOfComputer": ["usb"],
    "WipeDesk": ["sponge"],
    "Hockey": ["hockey_stick"],
    "SlideBlockToTarget": ["block"],
    "LiftNumberedBlock": ["block1", "block2", "block3"],
    "StackBlocks": ["stack_blocks_target0", "stack_blocks_target1", "stack_blocks_target2"],
    "PlayJenga": ["target_cuboid"],
    "PhoneOnBase": ["phone"],
    "PutMoneyInSafe": ["dollar_stack"],
    "ChangeChannel": ["tv_remote"],
    "PutUmbrellaInUmbrellaStand": ["umbrella"],
    "ReachAndDrag": ["stick"],
    "CloseBox": ["box_lid"],
    "HitBallWithQueue": ["pool_cue"],
}


def is_graspable_object(obj_name: str, task_name: str = "") -> bool:
    """
    Determine if an object is graspable based on its name and task context.

    Args:
        obj_name: Name of the object
        task_name: Name of the current task (optional)

    Returns:
        True if the object is considered graspable
    """
    # Check if explicitly listed as graspable for this task
    if task_name and task_name in GRASPABLE_OBJECTS_BY_TASK:
        if obj_name in GRASPABLE_OBJECTS_BY_TASK[task_name]:
            return True

    # Check against non-graspable keywords
    obj_name_lower = obj_name.lower()
    for keyword in NON_GRASPABLE_KEYWORDS:
        if keyword in obj_name_lower:
            return False

    return True


def get_bbox_info(obj: Shape) -> Dict[str, Any]:
    """
    Get bounding box information for an object.

    Args:
        obj: PyRep Shape object

    Returns:
        Dictionary with bbox dimensions and analysis
    """
    bbox = obj.get_bounding_box()  # [min_x, max_x, min_y, max_y, min_z, max_z]

    sizes = np.array([
        bbox[1] - bbox[0],  # x size
        bbox[3] - bbox[2],  # y size
        bbox[5] - bbox[4],  # z size
    ])

    # Identify axes by size
    sorted_indices = np.argsort(sizes)
    min_axis_idx = sorted_indices[0]  # narrowest
    mid_axis_idx = sorted_indices[1]  # middle
    max_axis_idx = sorted_indices[2]  # longest

    axis_names = ['x', 'y', 'z']

    return {
        "sizes": sizes,
        "sizes_mm": (sizes * 1000).astype(int),  # in millimeters for readability
        "min_axis": axis_names[min_axis_idx],
        "min_size": sizes[min_axis_idx],
        "mid_axis": axis_names[mid_axis_idx],
        "mid_size": sizes[mid_axis_idx],
        "max_axis": axis_names[max_axis_idx],
        "max_size": sizes[max_axis_idx],
        "bbox_raw": bbox,
    }


def get_grasp_axis_in_world(obj: Shape, local_axis: str) -> np.ndarray:
    """
    Convert a local axis direction to world frame.

    Args:
        obj: PyRep Shape object
        local_axis: 'x', 'y', or 'z'

    Returns:
        Unit vector in world coordinates
    """
    axis_map = {
        'x': np.array([1, 0, 0]),
        'y': np.array([0, 1, 0]),
        'z': np.array([0, 0, 1]),
    }

    local_vec = axis_map[local_axis]
    obj_quat = obj.get_quaternion()  # [x, y, z, w]
    obj_rot = R.from_quat(obj_quat)

    world_vec = obj_rot.apply(local_vec)
    return world_vec / np.linalg.norm(world_vec)


def world_axis_name(vec: np.ndarray, threshold: float = 0.7) -> str:
    """
    Determine which world axis a vector is most aligned with.

    Args:
        vec: 3D unit vector
        threshold: Minimum dot product to consider "aligned"

    Returns:
        Human-readable axis name (e.g., "+X", "-Z", "diagonal")
    """
    abs_vec = np.abs(vec)
    max_idx = np.argmax(abs_vec)

    if abs_vec[max_idx] < threshold:
        return "diagonal"

    axis_labels = ['X', 'Y', 'Z']
    sign = '+' if vec[max_idx] > 0 else '-'
    return f"{sign}{axis_labels[max_idx]}"


def generate_grasp_guidance(
    obj_name: str,
    bbox_info: Dict[str, Any],
    obj: Shape,
    robot_type: str = "panda"
) -> Dict[str, Any]:
    """
    Generate comprehensive grasp guidance for an object.

    Args:
        obj_name: Name of the object
        bbox_info: Bounding box analysis from get_bbox_info()
        obj: PyRep Shape object
        robot_type: Type of robot gripper

    Returns:
        Dictionary with grasp guidance information
    """
    gripper_spec = GRIPPER_SPECS.get(robot_type.lower(), GRIPPER_SPECS["panda"])
    max_opening = gripper_spec["max_opening"]

    sizes = bbox_info["sizes"]
    min_axis = bbox_info["min_axis"]
    min_size = bbox_info["min_size"]
    max_axis = bbox_info["max_axis"]
    max_size = bbox_info["max_size"]

    # Check if object is graspable by gripper
    can_grasp = min_size < max_opening

    # Get grasp axis in world frame
    grasp_axis_world = get_grasp_axis_in_world(obj, min_axis)
    grasp_axis_name = world_axis_name(grasp_axis_world)

    # Determine approach direction (perpendicular to grasp axis, usually from above)
    # Prefer approaching from +Z (top-down) if possible
    approach_options = []
    for axis in ['x', 'y', 'z']:
        if axis != min_axis:
            approach_vec = get_grasp_axis_in_world(obj, axis)
            approach_name = world_axis_name(approach_vec)
            approach_options.append({
                "local_axis": axis,
                "world_direction": approach_name,
                "vector": approach_vec.tolist(),
            })

    # Prefer top-down approach if available
    best_approach = None
    for opt in approach_options:
        if '+Z' in opt["world_direction"] or '-Z' in opt["world_direction"]:
            best_approach = opt
            break
    if best_approach is None and approach_options:
        best_approach = approach_options[0]

    return {
        "object_name": obj_name,
        "can_grasp": can_grasp,
        "grasp_width_mm": int(min_size * 1000),
        "grasp_axis_local": min_axis,
        "grasp_axis_world": grasp_axis_name,
        "grasp_axis_vector": grasp_axis_world.tolist(),
        "approach_options": approach_options,
        "recommended_approach": best_approach,
        "object_dimensions_mm": {
            "x": int(sizes[0] * 1000),
            "y": int(sizes[1] * 1000),
            "z": int(sizes[2] * 1000),
        },
        "longest_axis": max_axis,
        "longest_dimension_mm": int(max_size * 1000),
    }


def needs_alignment(grasp_info: Dict[str, Any], gripper_max_opening: float = 0.08) -> bool:
    """
    Determine if an object needs gripper alignment before grasping.

    Returns False (no alignment needed) if:
    1. Object is roughly spherical (all dimensions similar)
    2. Object is small enough that any orientation works (all dims < gripper opening)
    3. The two smallest dimensions are similar (can grasp from multiple angles)

    Args:
        grasp_info: Output from generate_grasp_guidance()
        gripper_max_opening: Maximum gripper opening in meters

    Returns:
        True if alignment is needed, False otherwise
    """
    dims = grasp_info["object_dimensions_mm"]
    sizes = np.array([dims['x'], dims['y'], dims['z']]) / 1000.0  # convert to meters

    sorted_sizes = np.sort(sizes)
    min_size = sorted_sizes[0]
    mid_size = sorted_sizes[1]
    max_size = sorted_sizes[2]

    # Case 1: All dimensions fit within gripper - no alignment needed
    if max_size < gripper_max_opening:
        return False

    # Case 2: Roughly spherical/cubic - all dimensions within 20% of each other
    size_ratio = max_size / (min_size + 1e-6)
    if size_ratio < 1.2:
        return False

    # Case 3: Two smallest dimensions are similar and both fit in gripper
    # BUT only if the largest dimension is also manageable (< 2x gripper opening)
    # Otherwise the object is elongated and alignment matters
    min_mid_ratio = mid_size / (min_size + 1e-6)
    if min_mid_ratio < 1.3 and mid_size < gripper_max_opening and max_size < gripper_max_opening * 2:
        return False

    return True


# Tasks that should not receive grasp guidance (they handle grasping differently)
SKIP_GRASP_GUIDANCE_TASKS = [
    "ChangeChannel",
    "PutUmbrellaInUmbrellaStand",
]


def generate_natural_language_guidance(grasp_info: Dict[str, Any], gripper_max_opening: float = 0.08, task_name: str = "", robot_type: str = "ur5") -> Optional[str]:
    """
    Generate a concise natural language description of how to grasp an object.
    Provides actionable parameters for the robot-specific align_gripper skill.
    Returns None if no alignment is needed.

    Args:
        grasp_info: Output from generate_grasp_guidance()
        gripper_max_opening: Maximum gripper opening in meters
        task_name: Name of the current task (optional)
        robot_type: Robot type ("ur5", "sawyer", "panda")

    Returns:
        Human-readable grasp guidance string with skill parameters, or None if alignment not needed
    """
    # Skip grasp guidance for specific tasks
    if task_name in SKIP_GRASP_GUIDANCE_TASKS:
        return None

    obj_name = grasp_info["object_name"]

    # Check if alignment is actually needed
    if not needs_alignment(grasp_info, gripper_max_opening):
        return None  # No guidance needed - object can be grasped from any direction

    grasp_axis_world = grasp_info["grasp_axis_world"]
    grasp_axis_local = grasp_info["grasp_axis_local"]
    recommended_approach = grasp_info.get("recommended_approach")

    # Determine approach_direction for align_gripper
    # Always use "down" for top-down approach which is the most common and stable approach
    approach_dir = "down"

    # Determine yaw_mode based on grasp axis vs object's local frame
    yaw_mode = _get_yaw_mode(grasp_axis_local, grasp_axis_world)

    # Select robot-specific align_gripper function name
    align_fn_map = {
        "ur5": "ur5_align_gripper",
        "sawyer": "sawyer_align_gripper",
        "panda": "align_to_quaternion",
    }
    align_fn = align_fn_map.get(robot_type.lower(), "ur5_align_gripper")

    return (
        f"'{obj_name}': Before alignment, move to above_grasp_pos (grasp_pos + [0,0,0.15]). "
        f"Use {align_fn} with reference_quat=<{obj_name}'s quaternion>, "
        f"approach_direction=\"{approach_dir}\" (default: \"down\"), yaw_mode=\"{yaw_mode}\"."
    )


def _world_axis_to_approach_direction(world_axis: str) -> str:
    """Convert world axis name to ur5_align_gripper approach_direction parameter."""
    mapping = {
        "+X": "right",
        "-X": "left",
        "+Y": "front",
        "-Y": "back",
        "+Z": "down",
        "-Z": "up",
        "diagonal": "down",
    }
    return mapping.get(world_axis, "down")


def _get_yaw_mode(grasp_axis_local: str, grasp_axis_world: str) -> str:
    """
    Determine yaw_mode for ur5_align_gripper.

    - If object's narrowest axis (grasp axis) is along its local X or Y,
      gripper fingers should align with that axis.
    - 'parallel': gripper fingers align with object's longest horizontal extent
    - 'perpendicular': gripper fingers perpendicular to object's longest extent
    """
    # If grasping along local X, fingers should be parallel to object's X
    # If grasping along local Y, fingers should be perpendicular
    if grasp_axis_local == 'x':
        return "parallel"
    elif grasp_axis_local == 'y':
        return "perpendicular"
    else:  # 'z' - object is tall and thin, grasp from side
        return "parallel"


def get_graspable_objects_info(
    objects_info: List[Dict[str, Any]],
    task_name: str = "",
    robot_type: str = "panda"
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Get grasp information for all graspable objects in the scene.
    Only returns alignment guidance for objects that actually need it.

    Args:
        objects_info: List of object info dicts (from get_scene_info)
        task_name: Name of the current task
        robot_type: Type of robot gripper

    Returns:
        Tuple of (list of grasp info dicts, natural language summary string)
    """
    gripper_spec = GRIPPER_SPECS.get(robot_type.lower(), GRIPPER_SPECS["panda"])
    gripper_max_opening = gripper_spec["max_opening"]

    grasp_infos = []
    nl_summaries = []

    for obj_info in objects_info:
        obj_name = obj_info.get("name", "")
        obj_type = obj_info.get("type", "")

        # Only process Shape objects
        if obj_type != "Shape":
            continue

        # Check if graspable
        if not is_graspable_object(obj_name, task_name):
            continue

        try:
            obj = Shape(obj_name)

            # Check if object is explicitly listed as graspable for this task
            is_explicit_graspable = (
                task_name in GRASPABLE_OBJECTS_BY_TASK and
                obj_name in GRASPABLE_OBJECTS_BY_TASK[task_name]
            )

            # For explicitly graspable objects, skip dynamic check entirely
            # (dynamic state can change after object is moved/dropped)
            if not is_explicit_graspable:
                # Check if object is dynamic (movable) - static objects usually aren't graspable
                try:
                    is_dynamic = obj.is_dynamic()
                    if not is_dynamic:
                        continue
                except Exception:
                    # If we can't check dynamic state, skip non-explicit objects
                    continue

            # Get bounding box info
            bbox_info = get_bbox_info(obj)

            # Skip very large objects (likely not graspable)
            if bbox_info["min_size"] > 0.15:  # > 15cm in smallest dimension
                continue

            # Generate grasp guidance
            grasp_info = generate_grasp_guidance(obj_name, bbox_info, obj, robot_type)
            grasp_infos.append(grasp_info)

            # Generate natural language description (only if alignment is needed)
            nl_guidance = generate_natural_language_guidance(grasp_info, gripper_max_opening, task_name, robot_type=robot_type)
            if nl_guidance is not None:
                nl_summaries.append(nl_guidance)

        except Exception as e:
            # Log error for debugging but continue processing other objects
            print(f"[grasp_info] Warning: Could not process object '{obj_name}': {type(e).__name__}: {e}")
            continue

    # Combine all natural language summaries
    if nl_summaries:
        nl_summary = "Objects requiring gripper alignment before grasping:\n" + "\n".join(f"  - {s}" for s in nl_summaries)
    else:
        nl_summary = ""  # Empty string if no objects need alignment

    return grasp_infos, nl_summary


def get_grasp_guidance_for_object(
    obj_name: str,
    task_name: str = "",
    robot_type: str = "panda"
) -> Optional[Dict[str, Any]]:
    """
    Get grasp guidance for a specific object by name.

    Args:
        obj_name: Name of the object
        task_name: Name of the current task
        robot_type: Type of robot gripper

    Returns:
        Grasp guidance dict or None if not graspable
    """
    if not is_graspable_object(obj_name, task_name):
        return None

    try:
        obj = Shape(obj_name)
        bbox_info = get_bbox_info(obj)
        grasp_info = generate_grasp_guidance(obj_name, bbox_info, obj, robot_type)
        gripper_spec = GRIPPER_SPECS.get(robot_type.lower(), GRIPPER_SPECS["panda"])
        grasp_info["natural_language"] = generate_natural_language_guidance(grasp_info, gripper_spec["max_opening"], task_name, robot_type=robot_type)
        return grasp_info
    except Exception:
        return None


def check_gripper_alignment(
    gripper_quat: np.ndarray,
    obj_name: str,
    task_name: str = "",
    robot_type: str = "panda",
    angle_threshold_deg: float = 30.0
) -> Optional[str]:
    """
    Check if the gripper is properly aligned with a graspable object.
    Returns alignment guidance if misaligned, None if aligned or alignment not needed.

    Args:
        gripper_quat: Current gripper quaternion [x, y, z, w]
        obj_name: Name of the object to check alignment with
        task_name: Name of the current task
        robot_type: Type of robot gripper
        angle_threshold_deg: Maximum angle difference to consider "aligned"

    Returns:
        Alignment guidance string if misaligned, None if aligned or not needed
    """
    gripper_spec = GRIPPER_SPECS.get(robot_type.lower(), GRIPPER_SPECS["panda"])
    gripper_max_opening = gripper_spec["max_opening"]

    if not is_graspable_object(obj_name, task_name):
        return None

    try:
        obj = Shape(obj_name)
        bbox_info = get_bbox_info(obj)
        grasp_info = generate_grasp_guidance(obj_name, bbox_info, obj, robot_type)

        # Check if alignment is needed at all
        if not needs_alignment(grasp_info, gripper_max_opening):
            return None

        # Get object's grasp axis in world frame
        obj_grasp_axis = np.array(grasp_info["grasp_axis_vector"])

        # Get gripper's finger axis (Y-axis for most grippers in their local frame)
        gripper_rot = R.from_quat(gripper_quat)
        gripper_finger_axis = gripper_rot.apply(np.array([0, 1, 0]))  # Y-axis

        # Calculate angle between gripper fingers and object's grasp axis
        dot_product = np.abs(np.dot(gripper_finger_axis, obj_grasp_axis))
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)

        # Check if aligned (either parallel or perpendicular to grasp axis is ok)
        # We want the gripper fingers to be perpendicular to the grasp axis
        perpendicular_angle = np.abs(90.0 - angle_deg)

        if perpendicular_angle < angle_threshold_deg:
            return None  # Already aligned

        # Generate alignment guidance
        nl_guidance = generate_natural_language_guidance(grasp_info, gripper_max_opening, robot_type=robot_type)
        if nl_guidance:
            return f"Gripper misaligned with '{obj_name}' (off by {perpendicular_angle:.1f} deg). {nl_guidance}"

        return None

    except Exception:
        return None


def check_gripper_alignment_for_scene(
    gripper_quat: np.ndarray,
    objects_info: List[Dict[str, Any]],
    task_name: str = "",
    robot_type: str = "panda",
    angle_threshold_deg: float = 30.0
) -> Optional[str]:
    """
    Check gripper alignment against all graspable objects in the scene.
    Returns guidance for the first misaligned object that needs alignment.

    Args:
        gripper_quat: Current gripper quaternion [x, y, z, w]
        objects_info: List of object info dicts from scene
        task_name: Name of the current task
        robot_type: Type of robot gripper
        angle_threshold_deg: Maximum angle difference to consider "aligned"

    Returns:
        Alignment guidance string if any object is misaligned, None otherwise
    """
    misaligned_guidances = []

    for obj_info in objects_info:
        obj_name = obj_info.get("name", "")
        obj_type = obj_info.get("type", "")

        if obj_type != "Shape":
            continue

        guidance = check_gripper_alignment(
            gripper_quat, obj_name, task_name, robot_type, angle_threshold_deg
        )
        if guidance:
            misaligned_guidances.append(guidance)

    if misaligned_guidances:
        return "Alignment issues detected:\n" + "\n".join(f"  - {g}" for g in misaligned_guidances)

    return None
