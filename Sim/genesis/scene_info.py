"""
Scene information extraction utilities for Genesis environment.

This module provides functions to extract scene state from the Genesis LMPWrapper,
including gripper state and object states.
"""

from typing import Any, Dict, Optional


def get_scene_info(env, target_robot: str = "gripper") -> Dict[str, Any]:
    """
    Get current scene information from Genesis environment.

    Returns a dictionary containing:
    - gripper: gripper state (position, is_open, held_object)
    - objects: list of object states
    - robot_type: type of end-effector being used

    This information is used for:
    1. Background validation of skill preconditions
    2. Repair request context for LLM
    3. State tracking for projected state validation

    Args:
        env: LMPWrapper environment instance
        target_robot: Target robot type (for fallback)

    Returns:
        Dictionary with scene information
    """
    try:
        scene_info: Dict[str, Any] = {}




        gripper_info = _extract_gripper_info(env)
        scene_info["gripper"] = gripper_info




        scene_info["robot_type"] = getattr(env, 'ee_type', target_robot)




        gripper_pos = gripper_info.get("position")
        objects_info = _extract_objects_info(env, gripper_info.get("held_object"), gripper_pos)
        scene_info["objects"] = objects_info

        return scene_info

    except Exception as e:
        print(f"[WARN] get_scene_info failed: {e}")
        return {"objects": [], "gripper": {}}


def _extract_gripper_info(env) -> Dict[str, Any]:
    """
    Extract gripper state from environment.

    Args:
        env: LMPWrapper environment instance

    Returns:
        Dictionary with gripper information
    """
    gripper_info: Dict[str, Any] = {
        "position": None,
        "quaternion": None,
        "is_open": True,
        "held_object": None,
    }

    try:

        if "gripper" in env.env.scene_objects:
            gripper_obj = env.env.scene_objects["gripper"]


            gripper_info["position"] = _get_gripper_position(env, gripper_obj)


            gripper_info["quaternion"] = _get_gripper_quaternion(gripper_obj)


            gripper_info["is_open"] = _get_gripper_open_state(env, gripper_obj)


        gripper_info["held_object"] = _get_held_object(env)

    except Exception as e:
        print(f"[WARN] Failed to get gripper info: {e}")

    return gripper_info


def _get_gripper_position(env, gripper_obj) -> Optional[list]:
    """Get gripper position as list."""
    try:

        ee_link = gripper_obj.get_link(env.ee_name) if hasattr(gripper_obj, 'get_link') else None
        if ee_link is not None:
            pos = ee_link.get_pos()
            return pos.cpu().numpy().tolist() if hasattr(pos, 'cpu') else pos.tolist()
        else:

            pos = gripper_obj.get_pos() if hasattr(gripper_obj, 'get_pos') else None
            if pos is not None:
                return pos.cpu().numpy().tolist() if hasattr(pos, 'cpu') else pos.tolist()
    except Exception:
        pass
    return None


def _get_gripper_quaternion(gripper_obj) -> Optional[list]:
    """Get gripper quaternion as list."""
    try:
        quat = gripper_obj.get_quat() if hasattr(gripper_obj, 'get_quat') else None
        if quat is not None:
            return quat.cpu().numpy().tolist() if hasattr(quat, 'cpu') else quat.tolist()
    except Exception:
        pass
    return None


def _get_gripper_open_state(env, gripper_obj) -> bool:
    """Get whether gripper is open."""
    try:
        if hasattr(gripper_obj, 'gripper_open'):
            return bool(gripper_obj.gripper_open)
        elif hasattr(env, 'gripper_is_open'):
            return bool(env.gripper_is_open())
    except Exception:
        pass
    return True


def _get_held_object(env) -> Optional[str]:
    """Get the name of the held object, if any."""
    try:
        if hasattr(env, '_grasp') and env._grasp.get("active"):
            return env._grasp.get("object")
        elif hasattr(env, '_welded') and env._welded.get("active"):
            return env._welded.get("object")
    except Exception:
        pass
    return None



SURFACE_OBSERVATION_HEIGHT = 0.60


def _update_visited_surface(
    env,
    obj_name: str,
    gripper_pos: Optional[list],
    bbox: Optional[list],
) -> None:
    """
    Check if gripper is above object and mark as visited if so.

    Surface info is only exposed for objects that the gripper has visited
    (been positioned above within SURFACE_OBSERVATION_HEIGHT).

    Args:
        env: LMPWrapper environment instance (stores _visited_surfaces)
        obj_name: Name of the object
        gripper_pos: Current gripper position [x, y, z]
        bbox: Object bounding box [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """
    if gripper_pos is None or bbox is None:
        return


    if obj_name in env._visited_surfaces:
        return

    try:
        gx, gy, gz = gripper_pos
        bbox_min, bbox_max = bbox[0], bbox[1]


        in_xy_range = (bbox_min[0] <= gx <= bbox_max[0] and
                       bbox_min[1] <= gy <= bbox_max[1])


        obj_top_z = bbox_max[2]
        above_surface = obj_top_z <= gz <= (obj_top_z + SURFACE_OBSERVATION_HEIGHT)

        if in_xy_range and above_surface:
            env._visited_surfaces.add(obj_name)

            obj = env.env.scene_objects.get(obj_name)
            if obj is not None:
                surface_info = getattr(obj, 'surface_info', None)
                if surface_info:
                    edges_info = surface_info.get("edges", {})
                    center_info = surface_info.get("center", {})
                    note = surface_info.get("note", "")
                    print(f"[INFO] Surface property observed for '{obj_name}':")
                    print(f"  - Center: friction={center_info.get('friction', 'N/A')}, slippery={center_info.get('slippery', False)}")
                    print(f"  - Edges: friction={edges_info.get('friction', 'N/A')}, slippery={edges_info.get('slippery', False)}")
                    if note:
                        print(f"  - Note: {note}")
    except (IndexError, TypeError):
        pass


def _extract_objects_info(
    env,
    held_obj_name: Optional[str],
    gripper_pos: Optional[list],
) -> list:
    """
    Extract object states from environment.

    Args:
        env: LMPWrapper environment instance
        held_obj_name: Name of currently held object (if any)
        gripper_pos: Current gripper position [x, y, z]

    Returns:
        List of object state dictionaries
    """

    if not hasattr(env, '_visited_surfaces'):
        env._visited_surfaces = set()

    objects_info = []

    for obj_name in env.env.scene_objects:
        if obj_name == "gripper":
            continue

        obj_info = _extract_single_object_info(env, obj_name, held_obj_name, gripper_pos)
        objects_info.append(obj_info)

    return objects_info


def _extract_single_object_info(
    env,
    obj_name: str,
    held_obj_name: Optional[str],
    gripper_pos: Optional[list],
) -> Dict[str, Any]:
    """
    Extract state for a single object.

    Args:
        env: LMPWrapper environment instance
        obj_name: Name of the object
        held_obj_name: Name of currently held object (if any)
        gripper_pos: Current gripper position [x, y, z]

    Returns:
        Dictionary with object state
    """
    try:
        obj = env.env.scene_objects[obj_name]
        pos = env.get_obj_pos(obj_name)
        bbox = env.get_obj_bbox(obj_name)


        quat = None
        try:
            quat_tensor = obj.get_quat() if hasattr(obj, 'get_quat') else None
            if quat_tensor is not None:
                quat = quat_tensor.cpu().numpy().tolist() if hasattr(quat_tensor, 'cpu') else quat_tensor.tolist()
        except Exception:
            pass


        raw_surface_info = getattr(obj, 'surface_info', None)



        bbox_list = bbox.tolist() if bbox is not None else None
        _update_visited_surface(env, obj_name, gripper_pos, bbox_list)


        surface_info = None
        if obj_name in env._visited_surfaces and raw_surface_info is not None:
            surface_info = raw_surface_info

        return {
            "name": obj_name,
            "position": pos.tolist() if pos is not None else None,
            "quaternion": quat,
            "bbox": bbox_list,
            "visible": env.is_obj_visible(obj_name),
            "is_grasped": obj_name == held_obj_name,
            "grasped_by": "gripper" if obj_name == held_obj_name else None,
            "surface_info": surface_info,
        }
    except Exception:
        return {
            "name": obj_name,
            "position": None,
            "quaternion": None,
            "bbox": None,
            "visible": False,
            "is_grasped": False,
            "grasped_by": None,
            "surface_info": None,
        }
