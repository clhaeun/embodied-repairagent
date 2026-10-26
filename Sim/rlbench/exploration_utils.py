"""Compact, task-independent observations from visible scene state."""

import numpy as np
from typing import Any, Dict, Optional


def compute_static_observation(
    task_name: str,
    scene_info: Dict[str, Any],
    failed_stmt_text: str,
) -> Optional[Dict[str, Any]]:
    """Report local failure context without exposing hidden goal markers."""
    gripper = scene_info.get("gripper") or {}
    pos = gripper.get("position")
    if not isinstance(pos, (tuple, list)) or len(pos) < 3:
        return None

    action = (failed_stmt_text or "").lower()
    if any(w in action for w in ("pick", "grasp", "close_gripper", "close_ee")):
        kind = "grasp"
    elif any(w in action for w in ("place", "release", "open_gripper", "open_ee")):
        kind = "release"
    elif any(w in action for w in ("move", "push", "align", "rotate")):
        kind = "motion"
    else:
        kind = "unknown"

    xyz = np.asarray(pos[:3], dtype=float)
    nearby = []
    for obj in scene_info.get("objects", []):
        if not isinstance(obj, dict):
            continue
        name, location = obj.get("name"), obj.get("position")
        if not name or not isinstance(location, (tuple, list)) or len(location) < 3:
            continue
        if obj.get("type") == "ProximitySensor" or any(
            marker in name.lower() for marker in ("success", "goal_sensor", "waypoint", "boundary")
        ):
            continue
        try:
            distance = float(np.linalg.norm(np.asarray(location[:3], dtype=float) - xyz))
        except (TypeError, ValueError):
            continue
        nearby.append({"name": name, "distance_m": round(distance, 3)})
    nearby.sort(key=lambda item: item["distance_m"])
    return {
        "probe": "static_state_check",
        "failed_action": kind,
        "gripper": {
            "position": [round(float(v), 3) for v in xyz],
            "is_open": bool(gripper.get("is_open", True)),
        },
        "nearby_objects": nearby[:5],
    }
