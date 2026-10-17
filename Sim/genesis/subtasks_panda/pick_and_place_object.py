"""
Pick and Place Object Subtask.

Picks up an object and places it at a target location.
Used by pick_place_2 variants.
"""

import numpy as np

from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
)


def find_place_position(env, target_pos, placed_objects, center_region_ratio=0.5, obj_radius=0.025):
    """
    Find a valid place position in the center region, avoiding already placed objects.

    Args:
        env: Environment instance.
        target_pos: Center of the target area (tray position).
        placed_objects: List of object names already placed on the tray.
        center_region_ratio: Ratio of tray size to use as center region (0.5 = 50%).
        obj_radius: Approximate radius of objects for collision avoidance.

    Returns:
        numpy array [x, y, z] of the best place position.
    """
    tray_aabb = env.get_obj_bbox("tray")
    tray_center = (tray_aabb[0] + tray_aabb[1]) / 2
    tray_half_size = (tray_aabb[1] - tray_aabb[0]) / 2

    # Define center region bounds
    center_half = tray_half_size[:2] * center_region_ratio
    center_min = tray_center[:2] - center_half
    center_max = tray_center[:2] + center_half

    # Get positions of already placed objects
    occupied_positions = []
    for obj in placed_objects:
        try:
            pos = env.get_obj_pos(obj)
            occupied_positions.append(pos[:2])
        except:
            continue

    # If no objects placed yet, return center
    if not occupied_positions:
        return np.array([tray_center[0], tray_center[1], target_pos[2]])

    # Search for best position in center region (grid search)
    best_pos = None
    best_min_dist = -1
    grid_steps = 5

    for i in range(grid_steps):
        for j in range(grid_steps):
            x = center_min[0] + (center_max[0] - center_min[0]) * (i + 0.5) / grid_steps
            y = center_min[1] + (center_max[1] - center_min[1]) * (j + 0.5) / grid_steps
            candidate = np.array([x, y])

            # Calculate minimum distance to any occupied position
            min_dist = float('inf')
            for occ_pos in occupied_positions:
                dist = np.linalg.norm(candidate - np.array(occ_pos))
                min_dist = min(min_dist, dist)

            # We want the position with maximum minimum distance (furthest from all objects)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = candidate

    # If best position is too close to existing objects, fall back to center
    if best_min_dist < obj_radius * 2:
        print(f"[PickAndPlace] Warning: tight space, placing near center")
        best_pos = tray_center[:2]

    return np.array([best_pos[0], best_pos[1], target_pos[2]])


def pick_and_place_object(env, obj_name, target_pos, place_height=0.06, spread_offset=0.0,
                          placed_objects=None, use_smart_placement=True):
    """
    Pick up an object and place it at a target position.

    Args:
        env: LMPWrapper environment instance.
        obj_name: Name of the object to pick.
        target_pos: Target position [x, y, z] (e.g., tray position).
        place_height: Height above target to release.
        spread_offset: X-axis offset for spreading multiple objects (legacy).
        placed_objects: List of objects already placed (for smart placement).
        use_smart_placement: If True, find optimal position avoiding other objects.

    Returns:
        True if successful, False otherwise.
    """
    print(f"[PickAndPlace] Processing {obj_name}")

    if placed_objects is None:
        placed_objects = []

    # Check if object is visible
    if not env.is_obj_visible(obj_name):
        print(f"[PickAndPlace] Object {obj_name} not visible, skipping")
        return False

    # Get object position
    obj_pos = env.get_obj_pos(obj_name)
    print(f"[PickAndPlace] {obj_name} position: {obj_pos}")

    # Ensure gripper is open before picking
    open_gripper(env)

    # Move to object and pick
    # Use deeper grip for ball objects to ensure proper grasp
    depth = 0.03 if "ball" in obj_name else 0.01
    print(f"[PickAndPlace] Moving to {obj_name} with depth={depth}")
    move_gripper_to(env, obj_name, pointing_to="down", depth=depth)
    close_gripper(env)

    # Lift up high enough to clear the box walls
    move_parallel(env, "up", 0.20, pointing_to="down")

    # Calculate place position
    if use_smart_placement and placed_objects:
        place_pos = find_place_position(env, target_pos, placed_objects)
        print(f"[PickAndPlace] Smart placement: found position {place_pos[:2]}")
    else:
        place_pos = target_pos.copy()
        place_pos[0] += spread_offset

    place_pos[2] = target_pos[2] + 0.12  # Above target

    # Move above target
    print(f"[PickAndPlace] Moving to place position: {place_pos}")
    move_to_position(env, place_pos, pointing_to="down")

    # Lower towards target
    place_pos[2] = target_pos[2] + place_height
    move_to_position(env, place_pos, pointing_to="down")

    # Release
    print(f"[PickAndPlace] Releasing {obj_name}")
    open_gripper(env)

    # Move up to clear
    move_parallel(env, "up", 0.08, pointing_to="down")

    print(f"[PickAndPlace] Successfully placed {obj_name}")
    return True