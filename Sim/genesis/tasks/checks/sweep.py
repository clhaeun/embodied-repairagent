"""
Subtask success/failure check functions for Sweep task.
"""

import numpy as np


def check_sweep_success(env, piece_name, target_box_name):
    """
    Check if sweep subtask succeeded (piece is inside target box).

    Args:
        env: LMPWrapper environment instance.
        piece_name: Name of the chess piece.
        target_box_name: Name of the target box ("left_chess_box" or "right_chess_box").

    Returns:
        True if piece bbox is inside target box bbox.
    """
    # Object should not be in gripper
    if env.obj_in_gripper(piece_name) and not env.gripper_is_open():
        return False

    # Get target box bbox with adjusted height
    box_bbox = env.get_obj_bbox(target_box_name)
    box_height = 0.04

    box_min = box_bbox[0].copy()
    box_max = box_bbox[1].copy()
    box_max[2] = box_min[2] + box_height + 0.1
    box_min[2] = box_min[2] - 0.15

    # Check if piece is inside box
    piece_bbox = env.get_obj_bbox(piece_name)
    piece_min, piece_max = piece_bbox[0], piece_bbox[1]

    in_box = (
        np.all(piece_min >= box_min) and
        np.all(piece_max <= box_max)
    )
    return in_box


def check_relocate_success(env, piece_name, target_name="board"):
    """
    Check if relocate subtask succeeded (piece is on the board).

    Args:
        env: LMPWrapper environment instance.
        piece_name: Name of the chess piece.
        target_name: Name of the target location (default: "board").

    Returns:
        True if piece is on the board area.
    """
    # Object should not be in gripper
    if env.obj_in_gripper(piece_name) and not env.gripper_is_open():
        return False

    # Get board bbox
    board_bbox = env.get_obj_bbox(target_name)
    board_min, board_max = board_bbox[0], board_bbox[1]

    # Get piece position
    piece_pos = env.get_obj_pos(piece_name)

    # Check if piece is within board's xy range
    in_board_xy = (
        board_min[0] - 0.02 <= piece_pos[0] <= board_max[0] + 0.02 and
        board_min[1] - 0.02 <= piece_pos[1] <= board_max[1] + 0.02
    )

    # Check if piece is at board level (not in a box)
    piece_bbox = env.get_obj_bbox(piece_name)
    piece_bottom_z = piece_bbox[0][2]
    at_board_level = piece_bottom_z < board_max[2] + 0.05

    return in_board_xy and at_board_level
