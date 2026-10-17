"""
Headers for Genesis code generation.

Defines import statements and function interface for different robot types.
"""

RUN_TASK_INTERFACE = '''
def run_task(env, task, descriptions=None, obs=None):

'''

COMMON_HEADER = '''
import numpy as np
'''

GRIPPER_HEADER = '''
from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_gripper,
    close_gripper,
    pick,
    place,
    rotate_gripper,
    grasp_handle,
    release_handle,
)
'''

SUCTION_HEADER = '''
from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    activate_vacuum,
    deactivate_vacuum,
    attach_vacuum_handle,
    detach_vacuum_handle,
    rotate_gripper,
)
'''

ROBOTIQ85_HEADER = '''
from skill_code import (
    move_gripper_to,
    move_to_position,
    move_parallel,
    open_robotiq85,
    close_robotiq85,
    pick_robotiq85,
    place_robotiq85,
    rotate_gripper,
    grasp_handle_robotiq85,
    release_handle_robotiq85,
)
'''

HEADER = {
    'gripper': COMMON_HEADER + GRIPPER_HEADER,
    'suction': COMMON_HEADER + SUCTION_HEADER,
    'robotiq85': COMMON_HEADER + ROBOTIQ85_HEADER,
}

# Skill function names for each robot type
SKILL_FUNCTIONS = {
    'gripper': [
        'move_gripper_to',
        'move_to_position',
        'move_parallel',
        'open_gripper',
        'close_gripper',
        'pick',
        'place',
        'rotate_gripper',
        'grasp_handle',
        'release_handle',
    ],
    'suction': [
        'move_gripper_to',
        'move_to_position',
        'move_parallel',
        'activate_vacuum',
        'deactivate_vacuum',
        'attach_vacuum_handle',
        'detach_vacuum_handle',
        'rotate_gripper',
    ],
    'robotiq85': [
        'move_gripper_to',
        'move_to_position',
        'move_parallel',
        'open_robotiq85',
        'close_robotiq85',
        'pick_robotiq85',
        'place_robotiq85',
        'rotate_gripper',
        'grasp_handle_robotiq85',
        'release_handle_robotiq85',
    ],
}


def normalize_robot_type(robot_type: str) -> str:
    if robot_type is None:
        return "suction"
    rt = str(robot_type).strip().lower()
    aliases = {
        "robotiq": "robotiq85",
        "robotiq_85": "robotiq85",
        "rq85": "robotiq85",
        "panda": "gripper",   # 필요 시
    }
    return aliases.get(rt, rt)