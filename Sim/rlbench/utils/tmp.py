from env import setup_environment, shutdown_environment
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push
import argparse
import importlib
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env import setup_environment, shutdown_environment
from rlbench.const import colors


def get_object_positions(index: int = None):
    """
    Retrieves the 3D positions of objects in the environment for task execution
    """

    blocks = Dummy('blocks')
    blocks_pos = blocks.get_position()
    success = ProximitySensor('success')
    success_pos = success.get_position()
    block1 = Shape('block1')
    block1_pos = block1.get_position()
    block3 = Shape('block3')
    block3_pos = block3.get_position()
    block2 = Shape('block2')
    block2_pos = block2.get_position()
    anchor1 = Dummy('anchor1')
    anchor1_pos = anchor1.get_position()
    anchor3 = Dummy('anchor3')
    anchor3_pos = anchor3.get_position()
    anchor2 = Dummy('anchor2')
    anchor2_pos = anchor2.get_position()

    return {
      'blocks': blocks_pos,
      'success': success_pos,
      'block1': block1_pos,
      'block3': block3_pos,
      'block2': block2_pos,
      'anchor1': anchor1_pos,
      'anchor3': anchor3_pos,
      'anchor2': anchor2_pos,
    }


def run_task(env, task, descriptions = None, obs = None, variation_index: int = 1):
    """
    Main function to run the LiftNumberedBlock task with the given environment and task.
    Task Description: Pick up the block with the number 1. Grasp the 1 numbered block and lift. Lift the 1 numbered block.
    """
    # Get dictionary of positions of objects
    positions = get_object_positions(variation_index)

    # Implement the rest of run_task here, following the 4-space indentation rule.