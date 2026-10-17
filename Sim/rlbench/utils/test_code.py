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

here is the hmm:

```python
def get_object_positions(index: int):
    """
    Retrieve the positions of objects relevant to the TurnOnLight task.
    Objects in the task: ['lamp_base', 'lamp_screw', 'lampshade', 'push_button_target', 'cable', 'bulb_phys', 'target_button_joint', 'light_bulb', 'bulb'
    Args:
        index (int): The variation index determining the target button's color.
    Returns:
        dict: A dictionary containing the positions of the target button, top plate, and joint.
    """
    lamp_base = Shape('lamp_base')
    lamp_screw = Shape('lamp_screw')
    lampshade = Shape('lampshade')
    push_button_target = Shape('push_button_target')
    cable = Shape('cable')
    bulb_phys = Shape('bulb_phys')
    target_button_joint = Joint('target_button_joint')
    light_bulb = Shape('light_bulb')
    bulb = Shape('bulb')

    lamp_base_pos = lamp_base.get_position()
    lamp_screw_pos = lamp_screw.get_position()
    lampshade_pos = lampshade.get_position()
    push_button_target_pos = push_button_target.get_position()
    cable_pos = cable.get_position()
    bulb_phys_pos = bulb_phys.get_position()
    target_button_joint_pos = target_button_joint.get_position()
    light_bulb_pos = light_bulb.get_position()
    bulb_pos = bulb.get_position()

    return {
        'lamp_base': lamp_base_pos,  # Target button for pushing
        'lamp_screw': lamp_screw_pos,  # Top plate of the button
        'lampshade': lampshade_pos,  # Joint position of the button
        'push_button_target': push_button_target_pos,  # Target button for pushing
        'cable': cable_pos,  # Top plate of the button
        'bulb_phys': bulb_phys_pos,  # Joint position of the button
        'target_button_joint': target_button_joint_pos,  # Target button for pushing
        'light_bulb': light_bulb_pos,  # Top plate of the button
        'bulb': bulb_pos,  # Joint position of the button
    }


def run_task(env, task, descriptions=None, obs=None, variation_index: int = 0):
    print("===== Starting TurnOnLight Task =====")
    if env is None or task is None:
        print("Environment or task is not set up correctly.")
        return

dfheifhefisehlf
    # Reset the task to get descriptions and initial observation
    if descriptions is None or obs is None:
        print("Resetting task to get initial descriptions and observations.")
        descriptions, obs = task.reset()
    target_color_name, _ = colors[variation_index]
    print(f"[Task] Objective: Push the {target_color_name} button")

    # Get object positions
    object_positions = get_object_positions(variation_index)
    button_pos = object_positions['target_button_joint']

    # Execute the pick and place actions
    # Adjusted position for picking
    push_pos = button_pos + np.array([0, 0, -0.03])
    obs, reward, done = push(env, task,
                             target_pos=push_pos,
                             approach_distance=0.1,
                             max_steps=100,
                             threshold=0.01,
                             approach_axis='z',
                             timeout=3.0
                             )

    if done:
        print("[Task] Task completed successfully! Reward:", reward)
    else:
        print("[Task] Task not completed yet (done=False).")

# INSTRUCTIO

```