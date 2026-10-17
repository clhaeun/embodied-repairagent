"""
Subtask success/failure check functions for Pour Liquid task.
"""

import numpy as np


def check_grasp_container_success(env, container_name="beaker"):
    """
    Check if grasp_container_vacuum subtask succeeded.
    Container should be attached to vacuum gripper.

    Args:
        env: LMPWrapper environment instance.
        container_name: Name of the container to grasp.

    Returns:
        True if container is attached to vacuum gripper.
    """

    if hasattr(env, '_welded') and env._welded.get('active'):
        return env._welded.get('object') == container_name


    return env.obj_in_gripper(container_name)


def check_pour_success(env, task, threshold=0.3):
    """
    Check if pour_into_container subtask succeeded.
    A portion of liquid should be in the mug.

    Args:
        env: LMPWrapper environment instance.
        task: Task instance containing scene_objects with liquid.
        threshold: Minimum ratio of particles in cup to consider success.

    Returns:
        True if enough liquid is in the mug.
    """
    liquid = task.scene_objects.get("liquid")
    if liquid is None:
        return False

    particle_positions = liquid.get_particles_pos().cpu().numpy()


    mug_bbox = env.get_obj_bbox("mug")
    mug_center = np.mean(mug_bbox, axis=0)
    mug_radius = (mug_bbox[1][0] - mug_bbox[0][0]) / 2 * 0.8


    particles_in_mug = 0
    for pos in particle_positions:
        dist_to_mug = np.linalg.norm(pos[:2] - mug_center[:2])
        in_mug_xy = dist_to_mug < mug_radius
        in_mug_z = mug_bbox[0][2] < pos[2] < mug_bbox[1][2] + 0.05

        if in_mug_xy and in_mug_z:
            particles_in_mug += 1

    mug_ratio = particles_in_mug / len(particle_positions)
    return mug_ratio > threshold


def check_release_container_success(env, container_name="beaker"):
    """
    Check if release_container_vacuum subtask succeeded.
    Container should be released (not attached to gripper).

    Args:
        env: LMPWrapper environment instance.
        container_name: Name of the container.

    Returns:
        True if container is not attached to vacuum gripper.
    """

    if hasattr(env, '_welded'):
        if env._welded.get('active') and env._welded.get('object') == container_name:
            return False


    return not env.obj_in_gripper(container_name)
