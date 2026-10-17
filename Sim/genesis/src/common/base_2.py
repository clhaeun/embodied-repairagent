from abc import ABC, abstractmethod
from random import Random

import genesis as gs

from src.common.constants import Z_OFFSET


class BaseTask(ABC):
    task_name = ""
    instruction = ""
    use_weld = True

    def __init__(self, variant, is_subtask=False, offset=(0.5, 0.0)):
        self.variant = variant
        self.is_subtask = is_subtask
        self.offset = offset

        self.scene_objects = {}
        self.rng = Random(variant)

    def __getitem__(self, key):
        """Allow dictionary-style access to task attributes for LLM-generated code compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Task has no attribute '{key}'")

    @abstractmethod
    def setup(self, scene):
        raise NotImplementedError

    @abstractmethod
    def post_setup(self):
        raise NotImplementedError

    def _place_floor(self, scene):
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(
            gs.morphs.Mesh(
                file="./assets/desk.glb",
                pos=(0.5, 0.0, 0.0),
                quat=(-0.707, 0.707, 0, 0),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7)),
        )
        self.scene_objects["floor"] = scene.add_entity(
            gs.morphs.Box(
                lower=(0.11, -0.60, 0.0),
                upper=(0.73, 0.60, Z_OFFSET),
                visualization=False,
                fixed=True,
            ),
        )

    def _ground_objects(self, object_pairs, adjust_xy=False, eps=5e-3):
        def get_pos(obj_name):
            return self.scene_objects[obj_name].get_pos().cpu().numpy()

        def get_aabb(obj_name):
            return self.scene_objects[obj_name].get_AABB().cpu().numpy()

        for bottom_obj, top_obj in object_pairs:
            bottom_obj_aabb = get_aabb(bottom_obj)
            top_obj_aabb = get_aabb(top_obj)

            top_obs_pos = get_pos(top_obj)
            bottom_obs_pos = get_pos(bottom_obj)

            if adjust_xy:
                top_obs_pos[:2] = bottom_obs_pos[:2]

            top_obs_pos[2] += bottom_obj_aabb[1][2] - top_obj_aabb[0][2] + eps
            self.scene_objects[top_obj].set_pos(top_obs_pos)
