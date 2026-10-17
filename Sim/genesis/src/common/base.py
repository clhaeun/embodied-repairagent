from abc import ABC, abstractmethod

import genesis as gs

from src.common.constants import Z_OFFSET


class BaseTask(ABC):
    task_name = ""
    instruction = ""

    def __init__(
        self,
        variant,
        is_subtask=False,
        offset=(0.5, 0.0),
        obst_level=0,
        multi_level=0,
    ):
        self.variant = variant
        self.is_subtask = is_subtask
        self.offset = offset
        self.obst_level = obst_level
        self.multi_level = multi_level

        self.reflect_y = False
        self.scene_objects = {}


        self.goal_achieve_seq = []
        self.goal_achieve_seq_label = []
        self.goal_achieve_timesteps = []

    def __getitem__(self, key):
        """Allow dictionary-style access to task attributes for LLM-generated code compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Task has no attribute '{key}'")

    @staticmethod
    def _rotate_xy_180(coords, origin):
        if len(coords) not in (2, 3):
            raise ValueError("Coordinates must include x and y components.")

        x, y = coords[0], coords[1]
        ox, oy = origin[0], origin[1]
        rx = 2 * ox - x
        ry = 2 * oy - y

        if len(coords) == 2:
            return (rx, ry)
        return (rx, ry, coords[2])

    def maybe_rotate_xy_180(self, coords, origin=None):
        if not self.reflect_y:
            return coords
        if origin is None:
            origin = self.offset
        return self._rotate_xy_180(coords, origin)

    def maybe_rotate_euler_180(self, euler):
        if not self.reflect_y:
            return euler
        roll, pitch, yaw = euler
        yaw = (yaw + 180.0) % 360.0
        return (roll, pitch, yaw)

    def maybe_rotate_yaw_180(self, yaw):
        if not self.reflect_y:
            return yaw
        return (yaw + 180.0) % 360.0

    @abstractmethod
    def setup(self, scene):
        raise NotImplementedError

    @abstractmethod
    def post_setup(self):
        raise NotImplementedError

    def get_goal_sequence(self):
        """Return goal tracking data for evaluation."""
        return (
            self.goal_achieve_seq_label,
            self.goal_achieve_seq,
            self.goal_achieve_timesteps,
        )

    def _place_floor(self, scene):
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(
            gs.morphs.Mesh(
                file="assets/desk.glb",
                pos=(0.5, 0.0, 0.0),
                quat=(-0.707, 0.707, 0, 0),
                fixed=True,
                collision=False,
            ),
        )
        self.scene_objects["floor"] = scene.add_entity(
            gs.morphs.Box(
                lower=(0.0, -1.0, 0.0),
                upper=(1.0, 1.0, Z_OFFSET),
                visualization=False,
                fixed=True,
            ),
        )

    def _ground_objects(self, object_pairs, adjust_xy=False):
        eps = 5e-3

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
