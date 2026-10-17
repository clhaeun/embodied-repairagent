from random import Random

import genesis as gs

from src.common.base_2 import BaseTask
from src.common.constants import Z_OFFSET
from src.common.utils import get_color


class InsertSlotTask(BaseTask):
    task_name = "insert_slot"
    instruction = "Remove the blocking object and insert the target into the slot."
    use_weld = False

    def __init__(self, variant, **kwargs):
        super().__init__(variant, **kwargs)


    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)


        left_offset = (self.offset[0], self.offset[1] - 0.12, Z_OFFSET)
        right_offset = (self.offset[0], self.offset[1] + 0.12, Z_OFFSET)

        if self.rng.choice([True, False]):
            target_offset, start_offset = left_offset, right_offset
        else:
            target_offset, start_offset = right_offset, left_offset

        self._place_slot(scene, "target_slot", target_offset)
        self._place_slot(scene, "auxiliary_slot", start_offset)

        self._place_objects(scene)

        return self.scene_objects

    def _place_slot(self, scene, name, pos):
        self.scene_objects[name] = scene.add_entity(
            gs.morphs.Mesh(
                file="./assets/slot.glb",
                pos=pos,
                fixed=True,
                parse_glb_with_zup=True,
            ),
            surface=gs.surfaces.Smooth(color=(0.6, 0.6, 0.6)),
        )

        if "target" in name:
            scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.08,
                    height=0.003,
                    pos=(pos[0], pos[1], 0.503),
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Smooth(color=get_color("blue")),
            )

    def _place_objects(self, scene):
        self.scene_objects["blocking_object"] = scene.add_entity(
            gs.morphs.Box(
                size=(0.025, 0.025, 0.10),
                pos=(0, 0, 0.55),
            ),
            surface=gs.surfaces.Smooth(color=(1.0, 1.0, 1.0, 0.15)),
        )

        self.scene_objects["target_object"] = scene.add_entity(
            gs.morphs.Box(
                size=(0.025, 0.025, 0.10),
                pos=(0, 0, 0.55),
            ),
            surface=gs.surfaces.Smooth(color=get_color("blue")),
        )

    def post_setup(self):
        slot_pairs = [
            ("floor", "target_slot"),
            ("floor", "auxiliary_slot"),
        ]
        self._ground_objects(slot_pairs, adjust_xy=False)

        obj_pairs = [
            ("target_slot", "blocking_object"),
            ("auxiliary_slot", "target_object"),
        ]
        self._ground_objects(obj_pairs, adjust_xy=True)

    def check_result(self, env):
        if env.obj_in_gripper("target_object") and not env.gripper_is_open():
            return None

        target_aabb = env.get_obj_bbox("target_object")
        slot_aabb = env.get_obj_bbox("target_slot")

        margin = 0.01

        target_in_slot_xy = (
            target_aabb[0][0] > slot_aabb[0][0] + margin
            and target_aabb[1][0] < slot_aabb[1][0] - margin
            and target_aabb[0][1] > slot_aabb[0][1] + margin
            and target_aabb[1][1] < slot_aabb[1][1] - margin
        )


        slot_mid_z = (slot_aabb[0][2] + slot_aabb[1][2]) / 2
        target_inserted_z = target_aabb[0][2] < slot_mid_z

        if target_in_slot_xy and target_inserted_z:
            return "full_success"

        return None
