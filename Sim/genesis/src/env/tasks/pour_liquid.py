from random import Random

import numpy as np
import genesis as gs

from src.common.base_2 import BaseTask


class PourLiquidTask(BaseTask):
    task_name = "pour_liquid"
    instruction = "Pour the liquid from the beaker into the mug without spilling."

    PEDESTAL_HEIGHT = 0.08

    def __init__(self, variant, **kwargs):
        super().__init__(variant, **kwargs)
        # rng is already initialized in BaseTask
        self.initial_liquid_amount = 100
        self.spilled_amount = 0

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        self._place_pedestal(scene)
        self._place_beaker(scene)
        self._place_cup(scene)
        self._place_liquid(scene)

        return self.scene_objects

    def _place_pedestal(self, scene):
        pedestal_pos = (self.offset[0] + 0.1, self.offset[1] - 0.1, 0.0)
        self.scene_objects["pedestal"] = scene.add_entity(
            gs.morphs.Box(
                pos=pedestal_pos,
                size=(0.18, 0.18, self.PEDESTAL_HEIGHT),
                fixed=True,
            ),
            surface=gs.surfaces.Default(color=(0.6, 0.6, 0.6, 1.0)),
        )

    def _place_beaker(self, scene):
        beaker_pos = (self.offset[0] + 0.1, self.offset[1] - 0.1, 0.0)
        self.scene_objects["beaker"] = scene.add_entity(
            gs.morphs.Mesh(
                file="./assets/beaker.glb",
                pos=beaker_pos,
                euler=(0, 0, 90),
                scale=0.85,
                parse_glb_with_zup=True,
            ),
            surface=gs.surfaces.Glass(color=(0.9, 0.95, 1.0, 0.5)),
        )

    def _place_cup(self, scene):
        cup_pos = (self.offset[0] + 0.1, self.offset[1] + 0.1, 0.0)
        self.scene_objects["mug"] = scene.add_entity(
            gs.morphs.Mesh(
                file="./assets/mug.glb",
                pos=cup_pos,
                convexify=True,
                fixed=True,
                parse_glb_with_zup=True,
            ),
            surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7)),  # Single color to avoid heterogeneous colors error
        )

    def _place_liquid(self, scene):
        beaker_pos = self.scene_objects["beaker"].morph.pos
        liquid_pos = (beaker_pos[0], beaker_pos[1], 0.57 + self.PEDESTAL_HEIGHT)
        self.scene_objects["liquid"] = scene.add_entity(
            material=gs.materials.SPH.Liquid(sampler="pbs", gamma=0.005),
            morph=gs.morphs.Cylinder(
                pos=liquid_pos,
                radius=0.025,
                height=0.12,
            ),
            surface=gs.surfaces.Default(
                color=(0.3, 0.6, 1.0, 0.5),
            ),
        )

    def post_setup(self):
        ground_pairs = [
            ("floor", "pedestal"),
            ("pedestal", "beaker"),
            ("floor", "mug"),
        ]
        self._ground_objects(ground_pairs, adjust_xy=False)
        self.liquid_initial_z = None

        self.scene_objects.pop("pedestal")

    def check_result(self, env):
        liquid = self.scene_objects["liquid"]
        particle_positions = liquid.get_particles_pos().cpu().numpy()

        cup_aabb = env.get_obj_bbox("mug")
        cup_center = np.mean(cup_aabb, axis=0)
        cup_radius = (cup_aabb[1][0] - cup_aabb[0][0]) / 2 * 0.8

        particles_in_cup = 0
        for pos in particle_positions:
            dist_to_cup = np.linalg.norm(pos[:2] - cup_center[:2])
            in_cup_xy = dist_to_cup < cup_radius
            in_cup_z = cup_aabb[0][2] < pos[2] < cup_aabb[1][2] + 0.05

            if in_cup_xy and in_cup_z:
                particles_in_cup += 1

        cup_ratio = particles_in_cup / len(particle_positions)
        print(f"PourLiquidTask: cup_ratio={cup_ratio:.2f}")
        if cup_ratio > 0.7:
            return "full_success"

        return None
