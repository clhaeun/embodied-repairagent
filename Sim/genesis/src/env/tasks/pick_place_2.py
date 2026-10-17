from random import Random

import numpy as np
import genesis as gs

from src.common.base_2 import BaseTask
from src.common.utils import get_color


class BasePickPlaceTask(BaseTask):
    task_name = "pick_place"
    instruction = "Place all the {target_type}s in the tray."

    OBJECT_TYPES = ["ball", "cube", "cylinder"]
    target_type = None  # Override in subclasses
    inverse = False  # If True: tray inverted, no walls

    def __init__(self, variant, target_type=None, num_targets=3, num_distractors=2):
        super().__init__(variant)

        if target_type is None:
            target_type = self.target_type or self.rng.choice(self.OBJECT_TYPES)
        self.target_type = target_type

        self.num_targets = num_targets
        self.num_distractors = num_distractors

        self.instruction = f"Place all the {target_type}s in the tray."

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        self._place_tray(scene)
        self._place_box_with_objects(scene)

        return self.scene_objects

    def _place_tray(self, scene):
        tray_pos = (self.offset[0], self.offset[1] - 0.2, 0.0)
        tray_euler = (180, 180, 0) if self.inverse else (0, 0, 0)

        # Use Box instead of tray.glb Mesh to avoid heterogeneous colors error
        # Visual tray box with euler rotation
        full_size = (0.287, 0.23, 0.016)
        tray_visual_z = 0.508

        if self.inverse:
            # Inverse tray: dark gray, euler rotated
            self.scene_objects["tray"] = scene.add_entity(
                gs.morphs.Box(
                    size=full_size,
                    pos=(tray_pos[0], tray_pos[1], tray_visual_z),
                    euler=tray_euler,
                    fixed=True,
                ),
                surface=gs.surfaces.Smooth(color=(0.15, 0.15, 0.15)),
            )
        else:
            # Normal tray: lighter gray, no rotation
            self.scene_objects["tray"] = scene.add_entity(
                gs.morphs.Box(
                    size=full_size,
                    pos=(tray_pos[0], tray_pos[1], tray_visual_z),
                    euler=tray_euler,
                    fixed=True,
                ),
                surface=gs.surfaces.Smooth(color=(0.35, 0.35, 0.35)),
            )

        if self.inverse:
            # Split collision into center (high friction) and outer (low friction)
            tray_collision_z = 0.508

            # Center region (50% of area)
            center_size = (full_size[0] * 0.5, full_size[1] * 0.5, full_size[2])
            self.scene_objects["tray_collision_center"] = scene.add_entity(
                gs.morphs.Box(
                    size=center_size,
                    pos=(tray_pos[0], tray_pos[1], tray_collision_z),
                    fixed=True,
                    visualization=False,
                ),
            )

            # Outer regions (4 strips around center, no overlap)
            outer_x = (full_size[0] - center_size[0]) / 2
            outer_y = (full_size[1] - center_size[1]) / 2

            outer_regions = [
                # Left strip
                ((outer_x, full_size[1], full_size[2]),
                 (tray_pos[0] - center_size[0] / 2 - outer_x / 2, tray_pos[1], tray_collision_z)),
                # Right strip
                ((outer_x, full_size[1], full_size[2]),
                 (tray_pos[0] + center_size[0] / 2 + outer_x / 2, tray_pos[1], tray_collision_z)),
                # Front strip
                ((center_size[0], outer_y, full_size[2]),
                 (tray_pos[0], tray_pos[1] - center_size[1] / 2 - outer_y / 2, tray_collision_z)),
                # Back strip
                ((center_size[0], outer_y, full_size[2]),
                 (tray_pos[0], tray_pos[1] + center_size[1] / 2 + outer_y / 2, tray_collision_z)),
            ]
            for i, (size, pos) in enumerate(outer_regions):
                self.scene_objects[f"tray_collision_outer_{i}"] = scene.add_entity(
                    gs.morphs.Box(
                        size=size,
                        pos=pos,
                        fixed=True,
                        visualization=False,
                    ),
                )
            return

        wall_height = 0.08
        wall_thickness = 0.005
        tray_half_x = 0.143
        tray_half_y = 0.115
        tray_top_z = 0.51
        walls = [
            (
                "tray_wall_front",
                (tray_pos[0] - tray_half_x, tray_pos[1], tray_top_z),
                (wall_thickness, tray_half_y * 2, wall_height),
            ),
            (
                "tray_wall_back",
                (tray_pos[0] + tray_half_x, tray_pos[1], tray_top_z),
                (wall_thickness, tray_half_y * 2, wall_height),
            ),
            (
                "tray_wall_left",
                (tray_pos[0], tray_pos[1] - tray_half_y, tray_top_z),
                (tray_half_x * 2, wall_thickness, wall_height),
            ),
            (
                "tray_wall_right",
                (tray_pos[0], tray_pos[1] + tray_half_y, tray_top_z),
                (tray_half_x * 2, wall_thickness, wall_height),
            ),
        ]

        for _, pos, size in walls:
            scene.add_entity(
                gs.morphs.Box(size=size, pos=pos, fixed=True, visualization=False),
            )

    def _place_box_with_objects(self, scene):
        box_pos = (self.offset[0], self.offset[1] + 0.2, 0.0)
        box = scene.add_entity(
            gs.morphs.URDF(
                file="./assets/box_w_handle/mobility.urdf",
                pos=box_pos,
                euler=(0, 0, 90),
                scale=0.28,
                convexify=False,
                fixed=True,
                merge_fixed_links=False,
            ),
        )
        self.scene_objects["box_full"] = box
        self.scene_objects["box_handle"] = box.get_link("handle")
        self.scene_objects["box_body"] = box.get_link("body")
        self.scene_objects["box_lid"] = box.get_link("lid")

        self.target_names = []
        self.distractor_names = []

        target_colors = ["red", "blue", "green", "yellow", "purple"]
        self.rng.shuffle(target_colors)

        for i in range(self.num_targets):
            color = target_colors[i % len(target_colors)]
            obj_name = f"{color}_{self.target_type}"
            self.target_names.append(obj_name)
            self._add_object(scene, obj_name, self.target_type, color, (0, 0, 1.0))

        distractor_types = [t for t in self.OBJECT_TYPES if t != self.target_type]
        distractor_colors = ["orange", "cyan", "pink"]

        for i in range(self.num_distractors):
            d_type = distractor_types[i % len(distractor_types)]
            color = distractor_colors[i % len(distractor_colors)]
            obj_name = f"{color}_{d_type}"
            self.distractor_names.append(obj_name)
            self._add_object(scene, obj_name, d_type, color, (0, 0, 1.0))

    def _add_object(self, scene, name, obj_type, color, pos):
        if obj_type == "ball":
            # Use Sphere with single color to avoid heterogeneous color point cloud issues
            # Using uniform gray color instead of individual colors to prevent "Loading point clouds with heterogeneous colors is not supported" error
            print(f"[DEBUG] Creating {name} as Sphere with uniform gray color (original color: {color})")
            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Sphere(
                    radius=0.02,
                    pos=pos,
                ),
                surface=gs.surfaces.Smooth(color=(0.5, 0.5, 0.5)),  # Single gray color for all balls
            )
            print(f"[DEBUG] Successfully created {name} as Sphere")
        elif obj_type == "cube":
            yaw = self.rng.uniform(-30, 30)
            print(f"[DEBUG] Creating {name} as Box with uniform gray color (original color: {color})")
            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Box(
                    size=(0.03, 0.03, 0.03),
                    pos=pos,
                    euler=(0, 0, yaw),
                ),
                surface=gs.surfaces.Smooth(color=(0.6, 0.6, 0.6)),  # Single gray color for all cubes
            )
            print(f"[DEBUG] Successfully created {name} as Box")
        elif obj_type == "cylinder":
            print(f"[DEBUG] Creating {name} as Cylinder with uniform gray color (original color: {color})")
            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.015,
                    height=0.04,
                    pos=pos,
                ),
                surface=gs.surfaces.Smooth(color=(0.7, 0.7, 0.7)),  # Single gray color for all cylinders
            )
            print(f"[DEBUG] Successfully created {name} as Cylinder")

    def post_setup(self):
        ground_pairs = [("floor", "tray"), ("floor", "box_full")]
        self._ground_objects(ground_pairs, adjust_xy=False)

        # Open lid first to get correct interior AABB
        self.scene_objects["box_full"].set_dofs_position(
            [np.pi / 2], dofs_idx_local=[0]
        )

        body_link = self.scene_objects["box_full"].get_link("body")
        body_aabb = body_link.get_AABB().cpu().numpy()
        box_min, box_max = body_aabb[0], body_aabb[1]
        margin = 0.02
        # Place objects slightly above box floor, inside the box
        drop_height = box_min[2] + 0.05

        # Close lid after getting positions
        self.scene_objects["box_full"].set_dofs_position(
            [-np.pi / 2], dofs_idx_local=[0]
        )

        all_objects = self.target_names + self.distractor_names
        self.rng.shuffle(all_objects)
        n = len(all_objects)

        # Grid layout: 2 rows (back row has fewer items)
        back_count = n // 2
        front_count = n - back_count

        x_center = (box_min[0] + box_max[0]) / 2
        y_range = box_max[1] - box_min[1] - 2 * margin

        positions = []
        # Back row
        for i in range(back_count):
            x = x_center + (i - (back_count - 1) / 2) * 0.1
            y = box_min[1] + margin + y_range * 0.7
            positions.append((x, y))
        # Front row
        for i in range(front_count):
            x = x_center + (i - (front_count - 1) / 2) * 0.1
            y = box_min[1] + margin + y_range * 0.3
            positions.append((x, y))

        for name, (x, y) in zip(all_objects, positions):
            self.scene_objects[name].set_pos(np.array([x, y, drop_height]))

        self.satisfied_targets = []
        self.scene_objects.pop("box_full")

        # Set friction for inverse tray: high in center, very slippery on edges
        if self.inverse:
            if "tray_collision_center" in self.scene_objects:
                self.scene_objects["tray_collision_center"].set_friction(3.0)
            for i in range(4):
                key = f"tray_collision_outer_{i}"
                if key in self.scene_objects:
                    self.scene_objects[key].set_friction(0.01)  # Minimum friction (very slippery)

            # Store surface_info metadata on tray for background validation
            # This info becomes observable when gripper moves above tray
            self.scene_objects["tray"].surface_info = {
                "center": {"friction": 3.0, "slippery": False},
                "edges": {"friction": 0.01, "slippery": True},
                "note": "Place objects at center, or use very low place_height (0.01) on edges to prevent sliding",
            }

    def check_result(self, env):
        self.satisfied_targets = []

        tray_aabb = env.get_obj_bbox("tray")
        center = (tray_aabb[0] + tray_aabb[1]) / 2
        half_size = (tray_aabb[1] - tray_aabb[0]) / 2 * 0.8
        tray_min = center - half_size
        tray_max = center + half_size

        def in_tray(obj_aabb):
            xy_ok = np.all(
                (tray_min[:2] - 0.04 < obj_aabb[0][:2])
                & (obj_aabb[1][:2] < tray_max[:2] + 0.04)
            )
            z_ok = tray_aabb[0][2] - 0.02 < obj_aabb[0][2] < tray_aabb[1][2] + 0.02
            return xy_ok and z_ok

        for name in self.target_names:
            if name in self.satisfied_targets:
                continue

            in_gripper = env.obj_in_gripper(name) and not env.gripper_is_open()

            if not in_gripper and in_tray(env.get_obj_bbox(name)):
                self.satisfied_targets.append(name)

        # print(f"satisfied: {self.satisfied_targets} / {self.target_names}")

        for name in self.distractor_names:
            if in_tray(env.get_obj_bbox(name)):
                print("Distractor", name, "in tray.")
                return None

        if len(self.satisfied_targets) == len(self.target_names):
            return "full_success"

        return None


class PickPlaceBallTask(BasePickPlaceTask):
    task_name = "pick_place_ball"
    target_type = "ball"


class PickPlaceCubeTask(BasePickPlaceTask):
    task_name = "pick_place_cube"
    target_type = "cube"


class PickPlaceCylinderTask(BasePickPlaceTask):
    task_name = "pick_place_cylinder"
    target_type = "cylinder"


# Inverse variants (tray flipped, no walls)
class PickPlaceInverseBallTask(BasePickPlaceTask):
    task_name = "pick_place_inverse_ball"
    target_type = "ball"
    inverse = True


class PickPlaceInverseCubeTask(BasePickPlaceTask):
    task_name = "pick_place_inverse_cube"
    target_type = "cube"
    inverse = True


class PickPlaceInverseCylinderTask(BasePickPlaceTask):
    task_name = "pick_place_inverse_cylinder"
    target_type = "cylinder"
    inverse = True
