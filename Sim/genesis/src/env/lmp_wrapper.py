"""
LMP Wrapper for Genesis Environment.

This module provides a high-level wrapper around the Genesis environment,
offering perception APIs and delegating action APIs to skill_code module.
"""

import os
import math

import torch
import numpy as np

from src.common.constants import FLOOR_AABB, RECORD_INTERVAL


from skill_code import (
    move_gripper_to as _move_gripper_to,
    move_to_position as _move_to_position,
    move_parallel as _move_parallel,
    rotate_gripper as _rotate_gripper,
    open_gripper as _open_gripper,
    close_gripper as _close_gripper,
    grasp_handle as _grasp_handle,
    release_handle as _release_handle,
    activate_vacuum as _activate_vacuum,
    deactivate_vacuum as _deactivate_vacuum,
    attach_vacuum_handle as _attach_vacuum_handle,
    detach_vacuum_handle as _detach_vacuum_handle,
    get_gripper_offset,
    direction_to_quat,
    _execute_joint_motion,
    _feasibility_check,
    BASE_ANGLE_OFFSET,
)


class LMPWrapper:
    """
    High-level wrapper for the Genesis environment.

    Provides:
    - Perception APIs (object positions, bounding boxes, etc.)
    - Action APIs (delegated to skill_code module)
    - Data collection utilities
    """

    def __init__(self, env):
        self.env = env
        self.env.lmp_wrapper = self

        self.trajectory = []

        self.gripper_state = {"angle": 0}
        self._grasp = {
            "active": False,
            "object": None,
            "obj_link_idx": None,
        }
        self._welded = {
            "active": False,
            "object": None,
            "obj_link_idx": None,
        }

        self._pending_record_path = None
        self._recording_active = False

        self.ee_type = "gripper"
        self.ee_name = "hand"
        self.ee_strict = True
        self.ee_set = False

    def set_ee_type(self, ee_type):
        """Set the end-effector type."""
        self.ee_type = ee_type
        self.ee_set = True
        if ee_type == "suction":
            print("[env] Using suction end-effector.")
            self.ee_name = "link7"
        elif ee_type == "robotiq85":
            print("[env] Using robotiq85 end-effector.")

            self.ee_name = "link7"

    def collect(self, data_type, show_viewer=True, record=False, record_path=None):
        """Collect demonstration or scene data."""
        if data_type == "scene":
            scene = self.env.task.collect_scene(self, show_viewer=show_viewer)
            return scene
        else:
            if record and record_path is None:
                raise ValueError("record_path must be provided when record=True")
            if record:
                self._prepare_recording(record_path)

            try:
                demo = self.env.task.collect_demo(self, show_viewer=show_viewer)
            finally:
                if record:
                    self._finalize_recording()
            return demo

    def load(self, data_type, path=None, show_viewer=False):
        """Load demonstration or scene data."""
        if data_type == "demo":
            task = self.env.task
            path = path or f"data/{data_type}"

            task_name = task.task_name + "_v" + str(task.variant)
            data_path = os.path.join(path, task_name)
            data = Demo.load(data_path)
        else:
            data = self.env.task.collect_scene(self, show_viewer=show_viewer)

        return data

    def step(self):
        """Execute one simulation step."""
        if self._grasp["active"]:
            self._control_gripped_link()
        self.env.step()

    def reset(self, *args, **kwargs):
        """Reset the environment."""
        if self._pending_record_path is not None:
            kwargs.setdefault("record", True)
            self._recording_active = False

        obs, info = self.env.reset(*args, **kwargs)
        self.trajectory = [obs]

        if self._pending_record_path is not None:
            self.env.record_camera.start_recording()
            self._recording_active = True
        return obs, info

    def check_result(self):
        """Check the task result."""
        for _ in range(100):
            self.step()

        if self.env.result is None:
            self.env.final_call = True
            for _ in range(100):
                self.step()

        return self.env.result





    def is_obj_visible(self, obj_name):
        """Check if an object is visible in the scene."""
        return obj_name in self.env.scene_objects

    def get_obj_names(self):
        """Get list of all object names in the scene."""
        scene_objects = list(self.env.scene_objects.keys())

        if self.ee_type == "suction" and "gripper" in scene_objects:
            scene_objects.remove("gripper")
            scene_objects.append("vacuum_gripper")

        return scene_objects

    def get_obj_pos(self, obj_name):
        """Get the position of an object."""
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_aabb = self.get_obj_bbox(obj_name)
        obj_pos = np.mean(obj_aabb, axis=0)

        if obj_name == "floor":
            obj_pos[2] = FLOOR_AABB[1][2]

        return obj_pos

    def get_obj_speed(self, obj_name):
        """Get the speed of an object."""
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_vel = self.env.scene_objects[obj_name].get_vel().cpu().numpy()
        obj_speed = np.linalg.norm(obj_vel)
        return obj_speed

    def get_obj_angle(self, obj_name):
        """Get the rotation angle of an object."""
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        q = self.env.scene_objects[obj_name].get_quat()
        w, x, y, z = q
        angle = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        if "handle" in obj_name:
            angle = angle + math.pi / 2

        return angle.cpu().numpy()

    def quat_changed(self, init_quat, cur_quat, thresh_deg):
        """Check if quaternion has changed beyond threshold."""
        thresh = math.radians(thresh_deg)

        norm_init_quat = init_quat / init_quat.norm()
        norm_cur_quat = cur_quat / cur_quat.norm()

        dot = torch.dot(norm_init_quat, norm_cur_quat).abs().clamp(-1.0, 1.0)
        angle = 2.0 * torch.arccos(dot)

        return angle.item() > thresh

    def get_obj_bbox(self, obj_name):
        """Get the axis-aligned bounding box of an object."""
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj = self.env.scene_objects[obj_name]
        bbox = None
        try:
            bbox = obj.get_AABB()
        except Exception:
            bbox = None

        if bbox is None:
            try:
                bbox = obj.get_vAABB()
            except Exception:
                bbox = None

        if bbox is None:
            return None

        return bbox.cpu().numpy()

    def get_obj_size(self, obj_name):
        """Get the size of an object."""
        bbox = self.get_obj_bbox(obj_name)
        size = bbox[1] - bbox[0]
        return size

    def gripper_is_open(self):
        """Check if the gripper is open."""
        return self.env.scene_objects["gripper"].gripper_open

    def obj_in_gripper(self, obj_name):
        """Check if an object is in the gripper."""
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        if self.ee_type == "gripper":
            gripper = self.env.scene_objects["gripper"]
            left_pos = gripper.get_link("left_finger").geoms[-1].get_pos()
            right_pos = gripper.get_link("right_finger").geoms[-1].get_pos()

            def line_segment_aabb_intersection_vectorized(p1, p2, box_min, box_max):
                eps = 1e-10
                d = p2 - p1

                inv_d = torch.where(torch.abs(d) > eps, 1.0 / d, torch.zeros_like(d))

                t1 = (box_min - p1) * inv_d
                t2 = (box_max - p1) * inv_d

                parallel = torch.abs(d) <= eps
                t1 = torch.where(parallel, torch.where(p1 >= box_min, 0.0, 2.0), t1)
                t2 = torch.where(parallel, torch.where(p1 <= box_max, 1.0, -1.0), t2)

                tmin_vals = torch.min(t1, t2)
                tmax_vals = torch.max(t1, t2)

                tmin = torch.max(tmin_vals.max(), torch.tensor(0.0, device=p1.device))
                tmax = torch.min(tmax_vals.min(), torch.tensor(1.0, device=p1.device))

                return tmin <= tmax

            obj_aabb = torch.from_numpy(self.get_obj_bbox(obj_name)).to(left_pos.device)

            margin = torch.tensor([0.005, 0.005, 0.02], device=left_pos.device)
            obj_aabb_min = obj_aabb[0] - margin
            obj_aabb_max = obj_aabb[1] + margin
            intersect = line_segment_aabb_intersection_vectorized(
                left_pos, right_pos, obj_aabb_min, obj_aabb_max
            )
            return bool(intersect)

        elif self.ee_type == "robotiq85":

            franka = self.env.franka
            left_finger = franka.get_link("left_inner_finger")
            right_finger = franka.get_link("right_inner_finger")


            left_pos = left_finger.get_pos()
            right_pos = right_finger.get_pos()


            pointing_to = self.env.scene_objects["gripper"].pointing_to
            if pointing_to == "down":
                tip_offset = torch.tensor([0.0, 0.0, -0.02], device=left_pos.device)
            elif pointing_to == "left":
                tip_offset = torch.tensor([0.0, -0.02, 0.0], device=left_pos.device)
            else:
                tip_offset = torch.tensor([0.0, 0.02, 0.0], device=left_pos.device)

            left_tip = left_pos + tip_offset
            right_tip = right_pos + tip_offset

            def line_segment_aabb_intersection_vectorized(p1, p2, box_min, box_max):
                eps = 1e-10
                d = p2 - p1

                inv_d = torch.where(torch.abs(d) > eps, 1.0 / d, torch.zeros_like(d))

                t1 = (box_min - p1) * inv_d
                t2 = (box_max - p1) * inv_d

                parallel = torch.abs(d) <= eps
                t1 = torch.where(parallel, torch.where(p1 >= box_min, 0.0, 2.0), t1)
                t2 = torch.where(parallel, torch.where(p1 <= box_max, 1.0, -1.0), t2)

                tmin_vals = torch.min(t1, t2)
                tmax_vals = torch.max(t1, t2)

                tmin = torch.max(tmin_vals.max(), torch.tensor(0.0, device=p1.device))
                tmax = torch.min(tmax_vals.min(), torch.tensor(1.0, device=p1.device))

                return tmin <= tmax

            obj_aabb = torch.from_numpy(self.get_obj_bbox(obj_name)).to(left_tip.device)

            margin = torch.tensor([0.005, 0.005, 0.02], device=left_tip.device)
            obj_aabb_min = obj_aabb[0] - margin
            obj_aabb_max = obj_aabb[1] + margin
            intersect = line_segment_aabb_intersection_vectorized(
                left_tip, right_tip, obj_aabb_min, obj_aabb_max
            )









            return bool(intersect)

        else:
            end_effector = self.env.franka.get_link(self.ee_name)
            pos = end_effector.get_pos().cpu().numpy()

            pointing_to = self.env.scene_objects["gripper"].pointing_to
            pos_offset = get_gripper_offset(self.ee_type, pointing_to)

            suction_pos = pos - pos_offset * 1.05

            obj_aabb = self.get_obj_bbox(obj_name)
            obj_min = obj_aabb[0]
            obj_max = obj_aabb[1]

            in_x = obj_min[0] <= suction_pos[0] <= obj_max[0]
            in_y = obj_min[1] <= suction_pos[1] <= obj_max[1]
            in_z = obj_min[2] <= suction_pos[2] <= obj_max[2]

            return in_x and in_y and in_z

    def get_empty_floor_xy(
        self,
        obj_name,
        grid_step: float | None = None,
        max_tries_sort_prefix: int = 0,
    ):
        """Find an empty position on the floor for an object."""
        inner_min = FLOOR_AABB[0] + np.array([0, 0.3, 0.0])
        inner_max = FLOOR_AABB[1] + np.array([0, -0.3, 0.0])

        obj_aabb = self.get_obj_bbox(obj_name)
        obj_size = obj_aabb[1] - obj_aabb[0]
        hx, hy = obj_size[0] / 2.0, obj_size[1] / 2.0

        sx_min = inner_min[0] + hx
        sx_max = inner_max[0] - hx - 0.2
        sy_min = inner_min[1] + hy + 0.15
        sy_max = inner_max[1] - hy - 0.15

        for name in ["hinge_body", "drawer_body"]:
            if name not in self.env.scene_objects:
                continue

            aabb = self.get_obj_bbox(name)
            sx_max = min(sx_max, aabb[0][0])

        if sx_min >= sx_max or sy_min >= sy_max:
            raise RuntimeError("Floor area too small after margin_ratio/object size.")

        others = []
        padding = 0.02
        for name in self.env.scene_objects:
            if name in ("floor", "gripper", obj_name):
                continue
            aabb = self.get_obj_bbox(name)
            padded_min = aabb[0] - np.array([padding, padding, 0.0])
            padded_max = aabb[1] + np.array([padding, padding, 0.0])
            others.append((padded_min, padded_max))

        def overlap_xy(x, y, bmin, bmax):
            return not (
                x + hx <= bmin[0]
                or x - hx >= bmax[0]
                or y + hy <= bmin[1]
                or y - hy >= bmax[1]
            )

        if grid_step is None:
            grid_step = max(obj_size[0], obj_size[1]) * 0.6

        nx = max(1, int(np.floor((sx_max - sx_min) / grid_step)) + 1)
        ny = max(1, int(np.floor((sy_max - sy_min) / grid_step)) + 1)

        xs = np.linspace(sx_min, sx_max, nx)
        ys = np.linspace(sy_min, sy_max, ny)
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

        obj_center = (obj_aabb[0] + obj_aabb[1]) * 0.5
        cur_xy = obj_center[:2]
        dists = np.linalg.norm(grid - cur_xy[None, :], axis=1)
        order = np.argsort(dists)
        if max_tries_sort_prefix > 0:
            order = order[:max_tries_sort_prefix]
        candidates = grid[order]

        def is_valid_point(x, y):
            return not any(overlap_xy(x, y, bmin, bmax) for bmin, bmax in others)

        if sx_min <= cur_xy[0] <= sx_max and sy_min <= cur_xy[1] <= sy_max:
            if is_valid_point(cur_xy[0], cur_xy[1]):
                return cur_xy.copy()

        for x, y in candidates:
            if is_valid_point(x, y):
                return np.array([x, y])

        if max_tries_sort_prefix > 0:
            for x, y in grid:
                if is_valid_point(x, y):
                    return np.array([x, y])

        raise RuntimeError("Failed to find an empty floor (x, y) from grid.")





    def move_gripper_to(self, obj_name, pointing_to="down", depth=0.01):
        """Move the gripper to a target object."""
        return _move_gripper_to(self, obj_name, pointing_to, depth)

    def move_to_position(self, pos, pointing_to="down", lift_clearance=0.12, angle=0.0):
        """Move the end-effector to a specific position."""
        return _move_to_position(self, pos, pointing_to, lift_clearance, angle)

    def move_parallel(self, move_dir, offset, pointing_to="down"):
        """Move the gripper in a direction parallel to the workspace."""
        return _move_parallel(self, move_dir, offset, pointing_to)

    def rotate_gripper(self, angle, steps=80):
        """
        Rotate the gripper around its pointing axis (wrist rotation).

        Args:
            angle: Rotation angle delta in degrees (relative to current angle).
            steps: Number of interpolation steps for smooth motion.
        """
        return _rotate_gripper(self, angle, steps)

    def open_gripper(self):
        """Open the gripper."""
        return _open_gripper(self)

    def close_gripper(self):
        """Close the gripper."""
        return _close_gripper(self)

    def grasp_handle(self, handle_name):
        """Grasp a handle object."""
        return _grasp_handle(self, handle_name)

    def release_handle(self):
        """Release a grasped handle."""
        return _release_handle(self)


    def attach_vacuum_handle(self, handle_name):
        """Attach the vacuum gripper to a handle."""
        return _attach_vacuum_handle(self, handle_name)

    def detach_vacuum_handle(self):
        """Detach the vacuum gripper from a handle."""
        return _detach_vacuum_handle(self)

    def activate_vacuum(self):
        """Activate the vacuum gripper."""
        return _activate_vacuum(self)

    def deactivate_vacuum(self):
        """Deactivate the vacuum gripper."""
        return _deactivate_vacuum(self)

    def __getattr__(self, name):
        return getattr(self.env, name)





    def _prepare_recording(self, record_path):
        """Prepare for recording."""
        record_dir = os.path.dirname(record_path)
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

        self._pending_record_path = record_path
        self._recording_active = False

    def _finalize_recording(self):
        """Finalize recording."""
        if self._recording_active:
            fps = max(1, 100 // RECORD_INTERVAL)
            self.env.record_camera.stop_recording(
                save_to_filename=self._pending_record_path,
                fps=fps,
            )

        self._pending_record_path = None
        self._recording_active = False

    def _control_gripped_link(self):
        """Control the gripped link (for handle manipulation)."""
        handle_pos = self.get_obj_pos(self._grasp["object"])
        ee = self.franka.get_link(self.ee_name)

        gripper_pos = ee.get_pos().cpu().numpy() - get_gripper_offset(
            self.ee_type, self.env.scene_objects["gripper"].pointing_to
        )
        displacement = gripper_pos - handle_pos
        distance = np.linalg.norm(displacement)

        if distance > 0.1:
            self.release_handle()
            return

        joint = self.scene_objects[self._grasp["object"]].entity.joints[0]
        if joint.type == 1:
            hinge_pos_w = joint.get_anchor_pos().cpu().numpy()
            hinge_axis_w = joint.get_anchor_axis().cpu().numpy()

            handle_idx = self._grasp["obj_link_idx"]

            f_dir = displacement / distance
            r = handle_pos - hinge_pos_w
            trial_tau = np.cross(r, f_dir)
            sgn = np.sign(np.dot(trial_tau, hinge_axis_w))
            tau_mag = 1.0
            tau_w = hinge_axis_w * (sgn * tau_mag)

            rigid = self.scene.sim.rigid_solver
            rigid.apply_links_external_torque(
                np.asarray([tau_w], dtype=np.float32),
                [handle_idx],
            )

        elif joint.type == 2:
            force = [(displacement / distance) * 8.0]
            rigid = self.scene.sim.rigid_solver
            rigid.apply_links_external_force(
                np.asarray(force, dtype=np.float32),
                [self._grasp["obj_link_idx"]],
            )