


import argparse
import importlib
import numpy as np
import json

from env import setup_environment, shutdown_environment

from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.helper import object_names

from rlbench.gym import RLBenchEnv

from grasp_info import (
    get_graspable_objects_info,
    check_gripper_alignment_for_scene,
    check_gripper_alignment,
    get_grasp_guidance_for_object,
    is_graspable_object,
    GRIPPER_SPECS,
)

def patched_close(self) -> None:
    pass

RLBenchEnv.close = patched_close


def get_scene_info_simple(task, task_name: str, robot_type: str):
    """간단한 scene info 반환"""
    from pyrep.objects.shape import Shape
    from pyrep.objects.joint import Joint
    from pyrep.objects.dummy import Dummy
    from pyrep.objects.proximity_sensor import ProximitySensor
    from utils.helper import list_task_objects, to_camel_case

    _type_registry = {'Shape': Shape, 'Joint': Joint, 'Dummy': Dummy, 'ProximitySensor': ProximitySensor}

    objects_info = []
    for o in list_task_objects(task):
        name = o.get_name()
        type_str = str(o.get_type())
        type_name = to_camel_case(type_str.split(".")[-1])

        if any(sub in name for sub in ('waypoint', 'visual', 'spawn', 'wrap', 'topPlate', 'boundary', 'base', 'dynamic')):
            if name != 'dirt_boundary':
                continue

        try:
            cls = _type_registry.get(type_name)
            if cls:
                obj = cls(name)
                pos = obj.get_position()
                quat = obj.get_quaternion()
                objects_info.append({
                    "name": name,
                    "type": type_name,
                    "position": pos.tolist() if hasattr(pos, 'tolist') else list(pos),
                    "quaternion": quat.tolist() if hasattr(quat, 'tolist') else list(quat)
                })
        except Exception:
            objects_info.append({
                "name": name,
                "type": type_name,
                "position": None,
                "quaternion": None
            })

    obs = task.get_observation()
    gripper_info = {
        "position": obs.gripper_pose[:3].tolist(),
        "quaternion": obs.gripper_pose[3:7].tolist(),
        "is_open": bool(obs.gripper_open > 0.5)
    }

    return {
        "objects": objects_info,
        "gripper": gripper_info
    }


def print_grasp_info(task, task_name: str, robot_type: str):
    """Grasp guidance 정보 출력"""
    scene_info = get_scene_info_simple(task, task_name, robot_type)
    objects_info = scene_info["objects"]
    gripper_info = scene_info["gripper"]

    print("\n" + "="*60)
    print("GRASP ALIGNMENT INFO")
    print("="*60)


    print(f"\n[Gripper Status]")
    print(f"  Position: {[f'{x:.3f}' for x in gripper_info['position']]}")
    print(f"  Quaternion: {[f'{x:.3f}' for x in gripper_info['quaternion']]}")
    print(f"  Is Open: {gripper_info['is_open']}")


    grasp_infos, grasp_guidance_text = get_graspable_objects_info(
        objects_info, task_name, robot_type
    )

    print(f"\n[Graspable Objects] ({len(grasp_infos)} found)")
    for info in grasp_infos:
        print(f"\n  Object: '{info['object_name']}'")
        print(f"    Dimensions (mm): x={info['object_dimensions_mm']['x']}, y={info['object_dimensions_mm']['y']}, z={info['object_dimensions_mm']['z']}")
        print(f"    Can Grasp: {info['can_grasp']}")
        print(f"    Grasp Width (mm): {info['grasp_width_mm']}")
        print(f"    Grasp Axis (local): {info['grasp_axis_local']}")
        print(f"    Grasp Axis (world): {info['grasp_axis_world']}")
        if info.get('recommended_approach'):
            print(f"    Recommended Approach: {info['recommended_approach']['world_direction']}")


    print(f"\n[Initial Grasp Guidance]")
    if grasp_guidance_text:
        print(f"  {grasp_guidance_text}")
    else:
        print("  No alignment needed for any objects.")


    print(f"\n[Current Gripper Alignment Check]")
    gripper_quat = np.array(gripper_info["quaternion"])
    alignment_guidance = check_gripper_alignment_for_scene(
        gripper_quat,
        objects_info,
        task_name,
        robot_type
    )
    if alignment_guidance:
        print(f"  {alignment_guidance}")
    else:
        print("  Gripper is properly aligned (or no alignment needed).")

    print("\n" + "="*60)


def print_object_alignment(task, task_name: str, robot_type: str, obj_name: str):
    """특정 object에 대한 alignment 정보 출력"""
    scene_info = get_scene_info_simple(task, task_name, robot_type)
    gripper_quat = np.array(scene_info["gripper"]["quaternion"])

    print(f"\n[Alignment Check for '{obj_name}']")

    if not is_graspable_object(obj_name, task_name):
        print(f"  '{obj_name}' is not a graspable object.")
        return

    grasp_info = get_grasp_guidance_for_object(obj_name, task_name, robot_type)
    if grasp_info:
        print(f"  Dimensions (mm): {grasp_info['object_dimensions_mm']}")
        print(f"  Grasp Axis (local): {grasp_info['grasp_axis_local']}")
        print(f"  Grasp Axis (world): {grasp_info['grasp_axis_world']}")
        if grasp_info.get('natural_language'):
            print(f"  Guidance: {grasp_info['natural_language']}")
        else:
            print(f"  No alignment needed.")

    alignment = check_gripper_alignment(gripper_quat, obj_name, task_name, robot_type)
    if alignment:
        print(f"  Current Status: MISALIGNED")
        print(f"  {alignment}")
    else:
        print(f"  Current Status: ALIGNED (or no alignment needed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grasp Alignment Test Executor - Test grasp_info.py alignment guidance"
    )
    parser.add_argument(
        "--robot", "-r",
        default="ur5",
        help="Robot type (panda, ur5, sawyer, jaco)"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="RLBench task class name, e.g. PhoneOnBase, MeatOnGrill"
    )
    args = parser.parse_args()
    robot_type = args.robot.lower()
    task_name = args.task


    tasks_module = importlib.import_module("rlbench.tasks")
    try:
        task_cls = getattr(tasks_module, task_name)
    except AttributeError:
        raise SystemExit(f"Error: Task class '{task_name}' not found.")


    print(f"[Main] Setting up environment with {robot_type} robot...")
    env, task = setup_environment(task_cls, robot_type)

    descriptions, obs = task.reset()

    objects_str, objects_dict, obj_names = object_names(task, task_name)
    print(f"[Main] Task: {task_name}")
    print(f"[Main] Description: {descriptions}")
    print(f"[Main] Objects: {obj_names}")


    print_grasp_info(task, task_name, robot_type)

    print("\n[Commands]")
    print("  info      - Show grasp alignment info")
    print("  check <obj> - Check alignment for specific object")
    print("  run       - Run task skill code")
    print("  restart   - Restart task")
    print("  exit      - Exit")

    cmd = "run"
    
    while True:
        if env.is_shutdown():
            print("[Main] Environment is shutdown.")
            break



        if cmd.lower() in ("exit", "quit"):
            print("[Main] Exiting...")
            shutdown_environment(env)
            break

        elif cmd.lower() == "restart":
            print("[Main] Restarting task...")
            descriptions, obs = task.reset()
            print("[Main] Task restarted.")
            print_grasp_info(task, task_name, robot_type)

        elif cmd.lower() == "info":
            print_grasp_info(task, task_name, robot_type)

        elif cmd.lower().startswith("check "):
            obj_name = cmd[6:].strip()
            print_object_alignment(task, task_name, robot_type, obj_name)

        elif cmd.lower() == "run":
            print("[Main] Running task skill...")
            try:
                module = importlib.import_module(f"tasks_{robot_type}.{task_name}")
                importlib.reload(module)
                run_task = getattr(module, "run_skill")
                run_task(env, task, descriptions, obs)

                success, terminate = task._task.success()
                if success:
                    print("[Main] Task completed successfully!")
                else:
                    print("[Main] Task not completed.")


                print("\n[Post-run Alignment Check]")
                print_grasp_info(task, task_name, robot_type)

                cmd = "exit"

            except (SkillFailure, PathOutOfWorkspace) as e:
                error_msg = getattr(e, 'message', str(e))
                print(f"[FAILURE] {type(e).__name__}: {error_msg}")


                scene_info = get_scene_info_simple(task, task_name, robot_type)
                gripper_quat = np.array(scene_info["gripper"]["quaternion"])
                alignment_guidance = check_gripper_alignment_for_scene(
                    gripper_quat,
                    scene_info["objects"],
                    task_name,
                    robot_type
                )
                if alignment_guidance:
                    print(f"\n[ALIGNMENT GUIDANCE]")
                    print(f"  {alignment_guidance}")
                
                cmd = "exit"

            except Exception as e:
                print(f"[ERROR] {type(e).__name__}: {e}")
                cmd = "exit"

        elif cmd.lower() == "gripper":
            scene_info = get_scene_info_simple(task, task_name, robot_type)
            gripper = scene_info["gripper"]
            print(f"[Gripper]")
            print(f"  Position: {gripper['position']}")
            print(f"  Quaternion: {gripper['quaternion']}")
            print(f"  Is Open: {gripper['is_open']}")

        else:
            print(f"Unknown command: {cmd}")
            print("Commands: info, check <obj>, run, restart, gripper, exit")
