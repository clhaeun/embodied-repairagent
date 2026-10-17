"""
Genesis Loop - Single task execution engine.

Two modes:
1. static: Load pre-written task code from tasks folder
2. remote_llm: Request code generation from remote LLM server

Usage:
    python genesis_loop.py --task pick_place --mode static --source_robot panda --target_robot panda
    python genesis_loop.py --task pour_liquid --mode remote_llm --source_robot panda --target_robot suction
"""

import argparse
import importlib
import sys
import json
import traceback
import time as time_module
import warnings
import numpy as np
from typing import Dict, List, Any, Optional


warnings.filterwarnings("ignore", module="pygltflib")


import genesis as gs

from src.env.env import GenesisEnv
from src.env.lmp_wrapper import LMPWrapper
from policy_provider import PolicyProvider
from prompts.prompt_utils import get_prompt
from headers import HEADER, RUN_TASK_INTERFACE, normalize_robot_type
from exceptions import SubtaskFailure, SubtaskSkip
from result_types import SubtaskResult, EpisodeResult, ExecuteTaskResult

from postprocess import postprocess_initial_code, postprocess_continuation_code 






def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def output_json(data: Dict[str, Any]):
    """Output JSON to stdout."""
    serializable_data = numpy_to_list(data)
    print(f"[JSON_START]\n{json.dumps(serializable_data, indent=2, ensure_ascii=False)}\n[JSON_END]")







def get_scene_info(env: LMPWrapper, task, task_name: str = "", robot_type: str = "panda") -> Dict[str, Any]:
    """
    Get current scene information.

    Args:
        env: LMPWrapper environment instance
        task: Task instance
        task_name: Name of the current task
        robot_type: Type of robot ("panda", "robotiq", or "suction")

    Returns:
        Dictionary containing objects info and gripper info
    """
    objects_info = []

    for obj_name in env.env.scene_objects:
        if obj_name == "gripper":
            continue

        try:
            pos = env.get_obj_pos(obj_name)
            bbox = env.get_obj_bbox(obj_name)
            objects_info.append({
                "name": obj_name,
                "position": numpy_to_list(pos) if pos is not None else None,
                "bbox": numpy_to_list(bbox) if bbox is not None else None,
                "visible": env.is_obj_visible(obj_name),
            })
        except Exception:
            objects_info.append({
                "name": obj_name,
                "position": None,
                "bbox": None,
                "visible": False,
            })


    gripper = env.env.scene_objects.get("gripper")
    gripper_info = {
        "is_open": getattr(gripper, "gripper_open", True),
        "pointing_to": getattr(gripper, "pointing_to", "down"),
        "robot_type": robot_type,
    }

    return {
        "objects": objects_info,
        "gripper": gripper_info,
    }


def get_initial_info(env: LMPWrapper, task, task_name: str, robot_type: str, descriptions: str) -> Dict[str, Any]:
    """Get initial environment information in JSON format."""
    scene_info = get_scene_info(env, task, task_name, robot_type)

    return {
        "type": "initial",
        "task_name": task_name,
        "task_description": descriptions,
        "robot_type": robot_type,
        "scene": scene_info,
    }






def load_task_class(task_name: str):
    """Load task class from src.env.tasks module."""
    task_class_map = {

        "pick_place": "PickPlaceTask",
        "sweep": "SweepTask",
        "put_in_hinge": "PutInHingeTask",
        "put_in_prismatic": "PutInPrismaticTask",
        "insert_slot": "InsertSlotTask",
        "pour_liquid": "PourLiquidTask",

        "pick_place_ball": "PickPlaceBallTask",
        "pick_place_cube": "PickPlaceCubeTask",
        "pick_place_cylinder": "PickPlaceCylinderTask",
        "pick_place_inverse_ball": "PickPlaceInverseBallTask",
        "pick_place_inverse_cube": "PickPlaceInverseCubeTask",
        "pick_place_inverse_cylinder": "PickPlaceInverseCylinderTask",
    }

    class_name = task_class_map.get(task_name)
    if not class_name:
        raise ValueError(f"Unknown task: {task_name}")

    try:

        if task_name in ["pick_place_ball", "pick_place_cube", "pick_place_cylinder",
                         "pick_place_inverse_ball", "pick_place_inverse_cube", "pick_place_inverse_cylinder"]:
            try:
                import sys
                if "src.env.tasks.pick_place_2" in sys.modules:
                    importlib.reload(sys.modules["src.env.tasks.pick_place_2"])
            except Exception:
                pass

        tasks_module = importlib.import_module("src.env.tasks")

        importlib.reload(tasks_module)
        task_cls = getattr(tasks_module, class_name)
        return task_cls
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load task class {class_name}: {e}")


def set_subtasks_target_robot(target_robot: str) -> None:
    """
    Set the target robot for subtasks module.

    This must be called before importing/executing task code that uses subtasks.

    Args:
        target_robot: Target robot type ("panda", "robotiq", or "suction")
    """
    import subtasks

    robot_map = {
        "panda": "panda",
        "robotiq": "robotiq85",
        "suction": "suction",
    }

    robot = robot_map.get(target_robot, target_robot)
    subtasks.set_target_robot(robot)
    print(f"[INFO] Subtasks module set to: subtasks_{robot}")


def load_task_code(task_name: str, source_robot: str) -> str:
    """
    Load task execution code from tasks folder.

    Args:
        task_name: Name of the task
        source_robot: Source robot type (for robot-specific code loading)

    Returns:
        Task code as string
    """

    pick_place_2_tasks = [
        "pick_place_ball",
        "pick_place_cube",
        "pick_place_cylinder",
        "pick_place_inverse_ball",
        "pick_place_inverse_cube",
        "pick_place_inverse_cylinder",
    ]

    file_name = task_name
    if task_name in pick_place_2_tasks:
        file_name = "pick_place_2"


    robot_specific_path = f"./tasks_{source_robot}/{file_name}.py"
    try:
        with open(robot_specific_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        pass


    task_path = f"./tasks/{file_name}.py"
    with open(task_path, "r", encoding="utf-8") as f:
        return f.read()


def create_execution_context(env: LMPWrapper, task, target_robot: str) -> Dict[str, Any]:
    """
    Create execution context with skill functions.

    Args:
        env: LMPWrapper environment instance
        task: Task instance
        target_robot: Target robot type ("panda", "robotiq", or "suction")

    Returns:
        Dictionary with execution context
    """
    import skill_code

    context = {
        'env': env,
        'task': task,
        'np': np,

        'move_gripper_to': skill_code.move_gripper_to,
        'move_to_position': skill_code.move_to_position,
        'move_parallel': skill_code.move_parallel,
        'rotate_gripper': skill_code.rotate_gripper,
    }

    if target_robot == "panda":

        context.update({
            'open_gripper': skill_code.open_gripper,
            'close_gripper': skill_code.close_gripper,
            'pick': skill_code.pick,
            'place': skill_code.place,
            'grasp_handle': skill_code.grasp_handle,
            'release_handle': skill_code.release_handle,
        })
    elif target_robot == "robotiq":

        context.update({
            'open_gripper': skill_code.open_robotiq85,
            'close_gripper': skill_code.close_robotiq85,
            'pick': skill_code.pick_robotiq85,
            'place': skill_code.place_robotiq85,
            'grasp_handle': skill_code.grasp_handle_robotiq85,
            'release_handle': skill_code.release_handle_robotiq85,
        })
    else:

        context.update({
            'activate_vacuum': skill_code.activate_vacuum,
            'deactivate_vacuum': skill_code.deactivate_vacuum,
            'attach_vacuum_handle': skill_code.attach_vacuum_handle,
            'detach_vacuum_handle': skill_code.detach_vacuum_handle,
        })


    context.update({
        'open_panda': getattr(skill_code, 'open_gripper', None),
        'close_panda': getattr(skill_code, 'close_gripper', None),
    })

    context.update({
        'open_robotiq85': getattr(skill_code, 'open_robotiq85', None),
        'close_robotiq85': getattr(skill_code, 'close_robotiq85', None),
        'pick_robotiq85': getattr(skill_code, 'pick_robotiq85', None),
        'place_robotiq85': getattr(skill_code, 'place_robotiq85', None),
        'grasp_handle_robotiq85': getattr(skill_code, 'grasp_handle_robotiq85', None),
        'release_handle_robotiq85': getattr(skill_code, 'release_handle_robotiq85', None),
    })

    context.update({
        'activate_suction': getattr(skill_code, 'activate_vacuum', None),
        'deactivate_suction': getattr(skill_code, 'deactivate_vacuum', None),
        'attach_suction': getattr(skill_code, 'attach_vacuum_handle', None),
        'detach_suction': getattr(skill_code, 'detach_vacuum_handle', None),
    })
    import math
    import time
    context['math'] = math
    context['time'] = time

    return context

def execute_task(
    env: LMPWrapper,
    task,
    task_name: str,
    source_robot: str,
    target_robot: str,
    provider: Optional[PolicyProvider] = None,
    max_subtask_repairs: int = 5,
    code_gen_only: bool = False,
) -> ExecuteTaskResult:
    """
    Execute task code with subtask-level generation and repair.

    New flow:
    1. Load task code (static file defining run_task structure)
    2. Task code calls provider.execute_subtask() for each subtask
    3. execute_subtask() handles: generation -> execution -> check -> repair loop
    4. SubtaskResult is collected for each subtask

    Args:
        env: LMPWrapper environment instance
        task: Task instance
        task_name: Name of the task
        source_robot: Source robot type (for reference code)
        target_robot: Target robot type (for execution)
        provider: PolicyProvider for code generation (required for remote_llm mode)
        max_subtask_repairs: Maximum repair attempts per subtask
        code_gen_only: If True, skip subtask import (only generate code)

    Returns:
        ExecuteTaskResult with final code and subtask results
    """
    result = ExecuteTaskResult(final_code="")



    if not code_gen_only:
        set_subtasks_target_robot(target_robot)



    task_code = load_task_code(task_name, source_robot)
    result.final_code = task_code

    print(f"[INFO] Loaded task code for: {task_name}")
    print(task_code)



    if code_gen_only:
        print(f"[INFO] Skipping task execution in code_gen_only mode")
        result.success = True
        return result


    context = create_execution_context(env, task, target_robot)


    subtask_results: List[SubtaskResult] = []
    context['provider'] = provider
    context['_subtask_results'] = subtask_results
    context['_max_subtask_repairs'] = max_subtask_repairs

    try:

        exec(task_code, context)


        run_task_func = context.get('run_task')
        if run_task_func is None:
            raise RuntimeError("run_task function not found in code")


        task_result = run_task_func(
            env=env,
            task=task,
            provider=provider,
            subtask_results=subtask_results,
            max_repairs=max_subtask_repairs,
        )


        result.subtask_results = subtask_results
        result.success = task_result is not None


        print(f"\n{'='*70}")
        if result.success:
            print(f"[TASK SUCCESS] {task_name}")
            print(f"  result: {task_result}")
        else:
            print(f"[TASK FAILED] {task_name}")
            print(f"  result: None")
        print(f"  total_subtasks: {len(subtask_results)}")
        successful_subtasks = sum(1 for s in subtask_results if s.success)
        print(f"  successful_subtasks: {successful_subtasks}/{len(subtask_results)}")
        print(f"{'='*70}\n")


        scene_info = get_scene_info(env, task, task_name, target_robot)
        output_json({
            "type": "complete",
            "success": result.success,
            "result": task_result,
            "scene": scene_info,
            "subtask_results": [s.to_dict() for s in subtask_results],
        })

    except SubtaskSkip as e:


        if e.result is not None:
            skip_result = e.result
        else:
            skip_result = SubtaskResult(
                subtask_name=e.subtask_name,
                obj_name=e.obj_name,
                skipped=True,
                repair_count=e.repair_attempts,
                error=e.message,
            )
        subtask_results.append(skip_result)
        result.subtask_results = subtask_results

        output_json({
            "type": "subtask_skipped",
            "subtask_name": e.subtask_name,
            "obj_name": e.obj_name,
            "repair_attempts": e.repair_attempts,
            "subtask_result": skip_result.to_dict(),
        })

        print(f"\n{'='*70}")
        print(f"[TASK FAILED - SUBTASK SKIPPED] {task_name}")
        print(f"  skipped_subtask: {e.subtask_name}")
        print(f"  obj_name: {e.obj_name}")
        print(f"  repair_attempts: {e.repair_attempts}")
        print(f"  message: {e.message}")
        print(f"  total_subtasks: {len(subtask_results)}")
        successful_subtasks = sum(1 for s in subtask_results if s.success)
        print(f"  successful_subtasks: {successful_subtasks}/{len(subtask_results)}")
        print(f"{'='*70}\n")
        print(f"[SUBTASK_SKIPPED] {e.message}")

    except Exception as e:

        error_msg = str(e)
        error_type = type(e).__name__
        tb = traceback.format_exc()

        result.subtask_results = subtask_results
        result.error = error_msg
        result.error_type = error_type


        if code_gen_only and env is None:
            scene_info = {"objects": [], "gripper": {}}
        else:
           scene_info = get_scene_info(env, task, task_name, target_robot)
        output_json({
            "type": "failure",
            "error": error_msg,
            "error_type": error_type,
            "traceback": tb,
            "scene": scene_info,
            "subtask_results": [s.to_dict() for s in subtask_results],
        })

        print(f"\n{'='*70}")
        print(f"[TASK FAILED - EXCEPTION] {task_name}")
        print(f"  error_type: {error_type}")
        print(f"  error: {error_msg}")
        print(f"  total_subtasks: {len(subtask_results)}")
        successful_subtasks = sum(1 for s in subtask_results if s.success)
        print(f"  successful_subtasks: {successful_subtasks}/{len(subtask_results)}")
        print(f"{'='*70}\n")
        print(f"[FAILURE] {error_type}: {error_msg}")

    return result






def run_task_episode(
    task_name: str,
    mode: str = "static",
    source_robot: str = "panda",
    target_robot: str = "panda",
    max_subtask_repairs: int = 5,
    code_source: str = "static",
    model: str = "code_agent",
    remote_host: str = "127.0.0.1",
    remote_port: int = 9000,
    show_viewer: bool = True,
    llm_call_logger: Optional[callable] = None,
    code_gen_only: bool = False,
    skip_env: bool = False,
) -> EpisodeResult:
    """
    Run a single task episode.

    Args:
        task_name: Name of the task (e.g., "pick_place", "sweep")
        mode: Execution mode (for compatibility, use code_source instead)
        source_robot: Source robot type for reference code ("panda", "robotiq", or "suction")
        target_robot: Target robot type for execution ("panda", "robotiq", or "suction")
        max_subtask_repairs: Maximum repair attempts per subtask
        code_source: Code source ("static" or "remote_llm")
        model: Model type for code generation ("code_agent" or "ours")
        remote_host: Remote LLM server host
        remote_port: Remote LLM server port
        show_viewer: Whether to show Genesis viewer
        llm_call_logger: Optional callback for logging LLM calls
        code_gen_only: If True, only generate subtask code without execution
        skip_env: If True, skip Genesis environment initialization

    Returns:
        EpisodeResult with execution details
    """
    result = EpisodeResult()
    start_time = time_module.time()


    if mode == "remote_llm":
        code_source = "remote_llm"

    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Source robot: {source_robot}")
    print(f"[INFO] Target robot: {target_robot}")
    print(f"[INFO] Code source: {code_source}")


    env = None
    provider = None
    task = None
    genesis_env = None


    if skip_env:
        print(f"[INFO] Skipping Genesis environment initialization (--skip_env)")
    else:

        import sys
        modules_to_clear = [k for k in sys.modules.keys() if 'pick_place_2' in k or 'src.env.tasks' in k]
        for mod in modules_to_clear:
            del sys.modules[mod]


        import torch
        if torch.cuda.is_available():
            gs.init(backend=gs.cuda, logging_level="error")
            print(f"[INFO] Using CUDA backend")
        else:
            gs.init(backend=gs.cpu, logging_level="error")
            print(f"[INFO] Using CPU backend (CUDA not available)")

    try:

        task_cls = load_task_class(task_name)



        try:
            task = task_cls(variant=42, multi_level=2, obst_level=2)
        except TypeError as te:
            if "multi_level" in str(te) or "obst_level" in str(te):
                print(f"[INFO] Task {task_name} doesn't support multi_level/obst_level, using variant only")
                task = task_cls(variant=42)
            else:
                raise

        if not skip_env:

            genesis_env = GenesisEnv()
            env = LMPWrapper(genesis_env)

            ee_type_map = {
                "panda": "gripper",
                "robotiq": "robotiq85",
                "suction": "suction",
            }
            ee_type = ee_type_map.get(target_robot, "gripper")
            if ee_type != "gripper":
                env.set_ee_type(ee_type)
            genesis_env.set_task(task)


        if not skip_env:
            obs, info = genesis_env.reset(show_viewer=show_viewer)
        else:
            obs, info = None, {}



        descriptions = getattr(task, "instruction", "") if task else ""


        if code_gen_only and env is None:

            scene_info = {"objects": [], "gripper": {}}
            object_names = []
        else:
            scene_info = get_scene_info(env, task, task_name, target_robot)
            object_names = [obj["name"] for obj in scene_info["objects"]]


        if not code_gen_only:
            initial_info = get_initial_info(env, task, task_name, target_robot, descriptions)
            output_json(initial_info)



        if code_source == "remote_llm":

            """
            prompt = get_prompt(
                task=task,
                task_name=task_name,
                object_names=object_names,
                target_robot=target_robot,
                descriptions=descriptions,
                model=model,
            )
            """

            provider = PolicyProvider(
                mode="remote_llm",
                task_name=task_name,
                model=model,
                initial_prompt="",

                remote_host=remote_host,
                remote_port=remote_port,
                object_names=object_names,
                descriptions="",

                scene_info=scene_info,
                source_robot=source_robot,
                target_robot=target_robot,
                code_gen_only=code_gen_only,
                llm_call_logger=llm_call_logger,
            )
            print(f"[INFO] Connected to remote LLM at {remote_host}:{remote_port}")
            if code_gen_only:
                print(f"[INFO] CODE_GEN_ONLY mode enabled - will skip execution")
        else:

            provider = PolicyProvider(
                mode="static",
                task_name=task_name,
                object_names=object_names,
                descriptions=descriptions,
                scene_info=scene_info,
                source_robot=source_robot,
                target_robot=target_robot,
                llm_call_logger=llm_call_logger,
            )
            print(f"[INFO] Using static subtask code from subtasks_{target_robot}/")


        exec_result = execute_task(
            env=env,
            task=task,
            task_name=task_name,
            source_robot=source_robot,
            target_robot=target_robot,
            provider=provider,
            max_subtask_repairs=max_subtask_repairs,
        )

        result.final_code = exec_result.final_code
        result.subtask_results = exec_result.subtask_results


        final_result = None if genesis_env is None else genesis_env.result
        result.success = (final_result is not None) and exec_result.success


        if exec_result.error:
            result.error = exec_result.error
            result.error_type = exec_result.error_type

    except Exception as e:
        print(f"[ERROR] Episode execution failed: {e}")
        traceback.print_exc()
        result.error = str(e)
        result.error_type = type(e).__name__
        result.success = False

    finally:

        if provider and hasattr(provider, 'llm_calls'):
            result.llm_calls = provider.llm_calls.copy()
            result.total_llm_time = provider.total_llm_time
            result.total_llm_tokens = provider.total_llm_tokens


        if provider:
            provider.close()

        if env and env.env.scene:
            try:

                if hasattr(env.env.scene, 'viewer') and env.env.scene.viewer is not None:
                    viewer = env.env.scene.viewer
                    if hasattr(viewer, '_pyrender_viewer'):
                        try:
                            pv = viewer._pyrender_viewer

                            if pv is not None and hasattr(pv, '_should_close'):
                                print("[INFO] Stopping viewer...")

                                pv._should_close = True
                                max_wait = 5.0
                                wait_interval = 0.1
                                waited = 0.0
                                while hasattr(pv, 'is_active') and pv.is_active and waited < max_wait:
                                    time_module.sleep(wait_interval)
                                    waited += wait_interval
                                if hasattr(pv, 'is_active') and pv.is_active:
                                    print("[WARN] Viewer did not close in time, forcing close...")
                                    try:
                                        if hasattr(pv, 'on_close'):
                                            pv.on_close()
                                    except Exception:
                                        pass
                                else:
                                    print("[INFO] Viewer stopped successfully")
                            else:
                                print("[INFO] Viewer not fully initialized, skipping cleanup")
                        except Exception as e:
                            print(f"[WARN] Viewer stop error: {e}")
                    time_module.sleep(0.3)


                try:
                    env.env.scene.destroy()
                except Exception as e:
                    print(f"[WARN] Scene destroy error: {e}")

            except Exception as cleanup_error:
                print(f"[WARN] Scene cleanup error: {cleanup_error}")


        try:
            gs.destroy()
        except Exception as destroy_error:
            print(f"[WARN] Genesis destroy error: {destroy_error}")

        result.execution_time = time_module.time() - start_time

    return result






def main():
    parser = argparse.ArgumentParser(description="Genesis Loop - Task Execution")
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="Task name (e.g., pick_place, sweep, put_in_hinge)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["static", "remote_llm"],
        default="remote_llm",
        help="Code source mode"
    )
    parser.add_argument(
        "--source_robot", "-s",
        choices=["panda", "robotiq", "suction"],
        default="panda",
        help="Source robot type (for reference code)"
    )
    parser.add_argument(
        "--target_robot", "-r",
        choices=["panda", "robotiq", "suction"],
        default="robotiq",
        help="Target robot type (for execution)"
    )
    parser.add_argument(
        "--code_source", "-c",
        choices=["static", "remote_llm"],
        default="remote_llm",
        help="Where to get task code"
    )
    parser.add_argument(
        "--model",
        choices=["code_agent", "ours"],
        default="code_agent",
        help="Model type for code generation"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Remote LLM server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Remote LLM server port"
    )
    parser.add_argument(
        "--max_repairs",
        type=int,
        default=5,
        help="Maximum repair attempts for general errors"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show Genesis viewer"
    )
    parser.add_argument(
        "--code_gen_only",
        action="store_true",
        help="Only generate subtask code without execution"
    )

    args = parser.parse_args()


    code_source = args.code_source if args.code_source != "static" else args.mode

    result = run_task_episode(
        task_name=args.task,
        mode=args.mode,
        source_robot=args.source_robot,
        target_robot=args.target_robot,
        max_subtask_repairs=args.max_repairs,
        code_source=code_source,
        model=args.model,
        remote_host=args.host,
        remote_port=args.port,
        show_viewer=args.viewer,
        code_gen_only=args.code_gen_only,
    )

    print(f"\n[RESULT] Success: {result.success}")
    print(f"[RESULT] Execution time: {result.execution_time:.2f}s")
    if result.error:
        print(f"[RESULT] Error: {result.error_type}: {result.error}")


if __name__ == "__main__":
    main()
