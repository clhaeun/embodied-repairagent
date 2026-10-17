import ast
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from ast_policy import parse_code_bundle, common_prefix_len, is_skill_call, CodeBundle
from policy_provider import PolicyProvider

from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

from utils.helper import to_camel_case, list_task_objects


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    return obj


def output_json(data: Dict[str, Any]):
    import json
    print(f"[JSON_START]\n{json.dumps(data, indent=2, ensure_ascii=False)}\n[JSON_END]")


def get_scene_info(task, task_name: str = "") -> Dict[str, Any]:
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
                    "position": numpy_to_list(pos),
                    "quaternion": numpy_to_list(quat)
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
        "position": numpy_to_list(obs.gripper_pose[:3]),
        "quaternion": numpy_to_list(obs.gripper_pose[3:7]),
        "is_open": bool(obs.gripper_open > 0.5)
    }
    return {"objects": objects_info, "gripper": gripper_info}


def create_execution_context(env, task, robot_type: str) -> Dict[str, Any]:
    import importlib
    import math
    import time

    if robot_type.lower() == "ur5":
        skill_module_name = "skill_code_ur5"
    elif robot_type.lower() == "sawyer":
        skill_module_name = "skill_code_sawyer"
    elif robot_type.lower() == "jaco":
        skill_module_name = "skill_code_jaco"
    else:
        skill_module_name = "skill_code"

    try:
        skill_module = importlib.import_module(skill_module_name)
    except ImportError:
        skill_module = importlib.import_module("skill_code")

    ctx = {
        'env': env,
        'task': task,
        'np': np,
        'Shape': Shape,
        'Joint': Joint,
        'Dummy': Dummy,
        'ProximitySensor': ProximitySensor,
        'math': math,
        'time': time,
        'sleep': time.sleep,
    }

    for name in [
        'pick', 'place', 'move', 'push',
        'open_gripper', 'close_gripper',
        'align_two_axes', 'align_to_quaternion',
        'normalize_quaternion', 'angle_diff',
    ]:
        ctx[name] = getattr(skill_module, name, None)

    return ctx


class ASTExecuteLoop:
    """
    - Provider와 완전 분리
    - helper는 helper_hash가 바뀔 때만 compile/exec
    - step마다 provider.revise_on_step() 호출(필수)
    - failure마다 provider.repair_on_failure() 호출(필수)
    """

    def __init__(
        self,
        env,
        task,
        task_name: str,
        robot_type: str,
        provider: PolicyProvider,
        mode: str,
        max_skill_calls: Optional[int] = None,
        max_repairs: int = 10,
    ):
        self.env = env
        self.task = task
        self.task_name = task_name
        self.robot_type = robot_type
        self.provider = provider
        self.mode = mode
        self.max_skill_calls = max_skill_calls
        self.max_repairs = max_repairs

        self.ctx = create_execution_context(env, task, robot_type)

        self.executed_stmt_texts: List[str] = []
        self.skill_call_count = 0
        self._compiled_helper_hash: Optional[str] = None

    def run(self, initial_code: str):
        bundle = parse_code_bundle(initial_code)
        self._ensure_helpers_compiled(bundle)

        idx = 0
        repairs = 0

        while True:
            success_check, _ = self.task._task.success()
            if success_check:
                scene = get_scene_info(self.task, self.task_name)
                output_json({
                    "type": "complete",
                    "skill_call_count": self.skill_call_count,
                    "success": True,
                    "scene": scene,
                })
                return

            if idx >= len(bundle.run_body):
                scene = get_scene_info(self.task, self.task_name)
                success_check, _ = self.task._task.success()
                output_json({
                    "type": "complete",
                    "total_steps": len(bundle.run_body),
                    "total_skill_calls": self.skill_call_count,
                    "success": success_check,
                    "scene": scene,
                })
                return

            stmt = bundle.run_body[idx]
            stmt_text = bundle.run_stmt_texts[idx]
            is_skill, skill_name = is_skill_call(stmt)

            try:
                self._exec_one_stmt(stmt)


                self.executed_stmt_texts.append(stmt_text)

                if is_skill:
                    self.skill_call_count += 1
                    scene = get_scene_info(self.task, self.task_name)
                    success_check, _ = self.task._task.success()

                    step_payload = {
                        "type": "step",
                        "stmt_index": idx + 1,
                        "skill_call_count": self.skill_call_count,
                        "statement": stmt_text,
                        "skill_name": skill_name,
                        "success": success_check,
                        "scene": scene,
                    }

                    if self.mode == "step":
                        output_json(step_payload)

                    if self.max_skill_calls is not None and self.skill_call_count >= self.max_skill_calls:
                        output_json({
                            "type": "max_steps_reached",
                            "stmt_index": idx + 1,
                            "skill_call_count": self.skill_call_count,
                            "max_skill_calls": self.max_skill_calls,
                            "success": success_check,
                            "scene": scene,
                        })
                        return


                    new_code = self.provider.revise_on_step(
                        current_code=bundle.code,
                        executed_stmt_texts=self.executed_stmt_texts,
                        next_index=idx + 1,
                        scene=scene,
                        last_step_payload=step_payload,
                    )

                    if new_code is not None:
                        new_bundle = parse_code_bundle(new_code)
                        self._ensure_helpers_compiled(new_bundle)


                        prefix = common_prefix_len(self.executed_stmt_texts, new_bundle.run_stmt_texts)
                        idx = prefix
                        bundle = new_bundle
                        continue

                idx += 1

            except (SkillFailure, PathOutOfWorkspace) as e:
                scene = get_scene_info(self.task, self.task_name)
                err_msg = getattr(e, "message", str(e))

                failure_payload = {
                    "type": "failure",
                    "stmt_index": idx + 1,
                    "statement": stmt_text,
                    "skill_name": skill_name if is_skill else None,
                    "error": err_msg,
                    "error_type": type(e).__name__,
                    "scene": scene,
                }
                output_json(failure_payload)

                repairs += 1
                if repairs > self.max_repairs:
                    raise RuntimeError(f"Exceeded max_repairs={self.max_repairs}")


                new_code = self.provider.repair_on_failure(
                    current_code=bundle.code,
                    executed_stmt_texts=self.executed_stmt_texts,
                    failed_stmt_text=stmt_text,
                    error_payload=failure_payload,
                    scene=scene,
                )

                new_bundle = parse_code_bundle(new_code)
                self._ensure_helpers_compiled(new_bundle)
                prefix = common_prefix_len(self.executed_stmt_texts, new_bundle.run_stmt_texts)
                idx = prefix
                bundle = new_bundle
                continue

            except Exception as e:
                scene = get_scene_info(self.task, self.task_name)
                tb = traceback.format_exc()

                error_payload = {
                    "type": "error",
                    "stmt_index": idx + 1,
                    "statement": stmt_text,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": tb,
                    "scene": scene,
                }
                output_json(error_payload)

                repairs += 1
                if repairs > self.max_repairs:
                    raise RuntimeError(f"Exceeded max_repairs={self.max_repairs}")


                new_code = self.provider.repair_on_failure(
                    current_code=bundle.code,
                    executed_stmt_texts=self.executed_stmt_texts,
                    failed_stmt_text=stmt_text,
                    error_payload=error_payload,
                    scene=scene,
                )

                new_bundle = parse_code_bundle(new_code)
                self._ensure_helpers_compiled(new_bundle)
                prefix = common_prefix_len(self.executed_stmt_texts, new_bundle.run_stmt_texts)
                idx = prefix
                bundle = new_bundle
                continue

    def _ensure_helpers_compiled(self, bundle: CodeBundle):
        if self._compiled_helper_hash == bundle.helper_hash:
            return

        for _, fn_node in bundle.helper_nodes.items():
            mod = ast.Module(body=[fn_node], type_ignores=[])
            ast.fix_missing_locations(mod)
            codeobj = compile(mod, "<helper>", "exec")
            exec(codeobj, self.ctx)

        self._compiled_helper_hash = bundle.helper_hash

    def _exec_one_stmt(self, stmt: ast.stmt):
        mod = ast.Module(body=[stmt], type_ignores=[])
        ast.fix_missing_locations(mod)
        codeobj = compile(mod, "<run_skill_stmt>", "exec")
        exec(codeobj, self.ctx)