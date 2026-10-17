import json
import re
from typing import List, Dict

INTERFACE_GET_OBJECT_POSITIONS = """
def get_object_positions(index: int = None):
    \"\"\"
    Retrieves the 3D positions of objects in the environment for task execution
    \"\"\"
"""

def add_indent(text, indent_spaces=4):
    indent = ' ' * indent_spaces
    return '\n'.join(indent + line if line.strip() else line for line in text.splitlines())

def implement_get_object_positions(objects_dict, index: int = None):
    obj_instances = []
    obj_pos_dict_str = "return {"
    for obj in objects_dict:
        type_name = obj['type']
        name = obj['name']
        obj_instances.append(f"{name} = {type_name}('{name}')")
        obj_instances.append(f"{name}_pos = {name}.get_position()")
        obj_pos_dict_str += f"\n  '{name}': {name}_pos,"
    
    obj_instances_str = '\n'.join(obj_instances)
    obj_pos_dict_str += "\n}"

    full_implementation = f"{INTERFACE_GET_OBJECT_POSITIONS}\n{add_indent(obj_instances_str)}\n\n{add_indent(obj_pos_dict_str)}"

    return full_implementation

def implement_full_func_instruction(objects_dict, descriptions: str, task_name, index: int = None):
    get_obj_pos = implement_get_object_positions(objects_dict, index)


    run_task = f"""
def run_task(env, task, descriptions = None, obs = None, variation_index: int = 1):
    \"\"\"
    Main function to run the {task_name} task with the given environment and task.
    Task Description: {descriptions}
    \"\"\"
    # Get dictionary of positions of objects
    positions = get_object_positions(variation_index)

    # The following lines perform the stacking using available skills:
"""

    full_func_instruction = f"{get_obj_pos}\n\n{run_task}"
    return full_func_instruction

def implement_run_task(objects_dict, descriptions: str, task_name, index: int = None):
    HEADER = """from env import setup_environment, shutdown_environment
from skill_code import pick, place, move, align_two_axes, normalize_quaternion, angle_diff, align_to_quaternion, open_gripper, close_gripper, push
import argparse
import importlib
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env import setup_environment, shutdown_environment
from rlbench.const import colors
import inspect

import numpy as np
import math
import inspect

def _quat_mul(q1, q2):
    # (x, y, z, w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=float)

def _quat_from_euler(roll, pitch, yaw):
    cr = math.cos(roll/2);  sr = math.sin(roll/2)
    cp = math.cos(pitch/2); sp = math.sin(pitch/2)
    cy = math.cos(yaw/2);   sy = math.sin(yaw/2)
    # XYZ (roll, pitch, yaw) → quaternion (x,y,z,w)
    return np.array([
        sr*cp*cy - cr*sp*sy,   # x
        cr*sp*cy + sr*cp*sy,   # y
        cr*cp*sy - sr*sp*cy,   # z
        cr*cp*cy + sr*sp*sy    # w
    ], dtype=float)

def _yaw_from_quat(q):
    x, y, z, w = q
    # ZYX 컨벤션에서 yaw(psi) 추출
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    t0 = 2.0*(w*z + x*y)
    t1 = 1.0 - 2.0*(y*y + z*z)
    return math.atan2(t0, t1)

_flip_dict = {
    # Top-down (현재 코드와 동일: +Z → -Z)
    'down' : np.array([1.0, 0.0, 0.0, 0.0]),  # x축 180도

    # 접근축이 월드 +X (오른쪽에서)
    'right' : _quat_from_euler(0.0, math.radians(90), 0.0),   # pitch=+90°

    # 접근축이 월드 -X (왼쪽에서)
    'left'  : _quat_from_euler(0.0, math.radians(-90), 0.0),  # pitch=-90°

    # 접근축이 월드 +Y (앞에서)
    'front' : _quat_from_euler(math.radians(-90), 0.0, 0.0),  # roll=-90°

    # 접근축이 월드 -Y (뒤에서)
    'back'  : _quat_from_euler(math.radians(90), 0.0, 0.0),   # roll=+90°
}

def _axis_from_vector(v, tol=1e-6):
    v = np.asarray(v, dtype=float)
    if v.shape != (3,):
        return None
    ax = np.abs(v)
    if ax.max() < tol:
        return None
    i = int(np.argmax(ax))              # 가장 큰 축 고르기
    sign = '' if v[i] >= 0 else '-'
    axis = ('x', 'y', 'z')[i]
    out = f'{sign}{axis}'
    # 하위 호환: '-z'는 'z'로 통일 (기존 코드가 'down'→'z'만 쓰던 관행 유지)
    if out == '-z':
        out = 'z'
    return out

_approach_axis_dict = {
    'down' : 'z',
    'right' : 'x',
    'left' : '-x',
    'front' : 'y',
    'back' : '-y',
}

# object mapping 읽어오기
import json
def _get_object_constraints(task_name, object_name):
    try:
        with open('/workspace/rlbench_server/utils/object_mapping.json', 'r', encoding='utf-8') as f:
            object_additional_info = json.load(f)['object_additional_info'][task_name][object_name]
    except Exception as e:
        print('Error occurs in _get_object_constraints: ', e)
        return ('', '', [], [], None, None)
    if ('approach_dist' in object_additional_info.keys()):
        approach_dist = float(object_additional_info['approach_dist'])
    else:
        approach_dist = 0.15
    approach_axis = None
    if ('approach_axis') in object_additional_info.keys():
        approach_axis = object_additional_info['approach_axis']
    return (object_additional_info['approach_dir'], object_additional_info['yaw_align'], object_additional_info['approach_offset'], object_additional_info['prepos_offset'], approach_dist, approach_axis)

import warnings

# 모든 RuntimeWarning을 예외로 승격
warnings.filterwarnings("error", category=RuntimeWarning)

"""

    run_task_interface = f"""def run_task(env, task, object_class_mapping, task_name, descriptions = None, obs = None, variation_index: int = 1):
    \"\"\"
    Main function to run the {task_name} task with the given environment and task.
    Task Description: {descriptions}
    scene_objects = {[obj['name'] for obj in objects_dict]}
    \"\"\""""
    
    instruction = f"""{run_task_interface}
    # The following lines perform the stacking using available skills:
"""

    helper_code = """
    ### helper_code_start
    common_args = dict(env=env, task=task, approach_distance=0.15, max_steps=100, threshold=0.01, approach_axis='z', timeout=5.0)
    def run_action(action_fn, target_name: str='', offset=None, **overrides):
        if ((action_fn.__name__ == 'close_gripper') or (action_fn.__name__ == 'open_gripper')):
            return action_fn(env=env, task=task)
        args = dict(common_args)
        args.update(overrides)
        args['approach_distance'] = max(0.0, min(args.get('approach_distance', 0.15), 0.5))
        args['timeout'] = float(args.get('timeout', 3.0))
        obj_type = object_class_mapping.get(target_name, None)
        if (obj_type is None):
            raise ValueError(f"[run_action] Unknown target_name '{target_name}' (not in object_type_mapping).")
        _type_registry = {'Shape': Shape, 'Joint': Joint, 'Dummy': Dummy, 'ProximitySensor': ProximitySensor}
        cls = _type_registry.get(obj_type, None)
        if (cls is None):
            raise ValueError(f"[run_action] Unsupported object type '{obj_type}' for target '{target_name}'.")
        obj = cls(target_name)
        pos = obj.get_position()
        print('object position: ', pos)
        quaternion = obj.get_quaternion()
        print('object orientation: ', quaternion)
        approach_dir = ''
        yaw_align = ''
        approach_offset = []
        prepos_offset = []
        approach_dist = None
        approach_axis = None
        if ((action_fn.__name__ == 'pick') or (action_fn.__name__ == 'place') or (action_fn.__name__ == 'move')) or (action_fn.__name__ == 'push'):
            (approach_dir, yaw_align, approach_offset, prepos_offset, approach_dist, approach_axis) = _get_object_constraints(task_name=task_name, object_name=target_name)
        if ((action_fn.__name__ == 'pick') or (action_fn.__name__ == 'place') or (action_fn.__name__ == 'move')):
            if (len(prepos_offset) == 3):
                prepos_offset = np.array(prepos_offset, dtype=float)
                pre_grasp_pos = (pos + prepos_offset)
                move(env, task, target_pos=pre_grasp_pos)
            yaw = _yaw_from_quat(quaternion)
            if ((approach_dir != '') and (yaw_align != '')):
                flip_quat = _flip_dict[approach_dir]
                yaw_offset = (0.0 if (yaw_align == 'parallel') else (math.pi / 2.0))
                yaw_quat = _quat_from_euler(0.0, 0.0, (yaw + yaw_offset))
                target_quat = _quat_mul(yaw_quat, flip_quat)
                target_quat = (target_quat / np.linalg.norm(target_quat))
                align_to_quaternion(env, task, target_quat=tuple(target_quat))
        if (offset is not None):
            pos = (pos + np.array(offset, dtype=float))
        elif (len(approach_offset) == 3):
            pos = (pos + np.array(approach_offset, dtype=float))
            print('approach_offset applied: ', approach_offset)
        target_pos = pos
        sig = inspect.signature(action_fn)
        valid_params = sig.parameters.keys()
        filtered_args = {k: v for (k, v) in args.items() if (k in valid_params)}
        if ('target_pos' in valid_params):
            filtered_args['target_pos'] = target_pos
        if (('approach_distance' in valid_params) and (approach_dist is not None)):
            filtered_args['approach_distance'] = approach_dist
        vec_for_axis = None
        if ((len(prepos_offset) == 3) and (((offset is not None) and (len(offset) == 3)) or (len(approach_offset) == 3))):
            tgt = (np.array(offset, float) if ((offset is not None) and (len(offset) == 3)) else np.array(approach_offset, float))
            vec_for_axis = (tgt - np.array(prepos_offset, float))
        elif ((offset is not None) and (len(offset) == 3)):
            vec_for_axis = np.array(offset, float)
        elif (len(approach_offset) == 3):
            vec_for_axis = np.array(approach_offset, float)
        approach_axis_auto = (_axis_from_vector(vec_for_axis) if (vec_for_axis is not None) else None)
        if ((approach_axis_auto is None) and (approach_dir != '')):
            approach_axis_auto = _approach_axis_dict.get(approach_dir)
        elif (approach_axis_auto is None):
            approach_axis_auto = common_args.get('approach_axis', 'z')
        if approach_axis is not None:
            approach_axis_auto = approach_axis
        if ('approach_axis' in valid_params):
            print('approach_axis(auto): ', approach_axis_auto, '  (from vec:', vec_for_axis, ')')
            filtered_args['approach_axis'] = approach_axis_auto
            
        print(f"[Plan] {action_fn.__name__} -> name='{target_name}', offset={offset}, overrides={overrides}")
        return action_fn(**filtered_args)
    ### helper_code_end
"""

    run_task_implementation = HEADER + "\n" + run_task_interface + helper_code
    
    return instruction, run_task_implementation
    