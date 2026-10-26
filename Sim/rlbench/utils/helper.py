from rlbench.task_environment import TaskEnvironment
from typing import List
from pyrep.objects.object import Object
import json
import re
import random
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor

def task_base(env: TaskEnvironment):
    return env._task.get_base()

def list_task_objects(env: TaskEnvironment) -> List[Object]:
    base = task_base(env)
    return base.get_objects_in_tree(exclude_base=True)

def to_camel_case(s):
    return ''.join(word.capitalize() for word in s.split('_'))

def to_words(s):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s).lower()

def object_names_prev(env: TaskEnvironment, task_name: str = ""):
    objects_info = []
    
    for o in list_task_objects(env):
        name = o.get_name()
        type = str(o.get_type())
        type_name = to_camel_case(type.split(".")[-1])
        
        if name != 'dirt_boundary' and any(sub in name for sub in ('waypoint', 'visual', 'spawn', 'wrap', 'topPlate', 'boundary', 'base', 'dynamic')):
            continue
        
        if task_name in ('WipeDesk',):
            exception_case_tuple = ('Cuboid',)
            if any(sub in name for sub in exception_case_tuple):
                continue
        
        if task_name in ('PutGroceriesInCupboard', 'PutAllGroceriesInCupboard'):
            exception_case_tuple = ('grasp_point', 'goal')
            if any(sub in name for sub in exception_case_tuple):
                continue

        objects_info.append({"type": type_name, "name": name})
    
    objects_info_str = ""
    for obj in objects_info:
        objects_info_str += f"\n- name: {obj['name']} | type: {obj['type']}"
    
    return objects_info_str, objects_info

def object_names(env: TaskEnvironment, task_name: str = ""):
    objects_info = []
    task_objects = None
    if len(task_name) > 0:
        try:
            with open('/workspace/rlbench_server/utils/object_mapping.json', 'r', encoding='utf-8') as f:
                task_objects_dict = json.load(f)['tasks']
                task_objects = task_objects_dict.get(task_name, None)
        except FileNotFoundError:
            print(f"WARNING: object_mapping.json not found. Falling back to default object filtering.")
            task_objects = None

    for o in list_task_objects(env):
        name = o.get_name()
        type = str(o.get_type())
        type_name = to_camel_case(type.split(".")[-1])
        if type_name == "ProximitySensor" or name.lower().startswith("success") or "goal_sensor" in name.lower():
            continue
        
        if task_objects is not None:
            if name not in task_objects:
                continue
        else:
            if name != 'dirt_boundary' and any(sub in name for sub in ('waypoint', 'visual', 'spawn', 'wrap', 'topPlate', 'boundary', 'base', 'dynamic')):
                continue
            
            if task_name in ('WipeDesk',):
                exception_case_tuple = ('Cuboid',)
                if any(sub in name for sub in exception_case_tuple):
                    continue
            
            if task_name in ('UnplugCharger',):
                exception_case_tuple = ('hole', 'sensor')
                if any(sub in name for sub in exception_case_tuple):
                    continue
            
            if task_name in ('PutGroceriesInCupboard', 'PutAllGroceriesInCupboard'):
                exception_case_tuple = ('grasp_point', 'goal')
                if any(sub in name for sub in exception_case_tuple):
                    continue
        
        _type_registry = {'Shape': Shape, 'Joint': Joint, 'Dummy': Dummy, 'ProximitySensor': ProximitySensor}
        cls = _type_registry.get(type_name, None)
        obj = cls(name)
        pos = obj.get_position()

        pos_list = pos.tolist() if hasattr(pos, 'tolist') else list(pos)
        objects_info.append({"type": type_name, "name": name, "position": pos_list})
    
    objects_info_str = ""
    for obj in objects_info:
        objects_info_str += f"\n- name: {obj['name']} | type: {obj['type']}"
    
    object_names = [obj['name'] for obj in objects_info]
    return objects_info_str, objects_info, object_names

def get_only_object_names(env: TaskEnvironment) -> List[str]:
    """Returns a list of object names in the task environment."""
    return [o.get_name() for o in list_task_objects(env)]

def get_scene_summary(env: TaskEnvironment, task_name: str = "") -> str:
    """
    Generates a string summarizing the state of the current scene.
    Includes the robot's gripper state, the position of each object,
    and also includes orientation information for specific objects defined in object_mapping.json.

    Args:
        env (TaskEnvironment): The RLBench task environment.
        task_name (str): The name of the current task.

    Returns:
        str: A summary string describing the state of the scene in English.
    """
    try:
        with open('/workspace/rlbench_server/utils/object_mapping.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
            orientation_dependent_objects = config.get("orientation_dependent_objects", [])
    except FileNotFoundError:
        print(f"ERROR: Could not find object_mapping.json. Please check the path.")
        orientation_dependent_objects = []

    obs = env.get_observation()

    gripper_open = "open" if obs.gripper_open else "closed"
    gripper_pose_xyz = obs.gripper_pose[:3]

    scene_summary_str = f"Robot gripper is currently '{gripper_open}' and is at [x, y, z] coordinates {['{:.2f}'.format(c) for c in gripper_pose_xyz]}.\n"
    scene_summary_str += "The objects in the scene are as follows:\n"

    _, objects_info = object_names(env, task_name)

    for obj_info in objects_info:
        pyrep_object = None
        for o in list_task_objects(env):
            if o.get_name() == obj_info['name']:
                pyrep_object = o
                break

        if pyrep_object:
            name = pyrep_object.get_name()
            obj_type = to_camel_case(str(pyrep_object.get_type()).split(".")[-1])
            obj_pos = pyrep_object.get_position()

            scene_summary_str += f"- name: '{name}', type: '{obj_type}', position: {['{:.2f}'.format(c) for c in obj_pos]}"

            if name in orientation_dependent_objects:
                obj_orientation = pyrep_object.get_orientation()
                scene_summary_str += f", orientation: {['{:.2f}'.format(c) for c in obj_orientation]}"

            scene_summary_str += "\n"

    return scene_summary_str

def object_names_with_noise(env: TaskEnvironment, task_name: str = "", noise_probability: float = 1.0):
    """
    Returns an object name dictionary with noise added with a certain probability.
    Noise is applied by either adding a non-existent object or removing an existing one.
    This function guarantees reproducibility by using a fixed random seed.

    Args:
        env (TaskEnvironment): The RLBench task environment.
        task_name (str): The name of the current task.
        noise_probability (float): The probability of injecting noise (between 0.0 and 1.0).

    Returns:
        tuple: A tuple containing the string representation of the noisy object list
               and the list of noisy object dictionaries.
    """

    random.seed(42)

    _, objects_info = object_names(env, task_name)

    if random.random() < noise_probability:
        if len(objects_info) >= 3:
            action = 'remove'
        else:
            action = 'add'

        try:
            with open('/workspace/rlbench_server/utils/object_mapping.json', 'r', encoding='utf-8') as f:
                tasks_dict = json.load(f)['tasks']
        except FileNotFoundError:
            print(f"ERROR: Could not find object_mapping.json. Please check the path. No noise will be added.")
            objects_info_str = ""
            for obj in objects_info:
                objects_info_str += f"\n- name: {obj['name']} | type: {obj['type']}"
            return objects_info_str, objects_info

        current_task_objects = {obj['name'] for obj in objects_info}
        all_other_objects = []
        for task, objects in tasks_dict.items():
            if task != task_name:
                for obj_name in objects:

                    if obj_name not in current_task_objects and "success" not in obj_name:
                        all_other_objects.append(obj_name)
        

        all_other_objects = list(set(all_other_objects))

        if action == 'add' and all_other_objects:
            num_to_add = random.randint(1, 3)

            num_to_add = min(num_to_add, len(all_other_objects))
            

            objects_to_add = random.sample(all_other_objects, num_to_add)
            
            for obj_name in objects_to_add:
                objects_info.append({'type': 'Object', 'name': obj_name})
            
            print(f"INFO: Noise Added - Added {num_to_add} object(s): {objects_to_add} to task '{task_name}'.")


        elif action == 'remove' and objects_info:
            initial_count = len(objects_info)


            max_removable = min(3, initial_count - 2)

            if max_removable >= 1:
                num_to_remove = random.randint(1, max_removable)
                objects_to_remove = random.sample(objects_info, num_to_remove)
                

                names_to_remove = {obj['name'] for obj in objects_to_remove}
                

                objects_info = [obj for obj in objects_info if obj['name'] not in names_to_remove]
                
                removed_names = [obj['name'] for obj in objects_to_remove]
                print(f"INFO: Noise Added - Removed {num_to_remove} object(s): {removed_names} from task '{task_name}'.")
            else:
                print(f"INFO: Noise Skipped - Cannot remove objects from task '{task_name}' as it would leave fewer than 2.")


    objects_info_str = ""
    for obj in objects_info:
        objects_info_str += f"\n- name: {obj['name']} | type: {obj['type']}"

    return objects_info_str, objects_info