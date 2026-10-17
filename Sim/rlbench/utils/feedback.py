import json
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np
from rlbench.environment import Environment
from rlbench.task_environment import TaskEnvironment
from pyrep.objects.shape import Shape
from .helper import object_names, get_only_object_names
from .analyze_failure import analyze_skill_failure

def get_all_object_positions(task):
    obj_names = get_only_object_names(task)
    filtered_obj_names = [name for name in obj_names if not name.startswith(('waypoint', 'success', 'boundary'))]
    result_dict = dict()

    for obj_name in filtered_obj_names:
        try:
            obj_shape = Shape(obj_name)
            obj_pos = obj_shape.get_position()
            result_dict[obj_name] = obj_pos
        except Exception:
            continue

    return result_dict

@dataclass
class Feedback:
    env: Environment
    task: TaskEnvironment
    robot_pos: np.ndarray = None
    object_positions: Dict[str, np.ndarray] = None
    reward: float = 0.0
    done: bool = False

    def __str__(self):
        if self.task and self.robot_pos is None:
            self.robot_pos = self.task.get_observation().gripper_pose

        if self.task and (self.object_positions is None or len(self.object_positions) == 0):
            self.object_positions = get_all_object_positions(self.task)

        failure_reason, feedback_message = analyze_skill_failure(
            is_skill_done=True,
            skill_type=self.skill_type,
            robot_pos=self.robot_pos, 
            objects=self.object_positions, 
            waypoints=self.waypoints, 
            waypoint_index=self.waypoint_index, 
            attempted_action=self.attempted_action, 
            step_index=self.step_index, 
            original_robot_pos=self.original_robot_pos, 
            msg=self.error_message,
        )

        return f"Failure reason: {failure_reason}\nFeedback: {feedback_message}"
        

        data = {
            "robot_pos": [round(x, 3) for x in self.robot_pos] if self.robot_pos is not None else None,
            "objects": {k: [round(x, 3) for x in v] for k, v in self.object_positions.items()} if self.object_positions else None,
            "reward": round(self.reward, 3),
            "done": self.done
        }
        return json.dumps(data, indent=2)

@dataclass
class FeedbackWithError(Feedback):
    skill_type: str = "unknown"
    attempted_action: np.ndarray = None
    waypoints: List[np.ndarray] = None
    waypoint_index: int = -1
    step_index: int = -1
    original_robot_pos: np.ndarray = None
    error_message: str = "An error occurred during the skill execution."

    def __str__(self):

        if self.task and self.robot_pos is None:
            self.robot_pos = self.task.get_observation().gripper_pose

        if self.task and (self.object_positions is None or len(self.object_positions) == 0):
            self.object_positions = get_all_object_positions(self.task)


        if self.waypoints is not None:
            failure_reason, feedback_message = analyze_skill_failure(
                is_skill_done=False,
                skill_type=self.skill_type,
                robot_pos=self.robot_pos, 
                objects=self.object_positions, 
                waypoints=self.waypoints, 
                waypoint_index=self.waypoint_index, 
                attempted_action=self.attempted_action, 
                step_index=self.step_index, 
                original_robot_pos=self.original_robot_pos, 
                msg=self.error_message,
            )
            return f"{failure_reason} - {feedback_message}"
        else:
            return self.error_message


        data = {
            "robot_pos": [round(x, 3) for x in self.robot_pos] if self.robot_pos is not None else None,
            "objects": {k: [round(x, 3) for x in v] for k, v in self.object_positions.items()} if self.object_positions else None,
            "reward": round(self.reward, 3),
            "done": self.done,
            "error": {
                "skill": self.skill_type,
                "attempted_action": [round(x, 3) for x in self.attempted_action] if self.attempted_action is not None else None,
                "waypoint_index": self.waypoint_index,
                "step_index": self.step_index,
                "original_robot_pos": [round(x, 3) for x in self.original_robot_pos] if self.original_robot_pos is not None else None,
                "msg": self.error_message
            }
        }
        return json.dumps(data, indent=2)