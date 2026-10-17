from enum import Enum
import numpy as np

def normalize_quaternion(q):
    return q / np.linalg.norm(q)

class FailureType(Enum):
    WAYPOINT_FAIL = "WAYPOINT_FAIL"
    GRIPPER_CLOSE_FAIL = "GRIPPER_CLOSE_FAIL"
    HOLD_NOTHING = "HOLD_NOTHING"
    NO_OBJECT = "NO_OBJECT"
    UNKNOWN = "UNKNOWN"

def analyze_skill_failure(is_skill_done: bool, skill_type, robot_pos, objects, waypoints, waypoint_index, attempted_action, step_index, original_robot_pos, msg):
    """
    Analyze the reason for skill execution failure and provide feedback.
    Args:
        skill_type
        robot_pos
        objects
        waypoints
        waypoint_index
        attempted_action
        step_index
    Returns:
        tuple: (failure_reason, feedback_message)
    """
    current_pos = robot_pos[:3]
    current_quat = normalize_quaternion(robot_pos[3:7])

    start_pos = original_robot_pos[:3]
    start_quat = normalize_quaternion(original_robot_pos[3:7])

    object_positions = objects
    error_message = msg

    target_pos = waypoints[-1][:3]
    target_quat = normalize_quaternion(target_pos)
    

    

    failure_reason = FailureType.UNKNOWN.value
    feedback_message = "Unable to identify failure reason."





    if not is_skill_done:

        if "timeout" in error_message.lower() or "exceeded max_steps" in error_message.lower():
            if waypoint_index >= 0 and waypoints:
                failed_waypoint = waypoints[waypoint_index]


                if target_pos is not None:
                    dist_to_target = np.linalg.norm(failed_waypoint - target_pos)
                    if dist_to_target < 0.05:
                        failure_reason = FailureType.WAYPOINT_FAIL.value
                        feedback_message = "Waypoint too close to target center. Adjust waypoint to approach nearby instead by first moving slightly backward, then repositioning the waypoint to approach the target from a short distance away."
                        return failure_reason, feedback_message
                    

                for obj_name, obj_pos in object_positions.items():
                    if obj_name != 'target_object':
                        dist_to_path = np.linalg.norm(np.cross(failed_waypoint - start_pos, obj_pos - start_pos)) / np.linalg.norm(failed_waypoint - start_pos)
                        if dist_to_path < 0.1:
                            failure_reason = FailureType.WAYPOINT_FAIL.value
                            feedback_message = "Blocked by another object. Remove the object or plan an alternative path."
                            return failure_reason, feedback_message

            failure_reason = FailureType.WAYPOINT_FAIL.value
            feedback_message = "Waypoint failure. Reset the planned path entirely or separate the approach into two distinct movements: first move to an intermediate waypoint near the target, then approach directly."


        elif "Gripper did not close properly" in error_message:
            if target_pos is not None:
                dist_to_target = np.linalg.norm(current_pos - target_pos)
                if dist_to_target < 0.02:
                    failure_reason = FailureType.GRIPPER_CLOSE_FAIL.value
                    feedback_message = "Too close to the object for the gripper to close. Slightly retreat backward, increase the distance between the gripper and the object, then retry gripping."
                else:
                    failure_reason = FailureType.GRIPPER_CLOSE_FAIL.value
                    feedback_message = "Failed to close the gripper. Adjust the distance between the gripper and the object slightly, ensuring it's neither too close nor too far, and retry closing."
            else:
                failure_reason = FailureType.GRIPPER_CLOSE_FAIL.value
                feedback_message = "Gripper close failure. Verify the precise position and stability of the target object before retrying. The object may have moved or may not be in the expected position."


    else:
        if skill_type == "pick":
            if target_pos is not None:
                dist_to_target = np.linalg.norm(current_pos - target_pos)
                if dist_to_target > 0.1:
                    failure_reason = FailureType.HOLD_NOTHING.value
                    feedback_message = "Pick attempt too far from the object. Move the gripper closer to the object until the gripper is properly positioned within gripping range, then retry."
                else:
                    failure_reason = FailureType.HOLD_NOTHING.value
                    feedback_message = "Close to object but failed to pick. Precisely realign the orientation of the gripper relative to the target object to match the optimal gripping orientation, and retry picking."
            else:
                failure_reason = FailureType.NO_OBJECT.value
                feedback_message = "Target object position unknown. Please check the name of target object and retry."

    return failure_reason, feedback_message







































