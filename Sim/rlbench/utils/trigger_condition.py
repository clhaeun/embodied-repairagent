from .feedback import Feedback
import numpy as np

class TriggerCondition(BaseException):
    def __init__(self, feedback: Feedback):
        self.feedback = feedback
        # Exception 메시지는 __str__() 결과로
        super().__init__(str(feedback))

class SkillFailure(TriggerCondition):
    pass

class PathOutOfWorkspace(TriggerCondition):
    pass

# class SkillFailure(TriggerCondition):
#     """
#     스킬 시퀀스 실행이 실패했을 때
#     - skill_type
#     - reward
#     - done: episode 종료 여부 (bool)
#     - step_index: 어느 단계에서 실패했는지 (int)
#     - message
#     """
#     def __init__(self, message: str, skill_type: str = "unknown", reward: float = 0.0, done: bool = False, step_index: int = -1):
#         super().__init__(message, skill_type, reward, done)
#         self.error_type = "SKILL_FAILURE"
#         self.step_index = step_index
#         self.skill_type = skill_type

# class PathOutOfWorkspace(TriggerCondition):
#     """
#     로봇 팔 가동 범위를 벗어난 경로로 이동하려 할 때
#     main으로 전달할 정보:
#     - message: 에러 메시지
#     - skill_type
#     - attempted_path: 시도하려던 경로 (list of waypoints)
#     - offending_segment: 문제를 일으킨 경로 구간 인덱스 (int)
#     - current_robot_pose: 경로 시도 직전 로봇 포즈 (x,y,z,quat 등)
#     """
#     def __init__(self, message: str, skill_type: str = "unknown", reward: float = 0.0, done: bool = False, attempted_path: list = None, offending_segment: int = -1, current_robot_pose: dict = None):
#         super().__init__(message, skill_type, reward, done)
#         self.error_type = "OUT_OF_WORKSPACE"
#         self.attempted_path = attempted_path if attempted_path is not None else []
#         self.offending_segment = offending_segment
#         self.current_robot_pose = current_robot_pose if current_robot_pose is not None else {}
#         self.skill_code = skill_type

# class ObjectOutOfReach(TriggerCondition):
#     """
#     객체 위치가 가동 범위를 벗어나 회수 불가능한 상태 => 완전 종료
#     - skill_name: 실행 중인 스킬 이름
#     - object_id: 대상 객체 식별자
#     - object_position: 객체의 좌표 (x,y,z)
#     - workspace_limits: 허용 작업 공간 범위 정보 (dict)
#     - distance_to_workspace: 작업 공간 경계까지의 최단 거리 (float)
#     - message: 종료 사유 설명
#     """
#     def __init__(self, message: str, skill_type: str = "unknown"):
#         super().__init__(message, skill_type)
#         self.error_type = "OUT_OF_REACH"