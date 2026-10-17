from .feedback import Feedback
import numpy as np

class TriggerCondition(BaseException):
    def __init__(self, feedback: Feedback):
        self.feedback = feedback

        super().__init__(str(feedback))

class SkillFailure(TriggerCondition):
    pass

class PathOutOfWorkspace(TriggerCondition):
    pass














































