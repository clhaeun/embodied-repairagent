RUN_SKILL_INTERFACE = '''
def run_skill(env, task, descriptions=None, obs=None, variations_index: int = 0):

'''

COMMON_HEADER = '''
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env_utils import quat_mul, get_bbox_sizes, normalize_vector, normalize_quaternion
'''

PANDA_HEADER = '''
from skill_code import pick, place, move, align_to_quaternion, open_gripper, close_gripper, push
'''

UR5_HEADER = '''
from skill_code_ur5 import ur5_move_to, ur5_grasp_at, ur5_release_at, ur5_align_gripper, close_ur5_ee, open_ur5_ee
'''

JACO_HEADER = '''

'''

SAWYER_HEADER = '''
from skill_code_sawyer import sawyer_move_to, sawyer_align_gripper, sawyer_open_gripper, sawyer_close_gripper, sawyer_pick, sawyer_place
'''

HEADER = {
    'panda': COMMON_HEADER + PANDA_HEADER,
    'ur5': COMMON_HEADER + UR5_HEADER,
    'jaco': COMMON_HEADER + JACO_HEADER,
    'sawyer': COMMON_HEADER + SAWYER_HEADER,
}