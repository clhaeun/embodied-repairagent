import numpy as np
import time
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np, time

def get_bbox_sizes(obj):
    bb = obj.get_bounding_box()
    dx = bb[1] - bb[0]
    dy = bb[3] - bb[2]
    dz = bb[5] - bb[4]
    return np.array([dx, dy, dz], dtype=np.float64), bb

def _to_world_vec(wa):
    """ 입력된 문자열 또는 임의 벡터를 월드 좌표계의 단위 벡터 v=[x,y,z]로 통일해 반환 """
    axis_map = {'x': np.array([1,0,0], dtype=np.float64),
                'y': np.array([0,1,0], dtype=np.float64),
                'z': np.array([0,0,1], dtype=np.float64)}
    v = axis_map[wa] if isinstance(wa, str) else np.asarray(wa, dtype=np.float64)
    return normalize_vector(v)


def quat_mul(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)


def normalize_quaternion(q):
    """쾌속 정규화 함수"""
    return q / np.linalg.norm(q)

def normalize_vector(v, eps=1e-9):
    v = np.array(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("[normalize_vector] near-zero vector")
    return v / n

def angle_diff(q1, q2):
    """두 쿼터니언 간 회전각 차이 계산"""
    return (R.from_quat(q1).inv() * R.from_quat(q2)).magnitude()