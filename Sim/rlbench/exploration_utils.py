"""
관찰-only 마이크로 탐색 모듈
- 로봇을 움직이지 않고 현재 scene 상태만 관찰
- 7B 모델 친화적 JSON 포맷으로 결과 반환
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor


def compute_static_observation(
    task_name: str,
    scene_info: Dict[str, Any],
    failed_stmt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    현재 scene 상태를 관찰하여 7B 친화적 JSON 반환

    Args:
        task_name: 태스크 이름 (예: MeatOnGrill)
        scene_info: get_scene_info()에서 반환된 scene 정보
        failed_stmt_text: 실패한 statement 코드

    Returns:
        {"probe": "static_goal_check", "targets": {...}, "dist": {...}, "decision": {...}}
        또는 None (관찰 불가 시)
    """

    # MeatOnGrill 태스크에 특화된 관찰
    if task_name == "MeatOnGrill":
        return _observe_meat_on_grill(scene_info, failed_stmt_text)

    # 다른 태스크는 일반 관찰 (추후 확장)
    return _observe_generic_place_task(scene_info, failed_stmt_text)


def _dist_tag(d: float) -> str:
    """거리를 near/medium/far 등급으로 변환"""
    if d < 0.03:
        return "near"
    elif d < 0.08:
        return "medium"
    else:
        return "far"


def _observe_meat_on_grill(
    scene_info: Dict[str, Any],
    failed_stmt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    MeatOnGrill 태스크 전용 관찰 (7B 친화 포맷)

    포함 정보:
    - 좌표: 2자리 소수점 + 단위(m)
    - 관계: d(place_target, success), dz, xy_offset
    - 등급화: near/medium/far 태그
    - 임계값 기반 boolean: inside_success_radius 등
    - chicken/steak 개별 추적
    - grill ≠ success 일 때 명시적 recommendation
    """

    try:
        objects = scene_info.get("objects", [])

        grill_pos = None
        success_pos = None
        chicken_pos = None
        steak_pos = None

        for obj in objects:
            name = obj.get("name", "")
            pos = obj.get("position")

            if pos is None:
                continue

            name_lower = name.lower()
            if "grill" in name_lower and "success" not in name_lower:
                grill_pos = np.array(pos)
            elif "success" in name_lower:
                success_pos = np.array(pos)
            elif "chicken" in name_lower:
                chicken_pos = np.array(pos)
            elif "steak" in name_lower or "meat" in name_lower:
                steak_pos = np.array(pos)

        if grill_pos is None or success_pos is None:
            return None

        # chicken이 조작 대상 (MeatOnGrill 태스크의 목표 물체)
        # chicken이 없으면 steak, 그것도 없으면 gripper 위치를 fallback으로 사용
        meat_pos = chicken_pos if chicken_pos is not None else steak_pos
        if meat_pos is None:
            gripper = scene_info.get("gripper", {})
            meat_pos = np.array(gripper.get("position", [0, 0, 0]))

        # --- 거리 계산 ---
        d_grill_success = float(np.linalg.norm(grill_pos - success_pos))
        d_meat_grill = float(np.linalg.norm(meat_pos - grill_pos))
        d_meat_success = float(np.linalg.norm(meat_pos - success_pos))

        # --- 관계: dz (높이 차이), xy_offset (수평 오프셋) ---
        dz_grill_success = float(success_pos[2] - grill_pos[2])
        xy_offset_grill_success = float(np.linalg.norm(grill_pos[:2] - success_pos[:2]))

        dz_meat_success = float(meat_pos[2] - success_pos[2])
        xy_offset_meat_success = float(np.linalg.norm(meat_pos[:2] - success_pos[:2]))

        # --- 임계값 기반 boolean ---
        success_radius_m = 0.05
        grill_in_success = d_grill_success < success_radius_m
        meat_in_success = d_meat_success < success_radius_m
        meat_closer_to = "success" if d_meat_success < d_meat_grill else "grill"

        target_used = _infer_target_from_code(failed_stmt_text)

        # obj 섹션: chicken과 steak을 개별 추적
        obj_section = {}
        if chicken_pos is not None:
            obj_section["chicken"] = [round(v, 2) for v in chicken_pos.tolist()]
        if steak_pos is not None:
            obj_section["steak"] = [round(v, 2) for v in steak_pos.tolist()]

        # 7B 친화적 JSON: 숫자는 2자리, 단위 명시, 등급 태그, 관계 포함
        observation = {
            "probe": "static_goal_check",
            "manipulated_object": "chicken" if chicken_pos is not None else "steak",
            "targets": {
                "grill": [round(v, 2) for v in grill_pos.tolist()],
                "success": [round(v, 2) for v in success_pos.tolist()],
            },
            "obj": obj_section,
            "dist": {
                "grill_success": {"m": round(d_grill_success, 3), "tag": _dist_tag(d_grill_success)},
                "chicken_grill": {"m": round(d_meat_grill, 3), "tag": _dist_tag(d_meat_grill)},
                "chicken_success": {"m": round(d_meat_success, 3), "tag": _dist_tag(d_meat_success)},
            },
            "relation": {
                "dz_grill_to_success_m": round(dz_grill_success, 3),
                "xy_offset_grill_to_success_m": round(xy_offset_grill_success, 3),
                "dz_chicken_to_success_m": round(dz_meat_success, 3),
                "xy_offset_chicken_to_success_m": round(xy_offset_meat_success, 3),
            },
            "threshold": {
                "success_radius_m": success_radius_m,
                "grill_in_success": grill_in_success,
                "chicken_in_success": meat_in_success,
                "chicken_closer_to": meat_closer_to,
            },
            "target_used": target_used,
        }

        # grill과 success가 떨어져 있으면 명시적 recommendation 추가
        if d_grill_success > 0.08:
            observation["recommendation"] = {
                "place_target": "success",
                "place_pos": [round(v, 2) for v in success_pos.tolist()],
                "hint": "grill center is {:.3f}m from success sensor. Place chicken near success sensor position, not grill center.".format(d_grill_success),
            }

        return observation

    except Exception as e:
        print(f"[EXPLORATION] Failed to observe MeatOnGrill: {e}")
        return None


def _is_grasp_failure(failed_stmt_text: str) -> bool:
    """실패 코드가 pick/grasp/lift 계열인지 판별"""
    code_lower = failed_stmt_text.lower()
    return any(kw in code_lower for kw in ("pick", "grasp", "lift", "close_gripper", "close_ee"))


def _observe_generic_place_task(
    scene_info: Dict[str, Any],
    failed_stmt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    일반적인 place 태스크에 대한 관찰 (7B 친화 포맷)
    (BasketballInHoop, PhoneOnBase 등)

    - place 실패: 물체-success 관계 중심 (그리퍼 거리 제외)
    - pick/grasp 실패: 그리퍼-물체 거리 포함
    """

    try:
        objects = scene_info.get("objects", [])

        success_pos = None
        # 조작 대상 물체 찾기 (success가 아닌 첫 번째 물체)
        manipulated_obj_name = None
        manipulated_obj_pos = None
        # 태스크별 주요 타깃 후보
        named_targets = {}

        for obj in objects:
            name = obj.get("name", "")
            pos = obj.get("position")

            if pos is None:
                continue

            if "success" in name.lower():
                success_pos = np.array(pos)
            else:
                # 태스크별 주요 객체 이름 매칭
                name_lower = name.lower()
                for kw in ("basketball", "phone", "umbrella", "money", "rubbish",
                           "knife", "charger", "ball", "meat", "steak", "lamp", "button"):
                    if kw in name_lower:
                        named_targets[name] = np.array(pos)
                        if manipulated_obj_name is None:
                            manipulated_obj_name = name
                            manipulated_obj_pos = np.array(pos)
                        break

        if success_pos is None:
            return None

        target_used = _infer_target_from_code(failed_stmt_text)
        is_grasp = _is_grasp_failure(failed_stmt_text)

        observation = {
            "probe": "static_goal_check",
            "targets": {
                "success": [round(v, 2) for v in success_pos.tolist()],
            },
            "target_used": target_used,
        }

        # 물체-success 관계 (place 실패 판정에 직결)
        if manipulated_obj_pos is not None:
            d_obj_success = float(np.linalg.norm(manipulated_obj_pos - success_pos))
            dz_obj_success = float(manipulated_obj_pos[2] - success_pos[2])
            xy_offset_obj_success = float(np.linalg.norm(manipulated_obj_pos[:2] - success_pos[:2]))

            success_radius_m = 0.05
            obj_in_success = d_obj_success < success_radius_m

            observation["obj"] = {
                manipulated_obj_name: [round(v, 2) for v in manipulated_obj_pos.tolist()],
            }
            observation["dist"] = {
                "obj_success": {"m": round(d_obj_success, 3), "tag": _dist_tag(d_obj_success)},
            }
            observation["relation"] = {
                "dz_obj_to_success_m": round(dz_obj_success, 3),
                "xy_offset_obj_to_success_m": round(xy_offset_obj_success, 3),
            }
            observation["threshold"] = {
                "success_radius_m": success_radius_m,
                "obj_in_success": obj_in_success,
            }

        # pick/grasp 실패 시에만 그리퍼-물체 거리 추가
        if is_grasp and manipulated_obj_pos is not None:
            gripper = scene_info.get("gripper", {})
            gripper_pos = np.array(gripper.get("position", [0, 0, 0]))
            d_gripper_obj = float(np.linalg.norm(gripper_pos - manipulated_obj_pos))

            observation["gripper_to_obj"] = {
                "d_m": round(d_gripper_obj, 3),
                "tag": _dist_tag(d_gripper_obj),
            }

        return observation

    except Exception:
        return None


def _infer_target_from_code(code: str) -> str:
    """
    실패한 코드에서 어떤 타깃을 사용했는지 추론

    Returns:
        "grill" | "success" | "unknown"
    """
    code_lower = code.lower()

    if "success" in code_lower:
        return "success"
    elif "grill" in code_lower:
        return "grill"
    elif "hoop" in code_lower:
        return "hoop"
    elif "base" in code_lower:
        return "base"
    else:
        return "unknown"


def _decide_recommendation(
    d_grill_success: float,
    d_obj_grill: float,
    d_obj_success: float,
    target_used: str,
) -> tuple[str, str]:
    """
    관찰 결과를 바탕으로 추천 타깃 결정

    Args:
        d_grill_success: grill과 success 사이 거리
        d_obj_grill: 객체와 grill 사이 거리
        d_obj_success: 객체와 success 사이 거리
        target_used: 현재 코드가 사용한 타깃

    Returns:
        (recommend_target, reason) 튜플
        recommend_target: "grill" | "success"
        reason: "grill_success_far" | "near" | "ambiguous"
    """

    # 핵심 휴리스틱:
    # - grill과 success가 멀리 떨어져 있으면 (> 0.1m)
    # - 객체가 success에 더 가까우면 success 추천

    if d_grill_success > 0.1:
        # grill과 success가 명확히 분리됨
        if d_obj_success < d_obj_grill:
            return "success", "grill_success_far"
        else:
            return "grill", "near"

    # grill과 success가 가까우면 현재 타깃 유지
    if d_obj_success < 0.05 and target_used == "success":
        return "success", "near"
    elif d_obj_grill < 0.05 and target_used == "grill":
        return "grill", "near"

    # 애매한 경우 success 추천 (보수적)
    return "success", "ambiguous"


# ============================================================================
# 초경량 모드 (3개 feature만)
# ============================================================================

def compute_minimal_observation(
    task_name: str,
    scene_info: Dict[str, Any],
    failed_stmt_text: str,
) -> Optional[Dict[str, Any]]:
    """
    초경량 관찰 (3개 feature만)
    - d_grill_success
    - target_used
    - recommend_target
    """

    try:
        objects = scene_info.get("objects", [])

        grill_pos = None
        success_pos = None

        for obj in objects:
            name = obj.get("name", "")
            pos = obj.get("position")

            if pos is None:
                continue

            if "grill" in name.lower() and "success" not in name.lower():
                grill_pos = np.array(pos)
            elif "success" in name.lower():
                success_pos = np.array(pos)

        if grill_pos is None or success_pos is None:
            return None

        d_grill_success = float(np.linalg.norm(grill_pos - success_pos))
        target_used = _infer_target_from_code(failed_stmt_text)

        # 간단한 추천 로직
        recommend_target = "success" if d_grill_success > 0.1 else target_used

        return {
            "probe": "static_goal_check",
            "d_grill_success": round(d_grill_success, 3),
            "target_used": target_used,
            "recommend_target": recommend_target
        }

    except Exception:
        return None
