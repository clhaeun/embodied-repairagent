#!/usr/bin/env python3
"""
RLBench Experiment Runner

static/experiment_tasks.yml을 읽어서 태스크별로 에피소드를 순차적으로 실행하고,
각 태스크/서브태스크별 결과를 파일에 기록합니다.

Usage:
    python rlbench_experiment.py
    python rlbench_experiment.py --output_dir results/exp_001
    python rlbench_experiment.py --task_ids task_000 task_001
    python rlbench_experiment.py --difficulty easy
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# rlbench_loop에서 run_task_episode와 EpisodeResult 가져오기
from rlbench_loop import run_task_episode, EpisodeResult
from env import setup_environment, shutdown_environment


# ============================================================================
# NumPy Type Conversion
# ============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """NumPy 타입을 JSON 직렬화 가능한 Python 타입으로 변환"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int_, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ============================================================================
# Result Data Structures
# ============================================================================

class SubtaskResult:
    """서브태스크(에피소드) 실행 결과"""
    def __init__(self):
        self.episode_name: str = ""
        self.success: bool = False
        self.execution_time_sec: float = 0.0
        self.llm_calls: List[Dict[str, Any]] = []
        self.total_llm_time_sec: float = 0.0
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        # Exception 상세 정보
        self.termination_reason: Optional[str] = None  # "success", "exception", "task_failure", "max_steps"
        self.traceback: Optional[str] = None
        self.exception_log: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_name": self.episode_name,
            "success": self.success,
            "execution_time_sec": self.execution_time_sec,
            "llm_calls": self.llm_calls,
            "total_llm_time_sec": self.total_llm_time_sec,
            "error": self.error,
            "error_type": self.error_type,
            "termination_reason": self.termination_reason,
            "traceback": self.traceback,
            "exception_log": self.exception_log,
        }


class TaskResult:
    """태스크 실행 결과 (여러 서브태스크 포함)"""
    def __init__(self):
        self.task_id: str = ""
        self.difficulty: str = ""
        self.subtasks: List[SubtaskResult] = []
        self.success: bool = False  # 모든 서브태스크 성공 시 True
        self.subtask_success_rate: float = 0.0
        self.total_execution_time_sec: float = 0.0
        self.total_llm_time_sec: float = 0.0

    def compute_metrics(self):
        """서브태스크 결과로부터 메트릭 계산"""
        if not self.subtasks:
            return

        success_count = sum(1 for s in self.subtasks if s.success)
        self.subtask_success_rate = success_count / len(self.subtasks)
        self.success = success_count == len(self.subtasks)
        self.total_execution_time_sec = sum(s.execution_time_sec for s in self.subtasks)
        self.total_llm_time_sec = sum(s.total_llm_time_sec for s in self.subtasks)

    def to_dict(self) -> Dict[str, Any]:
        self.compute_metrics()
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "success": self.success,
            "subtask_success_rate": self.subtask_success_rate,
            "total_execution_time_sec": self.total_execution_time_sec,
            "total_llm_time_sec": self.total_llm_time_sec,
            "subtasks": [s.to_dict() for s in self.subtasks],
        }


class ExperimentResult:
    """전체 실험 결과"""
    def __init__(self):
        self.experiment_id: str = ""
        self.start_time: str = ""
        self.end_time: str = ""
        self.config: Dict[str, Any] = {}
        self.tasks: List[TaskResult] = []
        self.overall_success_rate: float = 0.0
        self.overall_subtask_success_rate: float = 0.0

    def compute_metrics(self):
        """전체 메트릭 계산"""
        if not self.tasks:
            return

        for t in self.tasks:
            t.compute_metrics()

        success_count = sum(1 for t in self.tasks if t.success)
        self.overall_success_rate = success_count / len(self.tasks)

        total_subtasks = sum(len(t.subtasks) for t in self.tasks)
        if total_subtasks > 0:
            success_subtasks = sum(
                sum(1 for s in t.subtasks if s.success)
                for t in self.tasks
            )
            self.overall_subtask_success_rate = success_subtasks / total_subtasks

    def to_dict(self) -> Dict[str, Any]:
        self.compute_metrics()
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "overall_success_rate": self.overall_success_rate,
            "overall_subtask_success_rate": self.overall_subtask_success_rate,
            "tasks": [t.to_dict() for t in self.tasks],
        }


# ============================================================================
# File I/O
# ============================================================================

def load_experiment_tasks(yaml_path: str) -> List[Dict[str, Any]]:
    """experiment_tasks.yml 파일 로드"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


def save_result_json(result: Dict[str, Any], output_path: str):
    """결과를 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # NumPy 타입 변환
    result_converted = convert_numpy_types(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_converted, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Result saved to: {output_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[ExperimentResult]:
    """체크포인트 파일이 있으면 로드"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        result = ExperimentResult()
        result.experiment_id = data.get("experiment_id", "")
        result.start_time = data.get("start_time", "")
        result.config = data.get("config", {})

        for task_data in data.get("tasks", []):
            task_result = TaskResult()
            task_result.task_id = task_data.get("task_id", "")
            task_result.difficulty = task_data.get("difficulty", "")

            for subtask_data in task_data.get("subtasks", []):
                subtask_result = SubtaskResult()
                subtask_result.episode_name = subtask_data.get("episode_name", "")
                subtask_result.success = subtask_data.get("success", False)
                subtask_result.execution_time_sec = subtask_data.get("execution_time_sec", 0.0)
                subtask_result.llm_calls = subtask_data.get("llm_calls", [])
                subtask_result.total_llm_time_sec = subtask_data.get("total_llm_time_sec", 0.0)
                subtask_result.error = subtask_data.get("error")
                subtask_result.error_type = subtask_data.get("error_type")
                task_result.subtasks.append(subtask_result)

            result.tasks.append(task_result)

        print(f"[CHECKPOINT] Loaded checkpoint from: {checkpoint_path}")
        return result

    return None


# ============================================================================
# LLM Call Logger
# ============================================================================

class LLMCallLogger:
    """LLM 호출을 실시간으로 로깅하는 클래스"""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # 헤더 작성
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("timestamp,task,subtask,action,duration_sec,num_tokens\n")

    def log(self, call_info: Dict[str, Any]):
        """LLM 호출 정보를 로그 파일에 추가"""
        timestamp = datetime.now().isoformat()
        task = call_info.get("task", "")
        subtask = call_info.get("subtask", "")
        action = call_info.get("action", "")
        duration = call_info.get("duration_sec", 0.0)
        num_tokens = call_info.get("num_tokens", "")  # 토큰 수 (없을 수 있음)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{task},{subtask},{action},{duration:.3f},{num_tokens}\n")

        tokens_str = f", tokens: {num_tokens}" if num_tokens else ""
        print(f"[LLM_CALL] {task}/{subtask} - {action}: {duration:.2f}s{tokens_str}")


# ============================================================================
# Experiment Runner
# ============================================================================
from typing import Any, Dict, Optional

def normalize_termination_reason_from_episode(episode_result) -> str:
    # 0) 원본에 termination_reason이 있으면 그걸 우선 사용
    tr = getattr(episode_result, "termination_reason", None)
    if isinstance(tr, str) and tr.strip():
        # 혹시 다른 문자열이 들어오면 4종으로 매핑
        if tr in ("success", "task_failure", "exception", "unknown"):
            return tr
        # 흔한 케이스 매핑(원하시면 더 추가)
        if tr.lower() in ("fail", "failure", "task_fail", "task_failed"):
            return "task_failure"
        if "except" in tr.lower() or "error" in tr.lower():
            return "exception"
        return "unknown"

    # 1) 성공이면 success
    if getattr(episode_result, "success", False):
        return "success"

    # 2) 에러/트레이스백/exception_log가 있으면 exception
    err = getattr(episode_result, "error", None)
    err_type = getattr(episode_result, "error_type", None)
    tb = getattr(episode_result, "traceback", None)
    exlog = getattr(episode_result, "exception_log", None)

    if err or err_type or tb or (isinstance(exlog, list) and len(exlog) > 0):
        return "exception"

    # 3) 여기까지 왔는데 성공도 아니고 예외도 아니면: task_failure/unknown 중 택
    # run_task_episode가 “미완료 but 예외 없음”을 별도 에러 필드 없이 반환하는 경우 대비
    return "task_failure"



def run_experiment(
    tasks_config: List[Dict[str, Any]],
    output_dir: str,
    mode: str = "step",
    source_robot: str = "panda",
    target_robot: str = "panda",
    max_skill_calls: int = 10,
    max_repair_attempts: int = 5,
    code_source: str = "remote_llm",
    model: str = "ours",
    remote_host: str = "172.17.0.1",
    remote_port: int = 5000,
    task_ids: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    seed: Optional[int] = None,
    variation: int = 0,

) -> ExperimentResult:
    """
    실험 실행 메인 함수

    Args:
        tasks_config: experiment_tasks.yml에서 로드한 태스크 설정
        output_dir: 결과 저장 디렉토리
        mode: 실행 모드 (step 또는 failure)
        source_robot: Source robot type
        target_robot: Target robot type
        max_skill_calls: Maximum number of primitive skill calls
        max_repair_attempts: Maximum number of repair attempts before giving up (default: 5)
        code_source: Where to get task skill code
        model: Model type
        remote_host: Remote LLM server host
        remote_port: Remote LLM server port
        task_ids: 실행할 특정 태스크 ID 리스트 (None이면 전체)
        difficulty: 실행할 난이도 필터 (None이면 전체)
        resume_from_checkpoint: 체크포인트에서 이어서 실행할지 여부

    Returns:
        ExperimentResult: 실험 결과
    """
    # 실험 ID 생성
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 출력 디렉토리 설정
    os.makedirs(output_dir, exist_ok=True)

    # 파일 경로들
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    final_result_path = os.path.join(output_dir, "final_result.json")
    llm_log_path = os.path.join(output_dir, "llm_calls.csv")

    # LLM 로거 생성
    llm_logger = LLMCallLogger(llm_log_path)

    # 체크포인트 로드 시도
    result = None
    completed_tasks = set()

    if resume_from_checkpoint:
        result = load_checkpoint(checkpoint_path)
        if result is not None:
            experiment_id = result.experiment_id
            completed_tasks = {t.task_id for t in result.tasks}
            print(f"[RESUME] Resuming experiment {experiment_id}")
            print(f"[RESUME] Already completed tasks: {completed_tasks}")

    if result is None:
        result = ExperimentResult()
        result.experiment_id = experiment_id
        result.start_time = datetime.now().isoformat()

    # 설정 저장
    result.config = {
        "mode": mode,
        "source_robot": source_robot,
        "target_robot": target_robot,
        "max_skill_calls": max_skill_calls,
        "code_source": code_source,
        "model": model,
        "remote_host": remote_host,
        "remote_port": remote_port,
        "task_ids_filter": task_ids,
        "difficulty_filter": difficulty,
    }

    # 태스크 필터링
    filtered_tasks = tasks_config
    if task_ids:
        filtered_tasks = [t for t in filtered_tasks if t.get("id") in task_ids]
    if difficulty:
        filtered_tasks = [t for t in filtered_tasks if t.get("difficulty") == difficulty]

    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] Starting experiment: {experiment_id}")
    print(f"[EXPERIMENT] Total tasks: {len(filtered_tasks)}")
    print(f"[EXPERIMENT] Output directory: {output_dir}")
    print(f"{'='*60}\n")

    # 각 태스크 실행
    for task_idx, task_config in enumerate(filtered_tasks):
        task_id = task_config.get("id", f"task_{task_idx:03d}")
        task_difficulty = task_config.get("difficulty", "unknown")
        episodes = task_config.get("episodes", [])

        # 이미 완료된 태스크는 스킵
        if task_id in completed_tasks:
            print(f"[SKIP] Task {task_id} already completed")
            continue

        print(f"\n{'='*60}")
        print(f"[TASK {task_idx + 1}/{len(filtered_tasks)}] {task_id} ({task_difficulty})")
        print(f"[TASK] Episodes: {episodes}")
        print(f"{'='*60}\n")

        task_result = TaskResult()
        task_result.task_id = task_id
        task_result.difficulty = task_difficulty

        # 각 에피소드(서브태스크) 실행
        for ep_idx, episode_name in enumerate(episodes):
            print(f"\n[SUBTASK {ep_idx + 1}/{len(episodes)}] {episode_name}")
            print("-" * 40)

            subtask_result = SubtaskResult()
            subtask_result.episode_name = episode_name

            try:
                # run_task_episode 실행
                episode_result: EpisodeResult = run_task_episode(
                    task_name=episode_name,
                    mode=mode,
                    source_robot=source_robot,
                    target_robot=target_robot,
                    max_skill_calls=max_skill_calls,
                    max_repair_attempts=max_repair_attempts,
                    code_source=code_source,
                    model=model,
                    remote_host=remote_host,
                    remote_port=remote_port,
                    llm_call_logger=llm_logger.log,
                    seed=seed,
                    variation=variation,
                )

                # 결과 복사
                subtask_result.success = episode_result.success
                subtask_result.execution_time_sec = episode_result.execution_time
                subtask_result.llm_calls = episode_result.llm_calls
                subtask_result.total_llm_time_sec = sum(
                    c.get("duration_sec", 0) for c in episode_result.llm_calls
                )
                subtask_result.error = episode_result.error
                subtask_result.error_type = episode_result.error_type
                # Exception 상세 정보 복사
                # subtask_result.termination_reason = episode_result.termination_reason
                subtask_result.termination_reason = normalize_termination_reason_from_episode(episode_result)
                subtask_result.traceback = episode_result.traceback
                subtask_result.exception_log = episode_result.exception_log

                # grounded_skills 폴더에 최종 코드 저장
                if episode_result.final_code:
                    grounded_skills_dir = os.path.join(output_dir, "grounded_skills")
                    os.makedirs(grounded_skills_dir, exist_ok=True)

                    # 파일명: task_{task_번호}_{episode_이름}.py
                    # task_id에서 번호 추출 (예: task_000 -> 000)
                    task_num = task_id.replace("task_", "") if task_id.startswith("task_") else task_id
                    skill_filename = f"task_{task_num}_{episode_name}.py"
                    skill_filepath = os.path.join(grounded_skills_dir, skill_filename)

                    with open(skill_filepath, "w", encoding="utf-8") as f:
                        f.write(episode_result.final_code)
                    print(f"  - Grounded skill saved to: {skill_filepath}")

                # LLM에 전달된 프롬프트 저장
                if hasattr(episode_result, 'prompts') and episode_result.prompts:
                    prompts_dir = os.path.join(output_dir, "prompts")
                    os.makedirs(prompts_dir, exist_ok=True)
                    task_num = task_id.replace("task_", "") if task_id.startswith("task_") else task_id
                    prompt_filepath = os.path.join(prompts_dir, f"task_{task_num}_{episode_name}_prompts.txt")
                    with open(prompt_filepath, "w", encoding="utf-8") as f:
                        for i, p in enumerate(episode_result.prompts):
                            f.write(f"{'='*80}\n")
                            f.write(f"[{i+1}] action={p.get('action', '?')}  subtask={p.get('subtask', '?')}\n")
                            f.write(f"{'='*80}\n")
                            f.write(p.get('prompt', '') + "\n\n")
                            # hypothesis 프롬프트/응답 저장
                            hyp_prompt = p.get('hypothesis_prompt')
                            hyp_response = p.get('hypothesis_response')
                            if hyp_prompt or hyp_response:
                                f.write(f"{'-'*80}\n")
                                f.write(f"[{i+1}] HYPOTHESIS PROMPT\n")
                                f.write(f"{'-'*80}\n")
                                f.write((hyp_prompt or '(none)') + "\n\n")
                                f.write(f"{'-'*80}\n")
                                f.write(f"[{i+1}] HYPOTHESIS LLM RESPONSE\n")
                                f.write(f"{'-'*80}\n")
                                f.write((hyp_response or '(none)') + "\n\n")
                    print(f"  - Prompts saved to: {prompt_filepath}")

                status = "SUCCESS" if subtask_result.success else "FAILED"
                print(f"\n[SUBTASK RESULT] {episode_name}: {status}")
                print(f"  - Execution time: {subtask_result.execution_time_sec:.2f}s")
                print(f"  - LLM time: {subtask_result.total_llm_time_sec:.2f}s")
                print(f"  - LLM calls: {len(subtask_result.llm_calls)}")

            except Exception as e:
                import traceback as tb_module
                print(f"[SUBTASK ERROR] {episode_name}: {e}")
                tb_module.print_exc()
                subtask_result.success = False
                subtask_result.error = str(e)
                subtask_result.error_type = type(e).__name__
                subtask_result.traceback = tb_module.format_exc()
                subtask_result.termination_reason = "exception"
                subtask_result.exception_log.append({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": subtask_result.traceback,
                    "phase": "experiment_runner"
                })

            task_result.subtasks.append(subtask_result)

        # 태스크 결과 저장
        task_result.compute_metrics()
        result.tasks.append(task_result)

        print(f"\n[TASK RESULT] {task_id}")
        print(f"  - Success: {task_result.success}")
        print(f"  - Subtask success rate: {task_result.subtask_success_rate:.2%}")
        print(f"  - Total execution time: {task_result.total_execution_time_sec:.2f}s")
        print(f"  - Total LLM time: {task_result.total_llm_time_sec:.2f}s")

        # 체크포인트 저장 (태스크 완료 후마다)
        save_result_json(result.to_dict(), checkpoint_path)

    # 최종 결과 저장
    result.end_time = datetime.now().isoformat()
    result.compute_metrics()
    save_result_json(result.to_dict(), final_result_path)

    # Exception 로그 별도 저장
    exception_log_path = os.path.join(output_dir, "exception_log.json")
    exception_summary = _generate_exception_summary(result)
    save_result_json(exception_summary, exception_log_path)

    # 요약 출력
    print(f"\n{'='*60}")
    print("[EXPERIMENT COMPLETE]")
    print(f"  - Overall task success rate: {result.overall_success_rate:.2%}")
    print(f"  - Overall subtask success rate: {result.overall_subtask_success_rate:.2%}")
    print(f"  - Results saved to: {output_dir}")
    print(f"  - Exception log: {exception_log_path}")
    print(f"{'='*60}\n")

    return result


def _generate_exception_summary(result: ExperimentResult) -> Dict[str, Any]:
    """실험 결과에서 exception 정보만 추출하여 요약 생성"""
    exception_entries = []
    termination_stats = {
        "success": 0,
        "task_failure": 0,
        "exception": 0,
        "unknown": 0,
    }
    error_type_counts = {}

    for task in result.tasks:
        for subtask in task.subtasks:
            # termination_reason 통계
            reason = subtask.termination_reason or "unknown"
            if reason in termination_stats:
                termination_stats[reason] += 1
            else:
                termination_stats["unknown"] += 1

            # error_type 통계
            if subtask.error_type:
                error_type_counts[subtask.error_type] = error_type_counts.get(subtask.error_type, 0) + 1

            # exception 발생한 경우만 상세 기록
            if subtask.error or subtask.exception_log:
                entry = {
                    "task_id": task.task_id,
                    "episode_name": subtask.episode_name,
                    "success": subtask.success,
                    "termination_reason": subtask.termination_reason,
                    "error": subtask.error,
                    "error_type": subtask.error_type,
                    "traceback": subtask.traceback,
                    "exception_log": subtask.exception_log,
                }
                exception_entries.append(entry)

    return {
        "experiment_id": result.experiment_id,
        "total_subtasks": sum(len(t.subtasks) for t in result.tasks),
        "termination_stats": termination_stats,
        "error_type_counts": error_type_counts,
        "exceptions": exception_entries,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RLBench Experiment Runner - Run experiments from experiment_tasks.yml"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        default="./static/experiment_tasks.yml",
        help="Path to experiment_tasks.yml"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/exp_YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["step", "failure"],
        default="step",
        help="Execution mode"
    )
    parser.add_argument(
        "--source_robot", "-s",
        default="panda",
        help="Source robot type (panda, ur5, sawyer, jaco)"
    )
    parser.add_argument(
        "--target_robot", "-r",
        default="ur5",
        help="Target robot type for environment setup"
    )
    parser.add_argument(
        "--max_steps", "-n",
        type=int,
        default=None,
        help="Maximum number of primitive skill calls per episode"
    )
    parser.add_argument(
        "--max_repairs",
        type=int,
        default=5,
        help="Maximum number of repair attempts before giving up (default: 5)"
    )
    parser.add_argument(
        "--code_source", "-c",
        type=str,
        default="remote_llm",
        choices=["static", "local_llm", "remote_llm"],
        help="Where to get task skill code"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ours",
        choices=["code_agent", "ours"],
        help="Model type"
    )
    parser.add_argument(
        "--remote_host",
        type=str,
        default="127.0.0.1",
        help="Remote LLM server host"
    )
    parser.add_argument(
        "--remote_port",
        type=int,
        default=9000,
        help="Remote LLM server port"
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        type=str,
        default=None,
        help="Specific task IDs to run (e.g., task_000 task_001)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "mixed", "hard"],
        default=None,
        help="Filter tasks by difficulty"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (applied to numpy/random in environment setup)"
    )
    parser.add_argument(
        "--variation",
        type=int,
        default=0,
        help="RLBench task variation index"
    )

    args = parser.parse_args()

    # 출력 디렉토리 설정
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./results/exp_{timestamp}"

    # YAML 파일 로드
    if not os.path.exists(args.yaml_path):
        print(f"[ERROR] YAML file not found: {args.yaml_path}")
        sys.exit(1)

    tasks_config = load_experiment_tasks(args.yaml_path)
    print(f"[INFO] Loaded {len(tasks_config)} tasks from {args.yaml_path}")

    # 실험 실행
    run_experiment(
        tasks_config=tasks_config,
        output_dir=args.output_dir,
        mode=args.mode,
        source_robot=args.source_robot,
        target_robot=args.target_robot,
        max_skill_calls=args.max_steps,
        max_repair_attempts=args.max_repairs,
        code_source=args.code_source,
        model=args.model,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        task_ids=args.task_ids,
        difficulty=args.difficulty,
        resume_from_checkpoint=not args.no_resume,
        seed=args.seed,
        variation=args.variation,
    )


if __name__ == "__main__":
    main()



