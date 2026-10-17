#!/usr/bin/env python3
"""
Genesis Experiment Runner

Reads static/experiment_tasks.yml and executes each task once.
Records results for each task.

Usage:
    python genesis_experiment.py
    python genesis_experiment.py --output_dir results/exp_001
    python genesis_experiment.py --source_robot panda --target_robot suction
    python genesis_experiment.py --task_ids task_000 task_001
    python genesis_experiment.py --difficulty easy
    python genesis_experiment.py --seed 42
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from genesis_loop import run_task_episode, EpisodeResult






def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"[SEED] Random seed set to: {seed}")






def convert_numpy_types(obj: Any) -> Any:
    """Convert NumPy types to JSON-serializable Python types."""
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






class TaskResult:
    """Task execution result."""
    def __init__(self):
        self.task_id: str = ""
        self.task_name: str = ""
        self.difficulty: str = ""
        self.description: str = ""
        self.success: bool = False
        self.execution_time_sec: float = 0.0
        self.llm_calls: List[Dict[str, Any]] = []
        self.total_llm_time_sec: float = 0.0
        self.total_llm_tokens: int = 0
        self.error: Optional[str] = None
        self.error_type: Optional[str] = None
        self.subtask_results: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "difficulty": self.difficulty,
            "description": self.description,
            "success": self.success,
            "execution_time_sec": self.execution_time_sec,
            "llm_calls": self.llm_calls,
            "total_llm_time_sec": self.total_llm_time_sec,
            "total_llm_tokens": self.total_llm_tokens,
            "error": self.error,
            "error_type": self.error_type,
            "subtask_results": self.subtask_results,
            "subtask_summary": self._compute_subtask_summary(),
        }

    def _compute_subtask_summary(self) -> Dict[str, Any]:
        """Compute subtask-level summary statistics."""
        if not self.subtask_results:
            return {}

        total = len(self.subtask_results)
        successful = sum(1 for s in self.subtask_results if s.get("success", False))
        skipped = sum(1 for s in self.subtask_results if s.get("skipped", False))
        total_repairs = sum(s.get("repair_count", 0) for s in self.subtask_results)

        return {
            "total_subtasks": total,
            "successful_subtasks": successful,
            "skipped_subtasks": skipped,
            "failed_subtasks": total - successful - skipped,
            "total_repairs": total_repairs,
            "subtask_success_rate": successful / total if total > 0 else 0.0,
        }


class ExperimentResult:
    """Experiment result."""
    def __init__(self):
        self.experiment_id: str = ""
        self.start_time: str = ""
        self.end_time: str = ""
        self.config: Dict[str, Any] = {}
        self.tasks: List[TaskResult] = []
        self.overall_success_rate: float = 0.0
        self.overall_subtask_success_rate: float = 0.0
        self.total_subtasks: int = 0
        self.successful_subtasks: int = 0

    def compute_metrics(self):
        """Compute overall metrics."""
        if not self.tasks:
            return


        success_count = sum(1 for t in self.tasks if t.success)
        self.overall_success_rate = success_count / len(self.tasks)


        total_subtasks = 0
        successful_subtasks = 0
        for task in self.tasks:
            for subtask in task.subtask_results:
                total_subtasks += 1
                if subtask.get("success", False):
                    successful_subtasks += 1

        self.total_subtasks = total_subtasks
        self.successful_subtasks = successful_subtasks
        self.overall_subtask_success_rate = (
            successful_subtasks / total_subtasks if total_subtasks > 0 else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        self.compute_metrics()
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "config": self.config,
            "overall_success_rate": self.overall_success_rate,
            "overall_subtask_success_rate": self.overall_subtask_success_rate,
            "total_tasks": len(self.tasks),
            "successful_tasks": sum(1 for t in self.tasks if t.success),
            "total_subtasks": self.total_subtasks,
            "successful_subtasks": self.successful_subtasks,
            "tasks": [t.to_dict() for t in self.tasks],
        }






def load_experiment_tasks(yaml_path: str) -> List[Dict[str, Any]]:
    """Load experiment_tasks.yml file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tasks", [])


def save_result_json(result: Dict[str, Any], output_path: str):
    """Save result to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_converted = convert_numpy_types(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_converted, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Result saved to: {output_path}")


def load_checkpoint(checkpoint_path: str) -> Optional[ExperimentResult]:
    """Load checkpoint file if it exists."""
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
            task_result.task_name = task_data.get("task_name", "")
            task_result.difficulty = task_data.get("difficulty", "")
            task_result.description = task_data.get("description", "")
            task_result.success = task_data.get("success", False)
            task_result.execution_time_sec = task_data.get("execution_time_sec", 0.0)
            task_result.llm_calls = task_data.get("llm_calls", [])
            task_result.total_llm_time_sec = task_data.get("total_llm_time_sec", 0.0)
            task_result.error = task_data.get("error")
            task_result.error_type = task_data.get("error_type")
            task_result.subtask_results = task_data.get("subtask_results", [])
            result.tasks.append(task_result)

        print(f"[CHECKPOINT] Loaded checkpoint from: {checkpoint_path}")
        return result

    return None






class LLMCallLogger:
    """Real-time LLM call logging."""

    def __init__(self, log_file: str):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)


        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("timestamp,task,action,duration_sec,num_tokens\n")

    def log(self, call_info: Dict[str, Any]):
        """Log LLM call info to file."""
        timestamp = datetime.now().isoformat()
        task = call_info.get("task", "")
        action = call_info.get("action", "")
        duration = call_info.get("duration_sec", 0.0)
        num_tokens = call_info.get("num_tokens", "")

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{timestamp},{task},{action},{duration:.3f},{num_tokens}\n")

        tokens_str = f", tokens: {num_tokens}" if num_tokens else ""
        print(f"[LLM_CALL] {task} - {action}: {duration:.2f}s{tokens_str}")






def run_experiment(
    tasks_config: List[Dict[str, Any]],
    output_dir: str,
    source_robot: str = "panda",
    target_robot: str = "suction",
    max_subtask_repairs: int = 5,
    code_source: str = "static",
    model: str = "code_agent",
    remote_host: str = "127.0.0.1",
    remote_port: int = 9000,
    task_ids: Optional[List[str]] = None,
    difficulty: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    show_viewer: bool = False,
    skip_env: bool = False,
    code_gen_only: bool = False,
    seed: Optional[int] = None,


) -> ExperimentResult:
    """
    Main experiment execution function.

    Args:
        tasks_config: Task configuration from experiment_tasks.yml
        output_dir: Output directory for results
        source_robot: Source robot type ("panda", "robotiq", or "suction")
        target_robot: Target robot type ("panda" or "suction")
        max_subtask_repairs: Maximum repair attempts
        code_source: Code source ("static" or "remote_llm")
        model: Model type for code generation
        remote_host: Remote LLM server host
        remote_port: Remote LLM server port
        task_ids: Specific task IDs to run (None = all)
        difficulty: Filter by difficulty (None = all)
        resume_from_checkpoint: Whether to resume from checkpoint
        show_viewer: Whether to show Genesis viewer
        skip_env: If True, skip Genesis environment initialization
        seed: Random seed for reproducibility (None = no seed)
        enable_background_validation: If True, enable background validation in ours mode

    Returns:
        ExperimentResult: Experiment results
    """

    if seed is not None:
        set_seed(seed)


    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    os.makedirs(output_dir, exist_ok=True)


    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    final_result_path = os.path.join(output_dir, "final_result.json")
    llm_log_path = os.path.join(output_dir, "llm_calls.csv")


    llm_logger = LLMCallLogger(llm_log_path)


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


    result.config = {
        "source_robot": source_robot,
        "target_robot": target_robot,
        "max_subtask_repairs": max_subtask_repairs,
        "code_source": code_source,
        "model": model,
        "remote_host": remote_host,
        "remote_port": remote_port,
        "task_ids_filter": task_ids,
        "difficulty_filter": difficulty,
        "seed": seed,
    }


    filtered_tasks = tasks_config
    if task_ids:
        filtered_tasks = [t for t in filtered_tasks if t.get("id") in task_ids]
    if difficulty:
        filtered_tasks = [t for t in filtered_tasks if t.get("difficulty") == difficulty]

    print(f"\n{'='*60}")
    print(f"[EXPERIMENT] Starting experiment: {experiment_id}")
    print(f"[EXPERIMENT] Total tasks: {len(filtered_tasks)}")
    print(f"[EXPERIMENT] Source robot: {source_robot}")
    print(f"[EXPERIMENT] Target robot: {target_robot}")
    print(f"[EXPERIMENT] Code source: {code_source}")
    print(f"[EXPERIMENT] Seed: {seed if seed is not None else 'None (random)'}")
    print(f"[EXPERIMENT] Output directory: {output_dir}")
    print(f"{'='*60}\n")


    for task_idx, task_config in enumerate(filtered_tasks):
        task_id = task_config.get("id", f"task_{task_idx:03d}")
        task_name = task_config.get("task_name", "")
        task_difficulty = task_config.get("difficulty", "unknown")
        task_description = task_config.get("description", "")


        if task_id in completed_tasks:
            print(f"[SKIP] Task {task_id} already completed")
            continue

        print(f"\n{'='*60}")
        print(f"[TASK {task_idx + 1}/{len(filtered_tasks)}] {task_id}: {task_name}")
        print(f"[TASK] Difficulty: {task_difficulty}")
        print(f"[TASK] Description: {task_description}")
        print(f"{'='*60}\n")

        task_result = TaskResult()
        task_result.task_id = task_id
        task_result.task_name = task_name
        task_result.difficulty = task_difficulty
        task_result.description = task_description

        try:

            episode_result: EpisodeResult = run_task_episode(
                task_name=task_name,
                source_robot=source_robot,
                target_robot=target_robot,
                max_subtask_repairs=max_subtask_repairs,
                code_source=code_source,
                model=model,
                remote_host=remote_host,
                remote_port=remote_port,
                show_viewer=show_viewer,
                llm_call_logger=llm_logger.log,
                code_gen_only=code_gen_only,
                skip_env=skip_env,


            )


            task_result.success = episode_result.success
            task_result.execution_time_sec = episode_result.execution_time
            task_result.llm_calls = episode_result.llm_calls
            task_result.total_llm_time_sec = sum(
                c.get("duration_sec", 0) for c in episode_result.llm_calls
            )
            task_result.total_llm_tokens = episode_result.total_llm_tokens
            task_result.error = episode_result.error
            task_result.error_type = episode_result.error_type

            task_result.subtask_results = [
                s.to_dict() for s in episode_result.subtask_results
            ]


            if episode_result.final_code:
                grounded_tasks_dir = os.path.join(output_dir, f"tasks_{target_robot}")
                os.makedirs(grounded_tasks_dir, exist_ok=True)

                task_num = task_id.replace("task_", "") if task_id.startswith("task_") else task_id
                task_filename = f"{task_num}_{task_name}.py"
                task_filepath = os.path.join(grounded_tasks_dir, task_filename)

                with open(task_filepath, "w", encoding="utf-8") as f:
                    f.write(episode_result.final_code)
                print(f"  - Grounded task saved to: {task_filepath}")

            status = "SUCCESS" if task_result.success else "FAILED"
            print(f"\n[TASK RESULT] {task_name}: {status}")
            print(f"  - Execution time: {task_result.execution_time_sec:.2f}s")
            print(f"  - LLM time: {task_result.total_llm_time_sec:.2f}s")
            print(f"  - LLM calls: {len(task_result.llm_calls)}")

        except Exception as e:
            import traceback
            print(f"[TASK ERROR] {task_name}: {e}")
            traceback.print_exc()
            task_result.success = False
            task_result.error = str(e)
            task_result.error_type = type(e).__name__

        result.tasks.append(task_result)


        save_result_json(result.to_dict(), checkpoint_path)


    result.end_time = datetime.now().isoformat()
    result.compute_metrics()
    save_result_json(result.to_dict(), final_result_path)


    print(f"\n{'='*60}")
    print("[EXPERIMENT COMPLETE]")
    print(f"  - Overall success rate: {result.overall_success_rate:.2%}")
    print(f"  - Successful: {sum(1 for t in result.tasks if t.success)}/{len(result.tasks)}")
    print(f"  - Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    return result






def main():
    parser = argparse.ArgumentParser(
        description="Genesis Experiment Runner - Run experiments from experiment_tasks.yml"
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
        "--source_robot", "-s",
        choices=["panda", "suction", "robotiq"],
        default="panda",
        help="Source robot type (for reference code)"
    )
    parser.add_argument(
        "--target_robot", "-r",
        choices=["panda", "suction", "robotiq"],
        default="robotiq",
        help="Target robot type (for execution)"
    )
    parser.add_argument(
        "--code_source", "-c",
        choices=["static", "remote_llm"],
        default="remote_llm",
        help="Where to get task code"
    )
    parser.add_argument(
        "--model",
        choices=["code_agent", "ours"],
        default="code_agent",
        help="Model type for code generation"
    )
    parser.add_argument(
        "--max_repairs",
        type=int,
        default=5,
        help="Maximum repair attempts (default: 5)"
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
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter tasks by difficulty"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Do not resume from checkpoint"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show Genesis viewer"
    )
    parser.add_argument(
        "--skip_env",
        action="store_true",
        help="Skip Genesis environment initialization (faster code generation)"
    )
    parser.add_argument(
        "--code_gen_only",
        action="store_true",
        help="code_gen_only = False"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None = no seed)"
    )
    """
    parser.add_argument(
        "--no_bg_validation",
        action="store_true",
        help="Disable background validation in ours mode"
    )
    parser.add_argument(
        "--enable_code_cache",
        action="store_true",
        help="Enable subtask code caching (default: disabled)"
    )
    """

    args = parser.parse_args()


    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./results/exp_{timestamp}"


    if not os.path.exists(args.yaml_path):
        print(f"[ERROR] YAML file not found: {args.yaml_path}")
        sys.exit(1)

    tasks_config = load_experiment_tasks(args.yaml_path)
    print(f"[INFO] Loaded {len(tasks_config)} tasks from {args.yaml_path}")


    run_experiment(
        tasks_config=tasks_config,
        output_dir=args.output_dir,
        source_robot=args.source_robot,
        target_robot=args.target_robot,
        max_subtask_repairs=args.max_repairs,
        code_source=args.code_source,
        model=args.model,
        remote_host=args.remote_host,
        remote_port=args.remote_port,
        task_ids=args.task_ids,
        difficulty=args.difficulty,
        resume_from_checkpoint=not args.no_resume,
        show_viewer=args.viewer,
        skip_env=args.skip_env,
        code_gen_only=args.code_gen_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
