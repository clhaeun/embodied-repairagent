"""
Result dataclasses for Genesis Loop execution.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class SubtaskResult:
    """Individual subtask execution result."""
    subtask_name: str
    obj_name: Optional[str] = None
    target_name: Optional[str] = None
    success: bool = False
    repair_count: int = 0
    skipped: bool = False
    error: Optional[str] = None
    generated_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subtask_name": self.subtask_name,
            "obj_name": self.obj_name,
            "target_name": self.target_name,
            "success": self.success,
            "repair_count": self.repair_count,
            "skipped": self.skipped,
            "error": self.error,
            "generated_code": self.generated_code,
        }


@dataclass
class EpisodeResult:
    """Episode execution result."""
    success: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    error_type: Optional[str] = None
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_llm_time: float = 0.0
    total_llm_tokens: int = 0
    final_code: Optional[str] = None
    subtask_results: List[SubtaskResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "execution_time": self.execution_time,
            "error": self.error,
            "error_type": self.error_type,
            "llm_calls": self.llm_calls,
            "total_llm_time": self.total_llm_time,
            "total_llm_tokens": self.total_llm_tokens,
            "final_code": self.final_code,
            "subtask_results": [s.to_dict() for s in self.subtask_results],
        }


@dataclass
class ExecuteTaskResult:
    """Result of execute_task function."""
    final_code: str
    subtask_results: List[SubtaskResult] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    error_type: Optional[str] = None
