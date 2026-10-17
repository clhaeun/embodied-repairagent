"""
Background validation module for subtask code.

This module provides background validation of future statements in subtask code,
using projected state to detect precondition violations before execution.
"""

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ours_utils import (
    SkillCallInfo,
    extract_all_skill_calls,
    get_statement_code,
    create_projected_state,
    projected_state_to_scene_info,
    apply_skill_effect,
    basic_validate_skill_call,
)


class BackgroundValidator:
    """
    Background validator for subtask code.

    Validates future skill calls using projected state that accounts
    for the effects of earlier statements.
    """

    def __init__(
        self,
        batch_repair_func: Optional[Callable] = None,
        log_llm_call_func: Optional[Callable] = None,
    ):
        """
        Initialize background validator.

        Args:
            batch_repair_func: Function to call for batch repair requests.
                Signature: (current_code, executed_stmt_texts, invalid_statements, scene, subtask_name) -> Optional[str]
            log_llm_call_func: Function to log LLM calls.
                Signature: (action, subtask_name, duration, num_tokens) -> None
        """
        self.batch_repair_func = batch_repair_func
        self.log_llm_call_func = log_llm_call_func

        # Thread synchronization
        self._lock = threading.Lock()
        self._pending_repair_code: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._last_response_num_tokens: Optional[int] = None

    def start_validation(
        self,
        current_code: str,
        subtask_func_name: str,
        executed_stmt_texts: List[str],
        pre_captured_scene: Dict[str, Any],
    ) -> None:
        """
        Start background validation in a separate thread.

        Args:
            current_code: Current subtask code to validate
            subtask_func_name: Name of the subtask function
            executed_stmt_texts: List of already executed statement texts
            pre_captured_scene: Scene info captured before execution
        """
        self._thread = threading.Thread(
            target=self._validate_future_statements,
            args=(
                current_code,
                subtask_func_name,
                executed_stmt_texts,
                pre_captured_scene,
            ),
            daemon=True,
        )
        self._thread.start()

    def wait_for_completion(self, timeout: float = 1.0) -> None:
        """Wait for background validation to complete."""
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def get_pending_repair(self) -> Optional[str]:
        """
        Get and clear any pending repair code from background validation.

        Returns:
            Repair code if available, None otherwise
        """
        with self._lock:
            code = self._pending_repair_code
            self._pending_repair_code = None
            return code

    def has_pending_repair(self) -> bool:
        """Check if there's a pending repair."""
        with self._lock:
            return self._pending_repair_code is not None

    def _validate_future_statements(
        self,
        current_code: str,
        subtask_func_name: str,
        executed_stmt_texts: List[str],
        pre_captured_scene: Dict[str, Any],
    ) -> None:
        """
        Background thread function to validate future statements.

        This validates future skill calls using PROJECTED state that accounts
        for the effects of earlier statements. Each skill's effect is applied
        to the projected state before validating the next statement.
        """
        try:
            # Try to import validate_skill_preconditions if available
            try:
                from ours_utils.validate_conditions import validate_skill_preconditions
                has_validator = True
            except ImportError:
                has_validator = False

            invalid_statements: List[Dict[str, Any]] = []

            # Extract ALL skill calls including those in compound statements
            skill_calls, src_lines = extract_all_skill_calls(current_code, subtask_func_name)

            if not skill_calls:
                return

            # Create a mutable copy of scene state for projection
            projected_state = create_projected_state(pre_captured_scene)

            for skill_info in skill_calls:
                skill_name = skill_info.skill_name
                call_node = skill_info.call_node
                stmt = skill_info.stmt

                stmt_code = get_statement_code(stmt, src_lines)

                # Convert projected state back to scene_info format for validation
                current_scene = projected_state_to_scene_info(projected_state, pre_captured_scene)

                # Validate preconditions
                if has_validator:
                    validation_result = validate_skill_preconditions(
                        skill_name=skill_name,
                        scene_info=current_scene,
                        call_node=call_node,
                    )

                    if not validation_result.get("valid", True):
                        invalid_statements.append({
                            "line_number": skill_info.lineno,
                            "statement_code": stmt_code,
                            "skill_name": skill_name,
                            "violations": validation_result.get("violations", []),
                            "warnings": validation_result.get("warnings", []),
                            "projected_state": current_scene,
                            "in_conditional": skill_info.in_conditional,
                            "in_loop": skill_info.in_loop,
                            "parent_context": skill_info.parent_context,
                        })
                else:
                    # Basic validation without full validator
                    violations = basic_validate_skill_call(
                        skill_name, call_node, current_scene
                    )
                    if violations:
                        invalid_statements.append({
                            "line_number": skill_info.lineno,
                            "statement_code": stmt_code,
                            "skill_name": skill_name,
                            "violations": violations,
                            "warnings": [],
                            "projected_state": current_scene,
                            "in_conditional": skill_info.in_conditional,
                            "in_loop": skill_info.in_loop,
                            "parent_context": skill_info.parent_context,
                        })

                # Apply skill effect to projected state for next iteration
                apply_skill_effect(projected_state, skill_name, call_node)

            # If we found invalid statements, request batch repair
            if invalid_statements and self.batch_repair_func is not None:
                print(f"[BG_VALIDATION] Found {len(invalid_statements)} invalid statements, requesting repair...")
                for inv in invalid_statements:
                    ctx = f" (in {inv['parent_context']})" if inv.get('parent_context') else ""
                    print(f"  - Line {inv['line_number']}: {inv['skill_name']}{ctx}")

                repair_start = time.time()
                new_code = self.batch_repair_func(
                    current_code=current_code,
                    executed_stmt_texts=executed_stmt_texts,
                    invalid_statements=invalid_statements,
                    scene=pre_captured_scene,
                    subtask_name=subtask_func_name,
                )
                repair_duration = time.time() - repair_start

                if new_code:
                    with self._lock:
                        self._pending_repair_code = new_code
                    print(f"[BG_VALIDATION] Batch repair completed in {repair_duration:.2f}s")
                    if self.log_llm_call_func:
                        self.log_llm_call_func(
                            "bg_batch_repair",
                            subtask_func_name,
                            repair_duration,
                            self._last_response_num_tokens,
                        )
                else:
                    print(f"[BG_VALIDATION] Batch repair returned no code change")

        except Exception as e:
            print(f"[BG_VALIDATION] Background validation error: {e}")


def create_background_validator(
    batch_repair_func: Optional[Callable] = None,
    log_llm_call_func: Optional[Callable] = None,
) -> BackgroundValidator:
    """
    Factory function to create a BackgroundValidator.

    Args:
        batch_repair_func: Function for batch repair requests
        log_llm_call_func: Function for logging LLM calls

    Returns:
        BackgroundValidator instance
    """
    return BackgroundValidator(
        batch_repair_func=batch_repair_func,
        log_llm_call_func=log_llm_call_func,
    )
