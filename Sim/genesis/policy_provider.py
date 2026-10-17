"""
주석 처리하고 결과 내기
        # Cache the generated code
        self._code_cache[subtask_name] = code
        print(f"[CACHE SAVE] Cached code for {subtask_name}")

Policy Provider for Genesis Experiment.

Handles code generation and repair via remote LLM server.
"""

import os
import ast
import time
import threading
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from result_types import SubtaskResult
from exceptions import SubtaskFailure, SubtaskNotfinished, SubtaskSkip

from remote_llm_socket_client import RemoteLLMSocketClient

from postprocess import postprocess_initial_code, postprocess_continuation_code, debug_print_next_statements, merge_continuation_into_subtask
from prompts.prompt_utils import SUBTASK_SIGNATURES
from scene_info import get_scene_info
from prompts.prompt_utils import get_prompt

from ast_policy import parse_code_bundle

import traceback


def preprocess_reference_code(code: str) -> str:
    """
    Reference code에서 첫 번째(top-level) 함수의 body만 추출해서 그대로 반환합니다.
    - import/from 제거할 필요 없음(함수 body만 뽑기 때문)
    - 라인 사이에 # 삽입 없음
    - docstring 포함(원하시면 제거 옵션도 아래에 제공)
    """
    if not code or not code.strip():
        return ""

    src = code.strip() + "\n"

    try:
        tree = ast.parse(src)
    except Exception:

        return _fallback_extract_body_only(src)

    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            fn = node
            break
    if fn is None or not fn.body:
        return ""

    lines = src.splitlines(keepends=True)
    start = fn.body[0].lineno - 1
    end = fn.body[-1].end_lineno
    body = "".join(lines[start:end])

    # body를 "그대로" 쓰되, 최소 들여쓰기만 제거해서 깔끔하게
    return textwrap.dedent(body).rstrip()



def sha1(code: str) -> str:
    """Compute SHA1 hash of code string."""
    import hashlib
    return hashlib.sha1(code.encode('utf-8')).hexdigest()

"""
def extract_module_docstring(py_src: str) -> str:
    try:
        m = ast.parse(py_src)
        ds = ast.get_docstring(m) or ""
        return ds.strip()
    except Exception:
        return ""
"""

# skill_call = step
"""
class TraceRecorder:
    def __init__(self):
        self.executed_skill_texts: list[str] = []
        self.failed_skill_text: str = ""
        self.last_step: dict = {}
        self.skill_call_count: int = 0

    def record_call(self, skill_name: str, call_text: str):
        self.skill_call_count += 1
        self.executed_skill_texts.append(call_text)
        self.last_step = {
            "step_index": len(self.executed_skill_texts),   # 1-based “몇 번째 skill”
            "skill_call_count": self.skill_call_count,
            "statement": call_text,
            "skill_name": skill_name,
            "success": True,
        }

    def record_failure(self, skill_name: str, call_text: str, exc: Exception):
        # record_call이 이미 append 했으므로 +1 하면 안 맞습니다.
        self.failed_skill_text = call_text
        self.last_step = {
            "step_index": len(self.executed_skill_texts),   # 실패한 그 호출 index
            "skill_call_count": self.skill_call_count,
            "statement": call_text,
            "skill_name": skill_name,
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
"""
class TraceRecorder:
    def __init__(self):
        self.executed_skill_texts = []
        self.failed_skill_text = ""
        self.skill_call_count = 0

    def record_success(self, skill_name: str, call_text: str):
        self.skill_call_count += 1
        self.executed_skill_texts.append(call_text)

    def record_failure(self, skill_name: str, call_text: str, exc: Exception):
        self.failed_skill_text = call_text



class PolicyProvider:
    """
    Policy provider for Genesis task execution.

    Supports:
      - static: Load pre-written task code from file
      - remote_llm: Request code generation/repair from remote server
    """

    def __init__(
        self,
        mode: str,                 # "static" | "remote_llm"
        task_name: str,
        model: str = "code_agent",
        initial_prompt: str = "",
        remote_host: str = "127.0.0.1",
        remote_port: int = 9000,
        object_names: list = None,
        descriptions: str = "",
        scene_info: dict = None,
        source_robot: str = "panda",
        target_robot: str = "suction",
        code_gen_only: bool = False,
        llm_call_logger: Optional[callable] = None,
    ):
        self.mode = mode
        self.task_name = task_name
        self.model = model
        self.initial_prompt = initial_prompt
        # 넘겨받는 intial_prompt 무시, reference code에서 description 사용
        self.object_names = object_names or []
        self.descriptions = descriptions
        self.scene_info = scene_info
        self.source_robot = source_robot
        self.target_robot = target_robot
        self.code_gen_only = code_gen_only
        self.last_response_num_tokens: Optional[int] = None
        self._llm_call_logger = llm_call_logger

        # Tracking fields for LLM calls and execution metrics
        self.llm_calls: List[Dict[str, Any]] = []
        self.total_llm_tokens: int = 0
        self.total_llm_time: float = 0.0
        self.total_failure_count: int = 0
        self.total_repair_count: int = 0

        self.client: Optional[RemoteLLMSocketClient] = None
        self.bg_client: Optional[RemoteLLMSocketClient] = None
        self._bg_client_lock = threading.Lock()
        self._remote_host = remote_host
        self._remote_port = remote_port

        self.output_path = None
        self.reference_code = None

        # Cache for generated subtask code (subtask_name -> code)
        # self._code_cache: Dict[str, str] = {}
        self._ref_interface_cache: dict[str, str] = {}


        if self.mode == "remote_llm":
            self.client = RemoteLLMSocketClient(
                remote_host,
                remote_port,
                connect_timeout=30.0,
                recv_timeout=600.0
            )
            print(f"[INFO] Connecting to remote LLM server at {remote_host}:{remote_port}...")
            self.client.connect()
            print(f"[INFO] Successfully connected to {remote_host}:{remote_port}")

    def close(self):
        """Close remote connection."""
        if self.client:
            self.client.close()
            self.client = None
        if self.bg_client:
            self.bg_client.close()
            self.bg_client = None

    @staticmethod
    def _wrap_skill(fn, skill_name: str, tracer: TraceRecorder):
        def _fmt_arg(a):
            # 너무 정보 많은 repr 방지
            import numpy as np

            if a is None:
                return "None"
            if isinstance(a, (bool, int, float)):
                return repr(a)
            if isinstance(a, str):
                return repr(a)
            if isinstance(a, (list, tuple)):
                # 너무 길면 자르기
                xs = list(a)
                if len(xs) > 6:
                    xs = xs[:6] + ["..."]
                return "[" + ", ".join(_fmt_arg(x) for x in xs) + "]" if isinstance(a, list) \
                    else "(" + ", ".join(_fmt_arg(x) for x in xs) + ")"
            if isinstance(a, dict):
                # dict도 너무 길면 키 일부만
                items = list(a.items())
                if len(items) > 6:
                    items = items[:6] + [("...", "...")]
                return "{" + ", ".join(f"{_fmt_arg(k)}: {_fmt_arg(v)}" for k, v in items) + "}"
            if isinstance(a, np.ndarray):
                # 배열은 shape만
                return f"np.ndarray(shape={a.shape}, dtype={a.dtype})"

            # env/task/object 같은 건 repr 금지
            tname = type(a).__name__
            return f"<{tname}>"
        def _fmt_kwargs(kwargs):
            if not kwargs:
                return ""
            # 핵심 키 우선 정렬 (원하시는 키로 조정 가능)
            key_order = ["pointing_to", "depth", "lift_clearance", "offset", "direction", "amount", "velocity"]
            keys = sorted(kwargs.keys(), key=lambda k: (k not in key_order, key_order.index(k) if k in key_order else k))
            parts = []
            for k in keys:
                parts.append(f"{k}={_fmt_arg(kwargs[k])}")
            # 너무 길면 자르기
            s = ", ".join(parts)
            if len(s) > 200:
                s = s[:200] + "..."
            return s

        def wrapped(*args, **kwargs):
            # 1) 첫 인자는 env로 가정하고 "env"로 고정 (repr 금지)
            call_args = ["env"]

            # 2) 나머지 positional 인자 안전 포맷
            #    (너무 많으면 자르기)
            for a in args[1:4]:
                call_args.append(_fmt_arg(a))
            if len(args) > 4:
                call_args.append("...")

            # 3) kwargs 안전 포맷
            kw_text = _fmt_kwargs(kwargs)
            if kw_text:
                call_text = f"{skill_name}({', '.join(call_args)}, {kw_text})"
            else:
                call_text = f"{skill_name}({', '.join(call_args)})"

            try:
                out = fn(*args, **kwargs)
                tracer.record_success(skill_name, call_text)
                return out
            except Exception as exc:
                tracer.record_failure(skill_name, call_text, exc)
                raise
        return wrapped

    @dataclass
    class RefFuncInfo:
        signature_line: str
        docstring: str
        body: str

    def _extract_ref_func_info(self, full_code: str, func_name: str) -> "PolicyProvider.RefFuncInfo | None":
        if not full_code or not full_code.strip():
            return None

        src = full_code.strip() + "\n"
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None

        lines = src.splitlines(keepends=True)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                sig = lines[node.lineno - 1].rstrip("\n")
                ds = (ast.get_docstring(node) or "").strip()

                if node.body:
                    start = node.body[0].lineno - 1
                    end = node.body[-1].end_lineno
                    body_txt = textwrap.dedent("".join(lines[start:end])).rstrip()
                else:
                    body_txt = ""

                return self.RefFuncInfo(signature_line=sig, docstring=ds, body=body_txt)

        return None

    # =============================================================================
    # AST Helpers for Extracting Skill Calls from Subtask Code
    # =============================================================================

    # Genesis primitive skill names that can be validated
    GENESIS_PRIMITIVE_SKILLS = {
        "move_gripper_to", "move_to_position", "move_parallel", "rotate_gripper",
        "open_gripper", "close_gripper", "pick", "place",
        "grasp_handle", "release_handle",
        "activate_vacuum", "deactivate_vacuum",
        "attach_vacuum_handle", "detach_vacuum_handle",
        "open_robotiq85", "close_robotiq85",
        "pick_robotiq85", "place_robotiq85",
        "grasp_handle_robotiq85", "release_handle_robotiq85",
    }


    class SubtaskFunctionExtractor(ast.NodeVisitor):
        """Extract the body of a subtask function from parsed AST."""

        def __init__(self, func_name: str):
            self.func_name = func_name
            self.function_body: List[ast.stmt] = []
            self.found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node.name == self.func_name:
                self.function_body = node.body
                self.found = True
            self.generic_visit(node)


    def extract_skill_call_info(self, stmt: ast.stmt) -> Tuple[Optional[str], Optional[ast.Call]]:
        """
        Extract skill call information from an AST statement.

        Returns:
            (skill_name, call_node) or (None, None) if not a skill call
        """
        # Case 1: Bare expression statement with function call
        # e.g., move_gripper_to(env, "cube")
        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name):
                    return call.func.id, call
                elif isinstance(call.func, ast.Attribute):
                    return call.func.attr, call

        # Case 2: Assignment with function call
        # e.g., result = move_gripper_to(env, "cube")
        if isinstance(stmt, ast.Assign):
            if isinstance(stmt.value, ast.Call):
                call = stmt.value
                if isinstance(call.func, ast.Name):
                    return call.func.id, call
                elif isinstance(call.func, ast.Attribute):
                    return call.func.attr, call

        return None, None


    def get_statement_code(self, stmt: ast.stmt, src_lines: List[str]) -> str:
        """
        Get the source code text for an AST statement.

        Args:
            stmt: AST statement node
            src_lines: List of source code lines

        Returns:
            Source code text for the statement
        """
        try:
            start_line = stmt.lineno - 1  # 0-indexed
            end_line = getattr(stmt, 'end_lineno', stmt.lineno) - 1

            if 0 <= start_line < len(src_lines):
                if start_line == end_line:
                    return src_lines[start_line].strip()
                else:
                    lines = src_lines[start_line:end_line + 1]
                    return '\n'.join(line.rstrip() for line in lines)
        except Exception:
            pass

        return f"<line {stmt.lineno}>"



    def is_primitive_skill(self, skill_name: str) -> bool:
        """Check if a skill name is a Genesis primitive skill."""
        return skill_name.lower() in {s.lower() for s in GENESIS_PRIMITIVE_SKILLS}


    # ========================================================================
    # Metrics
    # ========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this provider instance."""
        return {
            "total_llm_tokens": self.total_llm_tokens,
            "total_llm_time": self.total_llm_time,
            "total_failure_count": self.total_failure_count,
            "total_repair_count": self.total_repair_count,
            "llm_calls": self.llm_calls,
        }

    def reset_metrics(self) -> None:
        """Reset all execution metrics."""
        self.llm_calls = []
        self.total_llm_tokens = 0
        self.total_llm_time = 0.0
        self.total_failure_count = 0
        self.total_repair_count = 0

    def _log_llm_call(
        self,
        action: str,
        subtask_name: str,
        duration: float,
        num_tokens: Optional[int] = None,
    ) -> None:
        """Log an LLM call for tracking."""
        call_info = {
            "task": self.task_name,
            "subtask": subtask_name,
            "action": action.lower(),  # Lowercase for evaluate_genesis.py compatibility
            "duration_sec": duration,
        }
        if num_tokens is not None:
            call_info["num_tokens"] = num_tokens
            self.total_llm_tokens += num_tokens
        self.total_llm_time += duration
        self.llm_calls.append(call_info)
        print(f"[LLM_CALL] {action} for {subtask_name}: {duration:.2f}s, tokens={num_tokens}")

        # Call external logger callback if provided
        if self._llm_call_logger is not None:
            try:
                self._llm_call_logger(call_info)
            except Exception as e:
                print(f"[WARN] LLM call logger failed: {e}")


    # ========================================================================
    # File I/O
    # ========================================================================

    def _save_subtask(self, subtask_name: str, code: str) -> None:
        """Save repaired subtask code to file."""
        # Fix common indentation issues where 'def' lines have leading spaces
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            # If line starts with spaces followed by 'def ', move to column 0
            stripped = line.lstrip()
            if stripped.startswith('def ') and line != stripped:
                # This is an indented def statement - move it to column 0
                print(f"[FIX] Correcting indentation for: {stripped[:50]}...")
                fixed_lines.append(stripped)
            else:
                fixed_lines.append(line)
        code = '\n'.join(fixed_lines)

        subtask_path = f"./subtasks_{self.target_robot}/{subtask_name}.py"
        os.makedirs(os.path.dirname(subtask_path), exist_ok=True)
        with open(subtask_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"[INFO] Saved repaired subtask to: {subtask_path}")

    def _save(self, code: str) -> None:
        """Save generated code to file."""
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(code)

    def _read_reference(self, source_robot: str, subtask_name: str) -> str:
        """Read reference code from source robot's subtasks."""
        path = f"./subtasks_{source_robot}/{subtask_name}.py"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"[WARN] Reference code not found: {path}")
            return ""

    def _extract_main_function(self, code: str, function_name: str) -> str:
        """Extract only the main function from code, excluding helper functions."""
        import ast
        try:
            tree = ast.parse(code)
            # Find the function with matching name
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Get the source code for this function only
                    lines = code.split('\n')
                    start_line = node.lineno - 1  # 0-indexed
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)

                    # Extract function code
                    func_lines = lines[start_line:end_line]
                    return '\n'.join(func_lines)


            print(f"[WARN] Function '{function_name}' not found in reference, using full code")
            return code
        except Exception as e:
            print(f"[WARN] Failed to extract function '{function_name}': {e}, using full code")
            return code

    def _extract_helper_functions(self, code: str, main_function_name: str, context: Dict[str, Any] ) -> Dict[str, callable]:
        """Extract helper functions from source subtask code (excluding the main function).

        Returns a dict mapping function names to compiled function objects.
        """
        import ast
        import numpy as np
        import skill_code

        helper_funcs = {}
        try:
            tree = ast.parse(code)
            lines = code.split('\n')

            # Collect all function definitions except the main one
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef) and node.name != main_function_name:
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else len(lines)
                    func_code = '\n'.join(lines[start_line:end_line])

                    # Compile and execute the helper function
                    import math
                    import time as time_module
                    exec(func_code, context)
                    if node.name in context:
                        helper_funcs[node.name] = context[node.name]
                        print(f"[HELPER] Loaded helper function: {node.name}")

        except Exception as e:
            print(f"[WARN] Failed to extract helper functions: {e}")

        return helper_funcs

    # ========================================================================
    # Subtask Generation & Execution
    # ========================================================================

    # initial (generate)
    def generate_subtask_code(
        self,
        subtask_name: str,
        obj_name: Optional[str] = None,
        target_name: Optional[str] = None,
        scene_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate subtask code via LLM.

        Called lazily when a subtask is first executed.

        TODO: Add caching/reuse policy for generated subtask code.

        Args:
            subtask_name: Name of the subtask (e.g., "pick_single").
            obj_name: Name of the object involved (if applicable).
            target_name: Name of the target (if applicable).
            scene_info: Current scene information.

        Returns:
            Generated subtask code string.
        """
        if self.mode == "static":
            # Load from file
            subtask_path = f"./subtasks_{self.target_robot}/{subtask_name}.py"
            with open(subtask_path, "r", encoding="utf-8") as f:
                return f.read()

        """
        # Check cache first
        if subtask_name in self._code_cache:
            print(f"[CACHE HIT] Using cached code for {subtask_name}")
            return self._code_cache[subtask_name]
        """

        # remote_llm mode
        assert self.client is not None

        # Load reference code from source robot
        reference_path = f"./subtasks_{self.source_robot}/{subtask_name}.py"
        reference_code = ""
        subtask_desc = ""
        ref_interface = None

        # First, try to get hardcoded signature
        if subtask_name in SUBTASK_SIGNATURES:
            ref_interface, subtask_desc = SUBTASK_SIGNATURES[subtask_name]
            print(f"[INFO] Using hardcoded signature for {subtask_name}: {ref_interface}")
            if ref_interface:
                self._ref_interface_cache[subtask_name] = ref_interface

        # Then try to load reference code (for reference_code body, not signature)
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                full_code = f.read()

            # If no hardcoded signature found, try extracting from reference file
            if ref_interface is None:
                info = self._extract_ref_func_info(full_code, subtask_name)
                if info is not None:
                    subtask_desc = info.docstring
                    ref_interface = info.signature_line
                    if ref_interface:
                        self._ref_interface_cache[subtask_name] = ref_interface

            # Extract main function only (excluding helpers)
            reference_code = self._extract_main_function(full_code, subtask_name)
        except FileNotFoundError:
            print(f"[WARN] Reference subtask not found: {reference_path}")

        # subtask_desc = extract_module_docstring(reference_code)
        prompt = get_prompt(
            task=None,
            task_name=self.task_name,               # 템플릿에 남아있어도 docstring에 안 쓰이게 했으니 무해
            object_names=self.object_names,
            target_robot=self.target_robot,
            descriptions=subtask_desc,              # 오직 subtask 설명만
            model=self.model,
            subtask_name=subtask_name,              # For signature lookup
        )

        print(f"[CACHE MISS] Generating code for {subtask_name}")
        resp = self.client.request({
            "type": "generate",
            "task_name": self.task_name,
            "skill_name": subtask_name,
            "simulator": "genesis",
            "robot_type": self.target_robot,
            "source_robot": self.source_robot,
            "target_robot": self.target_robot,

            # 넘겨받는 intial_prompt 무시, reference code에서 description 사용
            "prompt": prompt,
            
            "reference_code": reference_code,
            "available_objects": self.object_names,
            "descriptions": subtask_desc,              # 오직 subtask 설명만
            "target_scene_info": (scene_info or self.scene_info or {}),
            # "target_scene_info": (scene_info or self.scene_info or {}).get("objects", []),

            # Additional context for subtask
            "obj_name": obj_name,
            "target_name": target_name,

            "debug_prompt": True
        })

        print(f"[DEBUG] generate response type: {resp.get('type')}, ok: {resp.get('ok')}")
        if resp.get("type") != "generate_result" or not resp.get("ok", False):
            raise RuntimeError(f"Remote subtask generation failed: {resp}")

        code = resp.get("code", "")
        if not code.strip():
            print(f"[DEBUG] generate returned EMPTY code")
            raise RuntimeError(f"Remote subtask generation returned empty code for {subtask_name}")

        # print(f"[GENERATE] Received code from server for {subtask_name}:")
        # print(code[:800] + "..." if len(code) > 800 else code)
        
        self.last_response_num_tokens = resp.get("num_tokens")
        print(f"[DEBUG] generate returned code length: {len(code)}, tokens: {self.last_response_num_tokens}")
        print(f"[POLICY_PROVIDER] Subtask code(before preprocessing):\n{code}")

        # Cache the generated code
        # self._code_cache[subtask_name] = code
        # print(f"[CACHE SAVE] Cached code for {subtask_name}")

        return code


    # provider.repair_on_failure  -  return cont_body
    def repair_subtask(
        self,
        subtask_name: str,
        obj_name: Optional[str],
        target_name: Optional[str],
        current_code: str,
        executed_stmt_texts: List[str],
        executed_skill_texts: List[str],
        failed_skill_text: str,
        failed_stmt_text: str,
        error_info: Dict[str, Any],
        scene: Dict[str, Any],
        task_name: str,
    ) -> Optional[str]:
        if self.mode == "static":
            raise RuntimeError("static mode cannot repair subtasks. Use remote_llm.")

        assert self.client is not None

        # reference (source robot)
        reference_path = f"./subtasks_{self.source_robot}/{subtask_name}.py"
        reference_code = ""
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                full_code = f.read()
            reference_code = self._extract_main_function(full_code, subtask_name)
        except FileNotFoundError:
            print(f"[WARN] Reference subtask not found: {reference_path}")

        resp = self.client.request({
            "type": "repair",
            "task_name": task_name,
            "skill_name": subtask_name,
            "simulator": "genesis",
            "robot_type": self.target_robot,
            "source_robot": self.source_robot,
            "target_robot": self.target_robot,
            "model": self.model,
            "meta": {"model": self.model, "source_robot": self.source_robot},
            "constraints": {
                "must_keep_executed_prefix": True,
                # "executed_prefix_count": len(executed_skill_texts),
            },
            "inputs": {
                # "current_code": current_code,                 # 최신 누적 코드
                "executed_skill_texts": executed_skill_texts,   # 실행 prefix
                "failed_stmt_text": failed_stmt_text,         # 실제 실패 stmt_code
                "failed_skill_text": failed_skill_text,
                "reference_code": reference_code,
                "error": error_info,
                "scene": scene,
                "obj_name": obj_name,
                "target_name": target_name,
            },
            "debug_prompt": True
        })

        if resp.get("type") != "repair_result" or not resp.get("ok", False):
            print(f"[WARN] Remote subtask repair failed: {resp}")
            return None

        new_code = resp.get("code", "")
        if not new_code.strip():
            print("[WARN] Remote subtask repair returned empty code.")
            return None

        self.last_response_num_tokens = resp.get("num_tokens")

        return new_code


    # provider.repair_on_not_finished  -  return cont_body
    def revise_subtask(
        self,
        subtask_name: str,
        current_code: str,
        executed_stmt_texts: List[str],
        executed_skill_texts: List[str],
        scene: Dict[str, Any],
        last_step: Dict[str, Any],
    ) -> Optional[str]:
        if self.mode == "static":
            raise RuntimeError("static mode cannot revise subtasks. Use remote_llm.")

        assert self.client is not None

        # reference (source robot)
        reference_path = f"./subtasks_{self.source_robot}/{subtask_name}.py"
        reference_code = ""
        try:
            with open(reference_path, "r", encoding="utf-8") as f:
                full_code = f.read()
            reference_code = self._extract_main_function(full_code, subtask_name)
        except FileNotFoundError:
            print(f"[WARN] Reference subtask not found: {reference_path}")

        print(f"[DEBUG][REVISE_SUBTASK] Sending revise request for {subtask_name}")
        print(f"[DEBUG][REVISE_SUBTASK] executed_skill_texts count: {len(executed_skill_texts)}")
        print(f"[DEBUG][REVISE_SUBTASK] last_step: {last_step}")

        resp = self.client.request({
            "type": "revise",
            "task_name": self.task_name,
            "skill_name": subtask_name,
            "simulator": "genesis",
            "robot_type": self.target_robot,
            "source_robot": self.source_robot,
            "target_robot": self.target_robot,
            "model": self.model,
            "meta": {"model": self.model, "source_robot": self.source_robot},
            "constraints": {
                "must_keep_executed_prefix": True,
                # "executed_prefix_count": len(executed_skill_texts),
            },
            "inputs": {
                # "current_code": current_code,
                "executed_skill_texts": executed_skill_texts,
                "reference_code": reference_code,
                "scene": scene,
                "last_step": last_step,
            },
            "debug_prompt": True
        })

        print(f"[DEBUG][REVISE_SUBTASK] Response type: {resp.get('type')}, ok: {resp.get('ok')}")

        if resp.get("type") != "revise_result" or not resp.get("ok", False):
            print(f"[DEBUG][REVISE_SUBTASK] Remote subtask revise failed")
            print(f"[DEBUG][REVISE_SUBTASK] Response keys: {list(resp.keys())}")
            print(f"[DEBUG][REVISE_SUBTASK] Full response: {resp}")
            print(f"[WARN] Remote subtask revise failed: {resp}")
            return None

        new_code = resp.get("code", "")
        print(f"[DEBUG][REVISE_SUBTASK] Received code length: {len(new_code)}")

        if not new_code.strip():
            print("[DEBUG][REVISE_SUBTASK] Remote subtask revise returned EMPTY code")
            print("[WARN] Remote subtask revise returned empty code.")
            return None

        self.last_response_num_tokens = resp.get("num_tokens")

        return new_code



    def set_subtasks_target_robot(self, target_robot: Optional[str]) -> None:
        """
        Set the target robot for subtasks module.

        This must be called before importing/executing task code that uses subtasks.

        Args:
            target_robot: Target robot type ("panda", "robotiq", or "suction")
        """
        import subtasks
        # Normalize: "robotiq" -> "robotiq85" for subtasks module
        robot_map = {
            "panda": "panda",
            "robotiq": "robotiq85",
            "suction": "suction",
        }

        robot = robot_map.get((target_robot or "").lower(), target_robot)
        subtasks.set_target_robot(robot)
        print(f"[INFO] Subtasks module set to: subtasks_{robot}")
        return robot

    # execute_subtask = execute_subtask_on_failure = execute_failure_loop
    def execute_subtask(
        self,
        subtask_name: str,
        env,
        obj_name: Optional[str] = None,
        target_name: Optional[str] = None,
        check_func: Optional[callable] = None,
        max_repairs: int = 5,
        **kwargs,
    ) -> "SubtaskResult":

        # Get subtask count from context if available
        subtask_results = kwargs.get('_subtask_results', [])
        current_idx = len(subtask_results) + 1  # Current subtask is not in the list yet

        print(f"\n{'='*60}")
        print(f"[DEBUG] execute_subtask ENTRY - Subtask #{current_idx}")
        print(f"[DEBUG]   subtask_name: {subtask_name}")
        print(f"[DEBUG]   obj_name: {obj_name}")
        print(f"[DEBUG]   target_name: {target_name}")
        print(f"[DEBUG]   check_func: {'provided' if check_func is not None else 'None'}")
        print(f"[DEBUG]   max_repairs: {max_repairs}")
        print(f"[DEBUG]   model: {self.model}")
        print(f"[DEBUG]   mode: {self.mode}")
        print(f"[DEBUG]   kwargs: {list(kwargs.keys())}")
        print(f"{'='*60}")

        # target_robot = kwargs.get("target_robot", None)
        # target_robot = self.set_subtasks_target_robot(target_robot)

        # Pass subtask index to child functions
        kwargs['_current_subtask_idx'] = current_idx

        if self.model == "ours":
            return self._execute_subtask_ours_mode(
                subtask_name=subtask_name,
                env=env,
                obj_name=obj_name,
                target_name=target_name,
                check_func=check_func,
                max_repairs=max_repairs,
                **kwargs,
            )
        else:
            # code_agent mode (default): repair on failure
            return self.execute_subtask_on_failure(
                subtask_name=subtask_name,
                env=env,
                obj_name=obj_name,
                target_name=target_name,
                check_func=check_func,
                max_repairs=max_repairs,
                **kwargs,
            )


    def _create_subtask_execution_context(
        self,
        env,
        subtask_name: str,
        task=None,
    ) -> Dict[str, Any]:
        """Create execution context for statement-by-statement subtask execution."""
        import numpy as np
        import math
        import time as time_module
        import skill_code

        context = {
            'env': env,
            'task': task,  # LLM may reference task variable
            'np': np,
            'numpy': np,  # LLM may use either name
            'math': math,
            'time': time_module,
            # Common skill functions
            'move_gripper_to': skill_code.move_gripper_to,
            'move_to_position': skill_code.move_to_position,
            'move_parallel': skill_code.move_parallel,
            'rotate_gripper': skill_code.rotate_gripper,
        }

        # Always register ALL skill functions to avoid NameError from LLM-generated code
        # Generic names (aliased based on target robot)
        if self.target_robot == "panda":
            context.update({
                'open_gripper': skill_code.open_gripper,
                'close_gripper': skill_code.close_gripper,
                'pick': skill_code.pick,
                'place': skill_code.place,
                'grasp_handle': skill_code.grasp_handle,
                'release_handle': skill_code.release_handle,
            })
        elif self.target_robot in ("robotiq", "robotiq85"):
            context.update({
                'open_gripper': skill_code.open_robotiq85,
                'close_gripper': skill_code.close_robotiq85,
                'pick': skill_code.pick_robotiq85,
                'place': skill_code.place_robotiq85,
                'grasp_handle': skill_code.grasp_handle_robotiq85,
                'release_handle': skill_code.release_handle_robotiq85,
            })
        else:  # suction
            context.update({
                'activate_vacuum': skill_code.activate_vacuum,
                'deactivate_vacuum': skill_code.deactivate_vacuum,
                'attach_vacuum_handle': skill_code.attach_vacuum_handle,
                'detach_vacuum_handle': skill_code.detach_vacuum_handle,
            })

        # Also register ALL robot-specific names directly (LLM may use any of these)
        # Panda
        context.update({
            'open_panda': getattr(skill_code, 'open_gripper', None),
            'close_panda': getattr(skill_code, 'close_gripper', None),
        })
        # Robotiq85
        context.update({
            'open_robotiq85': getattr(skill_code, 'open_robotiq85', None),
            'close_robotiq85': getattr(skill_code, 'close_robotiq85', None),
            'pick_robotiq85': getattr(skill_code, 'pick_robotiq85', None),
            'place_robotiq85': getattr(skill_code, 'place_robotiq85', None),
            'grasp_handle_robotiq85': getattr(skill_code, 'grasp_handle_robotiq85', None),
            'release_handle_robotiq85': getattr(skill_code, 'release_handle_robotiq85', None),
        })
        # Suction
        context.update({
            'activate_suction': getattr(skill_code, 'activate_vacuum', None),
            'deactivate_suction': getattr(skill_code, 'deactivate_vacuum', None),
            'attach_suction': getattr(skill_code, 'attach_vacuum_handle', None),
            'detach_suction': getattr(skill_code, 'detach_vacuum_handle', None),
        })

        # Load helper functions from source subtask file (e.g., find_place_position)
        source_subtask_path = f"./subtasks_{self.source_robot}/{subtask_name}.py"
        try:
            with open(source_subtask_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            helper_funcs = self._extract_helper_functions(source_code, subtask_name, context)
            context.update(helper_funcs)
        except FileNotFoundError:
            pass

        return context



    # rlbench_loop - INITIAL/REVISE/REPAIR 분기에 대해 
    def execute_subtask_on_failure(
        self,
        subtask_name: str,
        env,
        obj_name: Optional[str] = None,
        target_name: Optional[str] = None,
        check_func: Optional[callable] = None,
        max_repairs: int = 5,
        max_skill_calls: Optional[int] = None,
        **kwargs,
    ) -> "SubtaskResult":
        """
        Genesis failure-mode execution loop (ported from RLBench execute_failure_loop).
        - current_code를 유지하면서 AST로 stmt 단위 실행
        - 실패 시 repair(cont) 생성 후 merge
        - stmt 모두 실행했는데 미완료면 revise(cont) 생성 후 merge
        """

        print(f"[DEBUG] _execute_subtask_on_failure called for {subtask_name}")
        print(f"[DEBUG]   obj_name: {obj_name}, target_name: {target_name}")
        print(f"[DEBUG]   check_func: {'provided' if check_func is not None else 'None'}")

        # target_robot = kwargs.get("target_robot", None)
        # target_robot = self.set_subtasks_target_robot(target_robot)

        result = SubtaskResult(
            subtask_name=subtask_name,
            obj_name=obj_name,
            target_name=target_name,
        )

        # -------------------------
        # 0) Scene (initial)
        # -------------------------

        scene_info = get_scene_info(env, target_robot=self.target_robot)
        print(f"[DEBUG] Scene info keys: {list(scene_info.keys()) if isinstance(scene_info, dict) else type(scene_info)}")

        # -------------------------
        # 1) INITIAL (GENERATE)
        # -------------------------
        start_time = time.time()
        try:
            raw_code = self.generate_subtask_code(
                subtask_name=subtask_name,
                obj_name=obj_name,
                target_name=target_name,
                scene_info=scene_info,
            )
            result.generated_code = raw_code
            gen_duration = time.time() - start_time
            self._log_llm_call("GENERATE", subtask_name, gen_duration, self.last_response_num_tokens)



            if self.code_gen_only:
                print(f"[CODE_GEN_ONLY] Skipping execution for {subtask_name}, assuming success")
                result.success = True
                results.total_repair_count = 0
                # results.repair_count = 0
                return result

        except Exception as e:
            current_idx = kwargs.get('_current_subtask_idx', '?')
            gen_duration = time.time() - start_time
            self._log_llm_call("GENERATE_FAILED", subtask_name, gen_duration, None)
            result.error = f"Failed to generate subtask code: {e}"
            self.total_failure_count += 1
            print(f"\n{'='*60}")
            print(f"[SUBTASK FAILED #{current_idx} - CODE GENERATION] {subtask_name}")
            print(f"  obj_name: {obj_name}")
            print(f"  target_name: {target_name}")
            print(f"  error: {result.error}")
            print(f"{'='*60}\n")
            raise SubtaskFailure(
                subtask_name=subtask_name,
                obj_name=obj_name,
                target_name=target_name,
                message=result.error,
            )

        # -------------------------
        # 2) Build current_code (full def + imports)
        # -------------------------
        
        # raw_code = self._postprocess_subtask_code(raw_code)

        ref_sig = self._ref_interface_cache.get(subtask_name)
        # 실행할 최종 INITIAL code
        current_code = postprocess_initial_code(
            llm_generated=raw_code,
            subtask_name=subtask_name,
            robot_type=self.target_robot,
            run_task_interface=ref_sig,
        )
        print(f"[POLICY_PROVIDER] Subtask code(after preprocessing_initial):\n{current_code}")

        # Save postprocessed code (with proper function definition and imports)
        self._save_subtask(subtask_name, current_code)
        current_lines = current_code.split("\n")

        # 실행 상태 변수 초기화
        executed_stmt_texts: List[str] = []
        total_repair_count = 0
        # repair_attempt_count = 0

        # -------------------------
        # 3) Execution context (RLBench create_execution_context와 유사)
        # -------------------------
        import numpy as np
        import skill_code
        import math
        import time as time_module

        tracer = TraceRecorder()
        executed_skill_texts = tracer.executed_skill_texts

        # task 확보(있으면 넣고, 없으면 env.env.task에서 추정)
        task = kwargs.get("task", None)
        if task is None and hasattr(env, "env") and hasattr(env.env, "task"):
            task = env.env.task

        # (A) 베이스 컨텍스트
        context = self._create_subtask_execution_context(env, subtask_name, task=task)

        # (B) tracer 주입 + 실행 인자 주입 (exec에서 NameError 방지)

        context["__trace__"] = tracer
        # obj_name, target_name: only inject if not None (to avoid "Object 'None' not found" error)
        if obj_name is not None:
            context["obj_name"] = obj_name
        if target_name is not None:
            context["target_name"] = target_name
        context.setdefault("pointing_to", kwargs.get("pointing_to", "down"))
        context.setdefault("height_offset", kwargs.get("height_offset", 0.1))

        # (B-1) SUBTASK_SIGNATURES에 정의된 파라미터명과 실제 전달되는 인자명 매핑
        # open_box_lid, open_hinge_lid, close_hinge_lid: handle_name, lid_name
        # Always inject with defaults based on subtask type (to avoid NameError)
        if subtask_name in ("open_hinge_lid", "close_hinge_lid"):
            context.setdefault("handle_name", obj_name if obj_name is not None else "hinge_handle")
            context.setdefault("lid_name", target_name if target_name is not None else "hinge_lid")
        elif subtask_name == "open_box_lid":
            context.setdefault("handle_name", obj_name if obj_name is not None else "box_handle")
            context.setdefault("lid_name", target_name if target_name is not None else "box_lid")
        else:
            # For other subtasks, only inject if provided
            if obj_name is not None:
                context.setdefault("handle_name", obj_name)
            if target_name is not None:
                context.setdefault("lid_name", target_name)
        # open_drawer, close_drawer: drawer_type and handle_name
        drawer_type = kwargs.get("drawer_type", "bottom")
        context.setdefault("drawer_type", drawer_type)
        if subtask_name in ("open_drawer", "close_drawer"):
            # Derive handle_name from drawer_type for drawer operations
            drawer_handle = "top_drawer_handle" if drawer_type == "top" else "bottom_drawer_handle"
            context.setdefault("handle_name", drawer_handle)
        # insert_into_slot, place_beside_slot: slot_pos
        context.setdefault("slot_pos", kwargs.get("slot_pos", None))
        # insert_into_slot: insert_height
        context.setdefault("insert_height", kwargs.get("insert_height", 0.05))
        # remove_from_slot: lift_height
        context.setdefault("lift_height", kwargs.get("lift_height", 0.15))
        # place_at_position: position (kwargs에서 position 또는 target_pos 체크)
        context.setdefault("position", kwargs.get("position", kwargs.get("target_pos", None)))
        # relocate_single: target_position
        context.setdefault("target_position", kwargs.get("target_position", kwargs.get("target_pos", None)))
        # place_beside_slot: offset_x
        context.setdefault("offset_x", kwargs.get("offset_x", -0.08))
        # pick_and_place_object: place_height, spread_offset, placed_objects, use_smart_placement
        context.setdefault("place_height", kwargs.get("place_height", 0.01))
        context.setdefault("spread_offset", kwargs.get("spread_offset", 0.0))
        context.setdefault("placed_objects", kwargs.get("placed_objects", None))
        context.setdefault("use_smart_placement", kwargs.get("use_smart_placement", True))
        # pour_into_container: pour_angle, pour_steps, hold_steps
        context.setdefault("pour_angle", kwargs.get("pour_angle", 120))
        context.setdefault("pour_steps", kwargs.get("pour_steps", 60))
        context.setdefault("hold_steps", kwargs.get("hold_steps", 100))
        # release_container: place_pos, original_pos
        context.setdefault("place_pos", kwargs.get("place_pos", None))
        context.setdefault("original_pos", kwargs.get("original_pos", None))
        # target_pos도 직접 주입 (pick_and_place_object 등에서 사용)
        context.setdefault("target_pos", kwargs.get("target_pos", None))

        # (B-2) LLM이 task["obj_name"] 등을 사용할 경우를 대비해 task 객체에도 속성 주입
        if task is not None:
            task.obj_name = obj_name
            task.target_name = target_name
            task.pointing_to = kwargs.get("pointing_to", "down")
            task.height_offset = kwargs.get("height_offset", 0.1)
            # kwargs에서 추가 속성도 주입
            for k, v in kwargs.items():
                if not hasattr(task, k):
                    setattr(task, k, v)

        # kwargs도 필요하면 넣되, 핵심 키는 덮어쓰지 않게 보호
        _reserved = {"env", "np", "numpy", "task", "__trace__", "math", "time"}
        for k, v in kwargs.items():
            if k in _reserved:
                continue
            context[k] = v

        # (C) primitive 래핑 (skill call count / executed_skill_texts 채우기)
        WRAP_NAMES = [
            "move_gripper_to", "move_to_position", "move_parallel", "rotate_gripper",
            "open_gripper", "close_gripper", "pick", "place", "grasp_handle", "release_handle",
            "open_panda", "close_panda",
            "open_robotiq85", "close_robotiq85", "pick_robotiq85", "place_robotiq85",
            "grasp_handle_robotiq85", "release_handle_robotiq85",
            "activate_vacuum", "deactivate_vacuum", "attach_vacuum_handle", "detach_vacuum_handle",
            "activate_suction", "deactivate_suction", "attach_suction", "detach_suction",
        ]
        for name in WRAP_NAMES:
            fn = context.get(name, None)
            if callable(fn):
                context[name] = self._wrap_skill(fn, name, tracer)

        # 이제 tracer.executed_skill_texts / tracer.skill_call_count가 실제 primitive 기준으로 쌓입니다.

                

        # helper 함수 로딩 (e.g., find_place_position)
        source_subtask_path = f"./subtasks_{self.source_robot}/{subtask_name}.py"
        try:
            with open(source_subtask_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            helper_funcs = self._extract_helper_functions(source_code, subtask_name, context)
            context.update(helper_funcs)
        except FileNotFoundError:
            pass

        # -------------------------
        # 4) Main loop (execute_failure_loop와 동일한 큰 흐름)
        # -------------------------
        while True:
            # current_code를 AST로 분해
            bundle = parse_code_bundle(current_code, subtask_name)
            current_lines = current_code.split("\n")

            # helper defs를 매 루프마다 exec
            for _, func_node in bundle.helper_nodes.items():
                mod = ast.Module(body=[func_node], type_ignores=[])
                ast.fix_missing_locations(mod)
                exec(compile(mod, "<helper>", "exec"), context)

            # 다음에 실행할 stmt 목록
            next_index = len(executed_stmt_texts)
            stmts_to_execute = bundle.run_body[next_index:]

            # -------------------------
            # (A) 실행할 stmt가 더 없으면: 성공 체크 or revise
            # -------------------------
            if not stmts_to_execute:
                # success 판단은 RLBench에서는 task.success(); 여기서는 check_func로 대체
                if check_func is not None:
                    try:
                        if check_func(env, obj_name, target_name):
                            current_idx = kwargs.get('_current_subtask_idx', '?')
                            result.success = True
                            result.repair_count = total_repair_count
                            # Save final code (includes any repairs)
                            if total_repair_count > 0:
                                self._save_subtask(subtask_name, current_code)
                            print(f"\n{'='*60}")
                            print(f"[SUBTASK SUCCESS #{current_idx}] {subtask_name}")
                            print(f"  obj_name: {obj_name}")
                            print(f"  target_name: {target_name}")
                            print(f"  repair_count: {total_repair_count}")
                            print(f"  skill_calls: {tracer.skill_call_count}")
                            print(f"{'='*60}\n")
                            return result
                    except Exception:
                        # check_func가 터져도 revise로 넘기는 편이 안전
                        pass

                if max_skill_calls is not None and tracer.skill_call_count >= max_skill_calls:
                    print(f"[MAX_STEPS] Reached maximum steps: {max_skill_calls}")
                    result.success = False
                    result.skipped = True
                    result.repair_count = total_repair_count
                    raise SubtaskSkip(
                        subtask_name=subtask_name,
                        obj_name=obj_name,
                        repair_attempts=total_repair_count,
                        result=result,
                    )

                # REVISE (not finished)
                print("[INFO] All statements executed but subtask incomplete. Requesting additional code...")
                if self.mode == "static":
                    result.error = "Subtask incomplete (static mode, no revise)"
                    result.skipped = True
                    result.repair_count = total_repair_count
                    raise SubtaskSkip(
                        subtask_name=subtask_name,
                        obj_name=obj_name,
                        repair_attempts=total_repair_count,
                        result=result,
                    )

                if total_repair_count > max_repairs:
                    current_idx = kwargs.get('_current_subtask_idx', '?')
                    result.error = f"Max repairs/revises ({max_repairs}) exceeded: not finished"
                    result.skipped = True
                    result.repair_count = total_repair_count
                    self.total_failure_count += 1
                    print(f"\n{'='*60}")
                    print(f"[SUBTASK FAILED #{current_idx} - MAX REPAIRS] {subtask_name}")
                    print(f"  obj_name: {obj_name}")
                    print(f"  target_name: {target_name}")
                    print(f"  repair_count: {total_repair_count} / {max_repairs}")
                    print(f"  skill_calls: {tracer.skill_call_count}")
                    print(f"  error: {result.error}")
                    print(f"{'='*60}\n")
                    raise SubtaskSkip(
                        subtask_name=subtask_name,
                        obj_name=obj_name,
                        repair_attempts=total_repair_count,
                        result=result,
                    )

                # 마지막 실행 정보
                last_step = {
                    "step_index": len(executed_stmt_texts),
                    "skill_call_count": tracer.skill_call_count,
                    "statement": executed_stmt_texts[-1] if executed_stmt_texts else "",
                    "skill_name": None,
                    "success": False,
                    "need_more_steps": True,
                }

                # revise
                revise_start = time.time()
                try:
                    self.total_repair_count += 1
                    total_repair_count += 1

                    executed_skill_texts = getattr(tracer, "executed_skill_texts", []) or []

                    # revise로 cont_body(new_code) 생성
                    new_code = self.revise_subtask(
                        subtask_name=subtask_name,
                        current_code=current_code,
                        executed_stmt_texts=executed_stmt_texts,
                        executed_skill_texts=executed_skill_texts,
                        scene=get_scene_info(env, target_robot=self.target_robot),
                        last_step=last_step,
                    )
                    revise_duration = time.time() - revise_start
                    self._log_llm_call("REVISE", subtask_name, revise_duration, self.last_response_num_tokens)

                    print("[DEBUG][REVISE] raw LLM output head:", new_code[:500])
                    cont_body = postprocess_continuation_code(new_code, subtask_name=subtask_name)
                    print("[DEBUG][REVISE] postprocessed chars=", len(cont_body or ""))
                    if not (cont_body or "").strip():
                        print("[DEBUG][REVISE] postprocess returned EMPTY")

                    if not cont_body.strip():
                        raise RuntimeError("Revise returned empty continuation")

                    # REVISE는 마지막 return 앞에 삽입
                    current_code = merge_continuation_into_subtask(
                        current_code=current_code,
                        cont_body=cont_body,
                        func_name=subtask_name,
                        insert_before_last_return=True,
                        insert_after_lineno=None,
                    )

                    # 마지막 stmt가 return 기록이었으면 pop
                    if executed_stmt_texts and executed_stmt_texts[-1].lstrip().startswith("return"):
                        executed_stmt_texts.pop()

                    # merge 직후(또는 루프 시작 직후) 기준으로, 다음에 실행될 statement들을 미리 출력
                    debug_print_next_statements(
                        current_code=current_code,
                        current_lines=current_code.split("\n"),
                        executed_stmt_texts=executed_stmt_texts,
                        subtask_name=subtask_name,
                        k=5,
                        tag="REVISE_AFTER_MERGE",
                        show_full=True,
                    )
                    continue

                except Exception as e:
                    revise_duration = time.time() - revise_start
                    self._log_llm_call("REVISE_FAILED", subtask_name, revise_duration, None)


                    print(f"[DEBUG][REVISE_FAILED] Exception type: {type(e).__name__}")
                    print(f"[DEBUG][REVISE_FAILED] Exception message: {str(e)}")
                    print(f"[DEBUG][REVISE_FAILED] Traceback:")
                    traceback.print_exc()

                    result.error = f"Revise failed: {e}"
                    result.skipped = True
                    result.repair_count = total_repair_count
                    raise SubtaskSkip(
                        subtask_name=subtask_name,
                        obj_name=obj_name,
                        repair_attempts=total_repair_count,
                        result=result,
                    )

            # -------------------------
            # (B) 실행할 stmt가 더 있으면: stmt 실행 루프
            # -------------------------
            repair_needed = False

            for statement_index, stmt in enumerate(stmts_to_execute, start=next_index + 1):
                stmt_code = self.get_statement_code(stmt, current_lines)

                # top-level return은 실행하지 않고 기록만 (최종 return)
                if isinstance(stmt, ast.Return):
                    print(f"[STEP {statement_index}] Skipping return statement (end of {subtask_name})")
                    executed_stmt_texts.append(stmt_code)
                    next_index += 1
                    break

                # nested return를 pass로 치환
                stmt_exec = stmt_exec = self._ReturnToPassTransformer._rewrite_stmt_return_to_pass(stmt)
                ast.fix_missing_locations(stmt_exec)

                try:
                    # 한 stmt씩 exec
                    mod = ast.Module(body=[stmt_exec], type_ignores=[])
                    ast.fix_missing_locations(mod)
                    exec(compile(mod, "<ast_step>", "exec"), context)

                    skill_call_count = tracer.skill_call_count

                    executed_stmt_texts.append(stmt_code)
                    next_index += 1

                    if check_func is not None:
                        try:
                            if check_func(env, obj_name, target_name):
                                result.success = True
                                result.repair_count = total_repair_count
                                # Save final code (includes any repairs)
                                if total_repair_count > 0:
                                    self._save_subtask(subtask_name, current_code)
                                return result
                        except Exception:
                            pass

                    # 상위 stmt step 예산 체크
                    if max_skill_calls is not None and tracer.skill_call_count >= max_skill_calls:
                        print(f"[MAX_STEPS] Reached maximum steps: {max_skill_calls}")
                        result.success = False
                        result.skipped = True
                        result.repair_count = total_repair_count
                        raise SubtaskSkip(
                            subtask_name=subtask_name,
                            obj_name=obj_name,
                            repair_attempts=total_repair_count,
                            result=result,
                        )


                except Exception as e:
                    # ---- 핵심: 실패 시 "삽입 라인"을 위해 lineno/end_lineno를 context로 넣기 ----
                    
                    failed_skill_text = getattr(tracer, "failed_skill_text", "") or ""
                    executed_skill_texts = getattr(tracer, "executed_skill_texts", []) or []

                    # failed stmt (failed skill 아님)
                    executed_stmt_texts.append(stmt_code)
                    next_index += 1
                    
                    tb = traceback.format_exc()
                    scene_fail = get_scene_info(env, target_robot=self.target_robot)

                    failed_lno = getattr(stmt, "lineno", None)
                    failed_end = getattr(stmt, "end_lineno", failed_lno)
                    after_ln_0based = (int(failed_end) - 1) if failed_end is not None else None

                    print(f"[FAILURE] {type(e).__name__}: {e}")

                    if self.mode == "static":
                        result.error = f"Subtask failed (static mode, no repair): {type(e).__name__}: {e}"
                        result.skipped = True
                        result.repair_count = total_repair_count
                        raise SubtaskSkip(
                            subtask_name=subtask_name,
                            obj_name=obj_name,
                            repair_attempts=total_repair_count,
                            result=result,
                        )

                    if total_repair_count > max_repairs:
                        result.error = f"Max repairs ({max_repairs}) exceeded: {type(e).__name__}: {e}"
                        result.skipped = True
                        current_idx = kwargs.get('_current_subtask_idx', '?')
                        result.repair_count = total_repair_count
                        self.total_failure_count += 1
                        print(f"\n{'='*60}")
                        print(f"[SUBTASK FAILED #{current_idx} - MAX REPAIRS DURING EXECUTION] {subtask_name}")
                        print(f"  obj_name: {obj_name}")
                        print(f"  target_name: {target_name}")
                        print(f"  repair_count: {total_repair_count} / {max_repairs}")
                        print(f"  skill_calls: {tracer.skill_call_count}")
                        print(f"  last_error: {type(e).__name__}: {e}")
                        print(f"{'='*60}\n")
                        raise SubtaskSkip(
                            subtask_name=subtask_name,
                            obj_name=obj_name,
                            repair_attempts=total_repair_count,
                            result=result,
                        )

                    # repair 시도 루프 - max_repairs까지 재시도
                    while True:
                        # max_repairs 체크 (repair 시도 전에 체크)
                        if total_repair_count >= max_repairs:
                            result.error = f"Max repairs ({max_repairs}) exceeded during repair loop"
                            result.skipped = True
                            result.repair_count = total_repair_count
                            self.total_failure_count += 1
                            print(f"\n[REPAIR_MAX_EXCEEDED] {subtask_name}: {total_repair_count}/{max_repairs} repairs attempted")
                            raise SubtaskSkip(
                                subtask_name=subtask_name,
                                obj_name=obj_name,
                                repair_attempts=total_repair_count,
                                result=result,
                            )

                        repair_start = time.time()
                        try:
                            self.total_repair_count += 1
                            total_repair_count += 1
                            new_code = self.repair_subtask(
                                subtask_name=subtask_name,
                                obj_name=obj_name,
                                target_name=target_name,
                                current_code=current_code,
                                executed_stmt_texts=executed_stmt_texts,
                                executed_skill_texts=executed_skill_texts,
                                failed_skill_text=failed_skill_text,
                                failed_stmt_text=stmt_code,
                                error_info={
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "traceback": tb,
                                    "failed_lineno": failed_lno,
                                    "failed_end_lineno": failed_end,
                                },
                                scene=scene_fail,
                                task_name=self.task_name,
                            )
                            repair_duration = time.time() - repair_start
                            self._log_llm_call("REPAIR", subtask_name, repair_duration, self.last_response_num_tokens)


                            raw_head = (new_code[:1200] + "\n...") if isinstance(new_code, str) and len(new_code) > 1200 else str(new_code)
                            raw_tail = ("...\n" + new_code[-800:]) if isinstance(new_code, str) and len(new_code) > 2000 else ""
                            print("\n[DEBUG][REPAIR] raw LLM continuation (head):\n", raw_head)
                            if raw_tail:
                                print("\n[DEBUG][REPAIR] raw LLM continuation (tail):\n", raw_tail)


                            cont_body = postprocess_continuation_code(new_code, subtask_name=subtask_name)
                            cont_head = (cont_body[:1200] + "\n...") if len(cont_body) > 1200 else cont_body
                            cont_tail = ("...\n" + cont_body[-800:]) if len(cont_body) > 2000 else ""
                            print("\n[DEBUG][REPAIR] postprocessed continuation:\n", cont_head)
                            if cont_tail:
                                print("\n[DEBUG][REPAIR] postprocessed continuation (tail):\n", cont_tail)


                            if not cont_body.strip():
                                raise RuntimeError("Repair returned empty continuation")

                            # (3) Merge: REPAIR는 실패 stmt 뒤에 삽입 (RLBench와 동일: insert_after_lineno 사용)
                            current_code = merge_continuation_into_subtask(
                                current_code=current_code,
                                cont_body=cont_body,
                                func_name=subtask_name,
                                insert_before_last_return=False,
                                insert_after_lineno=after_ln_0based,  # lineno/end_lineno를 넣는 이유입니다.
                            )


                            debug_print_next_statements(
                                current_code=current_code,
                                current_lines=current_code.split("\n"),
                                executed_stmt_texts=executed_stmt_texts,
                                subtask_name=subtask_name,
                                k=5,
                                tag="REPAIR_AFTER_MERGE",
                                show_full=True,
                            )

                            repair_needed = True
                            break  # repair 성공, repair 루프 탈출

                        except Exception as repair_error:
                            repair_duration = time.time() - repair_start
                            self._log_llm_call("REPAIR_FAILED", subtask_name, repair_duration, None)
                            tb2 = traceback.format_exc()
                            print(f"[REPAIR_FAILED] Attempt {total_repair_count}/{max_repairs}: {repair_error}")
                            # max_repairs 체크는 while 루프 시작에서 수행, 여기서는 continue로 재시도
                            continue

            if repair_needed:
                # merge 후 루프 재시작
                continue

            # 여기까지 왔는데도 성공 판정이 안 나면, 다음 while에서 (A)로 가서 revise 처리됩니다.


    # ========================================================================
    # Code Execution
    # ========================================================================

    class _ReturnToPassTransformer(ast.NodeTransformer):
        def visit_Return(self, node: ast.Return):
            return ast.Pass()

        @staticmethod
        def _rewrite_stmt_return_to_pass(stmt: ast.stmt) -> ast.stmt:
            new_stmt = PolicyProvider._ReturnToPassTransformer().visit(stmt)
            ast.fix_missing_locations(new_stmt)
            return new_stmt
        

        def _postprocess_subtask_code(self, code: str) -> str:
            """
            Remove code block markers from subtask code.

            Args:
                code: Raw subtask code string.

            Returns:
                Cleaned subtask code string.
            """
            code = code.strip()

            # Remove markdown code block markers
            if code.startswith("```python"):
                code = code[len("```python"):]
            elif code.startswith("```"):
                code = code[len("```"):]

            if code.endswith("```"):
                code = code[:-len("```")]

            return code.strip()





