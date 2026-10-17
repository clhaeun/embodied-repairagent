import ast
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


class RunTaskExtractor(ast.NodeVisitor):
    def __init__(self, subtask_name: str):
        self.subtask_name = subtask_name
        self.subtask_name_body = None
        self.helper_functions = {}

    def visit_FunctionDef(self, node):
        # target function body
        if node.name == self.subtask_name:
            self.subtask_name_body = node.body

        # helper functions
        else:
            self.helper_functions[node.name] = node

        self.generic_visit(node)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_statement_code(stmt: ast.stmt, source_lines: List[str]) -> str:
    start_line = stmt.lineno - 1
    end_line = getattr(stmt, "end_lineno", stmt.lineno)
    return "\n".join(source_lines[start_line:end_line])


def is_skill_call(stmt: ast.stmt) -> Tuple[bool, Optional[str]]:
    skill_names = {
            "move_gripper_to", "move_to_position", "move_parallel",
            "open_gripper", "close_gripper", "pick", "place",
            "rotate_gripper", "grasp_handle", "release_handle",
            "open_robotiq85", "close_robotiq85", "pick_robotiq85", "place_robotiq85",
            "grasp_handle_robotiq85", "release_handle_robotiq85",
            "activate_vacuum", "deactivate_vacuum",
            "attach_vacuum_handle", "detach_vacuum_handle",
    }

    def find_call(node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in skill_names:
                return node.func.id
            if isinstance(node.func, ast.Attribute) and node.func.attr in skill_names:
                return node.func.attr
        return None

    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        s = find_call(stmt.value)
        return (s is not None), s

    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        s = find_call(stmt.value)
        return (s is not None), s

    return False, None


def common_prefix_len(a: List[str], b: List[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i].strip() == b[i].strip():
        i += 1
    return i


@dataclass
class CodeBundle:
    code: str
    lines: List[str]
    run_body: List[ast.stmt]
    run_stmt_texts: List[str]
    helper_nodes: Dict[str, ast.FunctionDef]
    helper_hash: str
    code_hash: str

def parse_code_bundle(code: str, subtask_name: str) -> CodeBundle:
    lines = code.split("\n")
    tree = ast.parse(code)

    # FIX 1) extractor 이름 수정
    ex = RunTaskExtractor(subtask_name)
    ex.visit(tree)

    if not ex.subtask_name_body:
        raise ValueError(f"Subtask function '{subtask_name}' not found in policy code.")

    helper_src_parts = []
    for name in sorted(ex.helper_functions.keys()):
        node = ex.helper_functions[name]
        start = node.lineno - 1
        end_inclusive = getattr(node, "end_lineno", node.lineno)

        # FIX 2) end_lineno는 inclusive이므로 +1 해서 slice
        helper_src_parts.append("\n".join(lines[start:end_inclusive]))

    helper_src = "\n\n".join(helper_src_parts)
    run_stmt_texts = [get_statement_code(st, lines) for st in ex.subtask_name_body]

    return CodeBundle(
        code=code,
        lines=lines,
        run_body=ex.subtask_name_body,
        run_stmt_texts=run_stmt_texts,
        helper_nodes=ex.helper_functions,
        helper_hash=sha1(helper_src),
        code_hash=sha1(code),
    )



"""
def parse_code_bundle(code: str, subtask_name: str) -> CodeBundle:


    lines = code.split("\n")
    tree = ast.parse(code)
    ex = RunTaskExtractor(subtask_name)
    ex.visit(tree)

    if not ex.subtask_name_body:
        raise ValueError(f"Subtask function '{subtask_name}' not found in policy code.")


    helper_src_parts = []
    for name in sorted(ex.helper_functions.keys()):
        node = ex.helper_functions[name]
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        helper_src_parts.append("\n".join(lines[start:end]))
    helper_src = "\n\n".join(helper_src_parts)

    run_stmt_texts = [get_statement_code(st, lines) for st in ex.subtask_name_body]

    return CodeBundle(
        code=code,
        lines=lines,
        run_body=ex.subtask_name_body,
        run_stmt_texts=run_stmt_texts,
        helper_nodes=ex.helper_functions,
        helper_hash=sha1(helper_src),
        code_hash=sha1(code),
    )
"""
