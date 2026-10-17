import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Skill Names 동적 로드 (skill_code*.py 파일들에서 추출)
# ============================================================================

def _load_skill_names_from_files() -> set:
    """skill_code*.py 파일들에서 스킬 함수 이름들을 동적으로 추출"""
    skill_code_dir = Path(__file__).parent
    skill_names = set()

    # 헬퍼/유틸 함수들 (스킬이 아닌 것들)
    helper_functions = {
        'save_camera_image', 'normalize_quaternion', 'euler_from_quat',
        'safe_step', 'angle_diff', 'open_and_retract',
    }

    for path in skill_code_dir.glob("skill_code*.py"):
        try:
            with open(path) as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # _로 시작하는 private 함수 제외
                    if not node.name.startswith('_') and node.name not in helper_functions:
                        skill_names.add(node.name)
        except Exception as e:
            print(f"Warning: Failed to parse {path}: {e}")

    return skill_names

# 전역 스킬 이름 집합 (모듈 로드 시 한 번만 초기화)
SKILL_NAMES: set = _load_skill_names_from_files()


class RunSkillExtractor(ast.NodeVisitor):
    def __init__(self):
        self.run_skill_body: List[ast.stmt] = []
        self.helper_functions: Dict[str, ast.FunctionDef] = {}

    def visit_FunctionDef(self, node):
        if node.name == "run_skill":
            self.run_skill_body = node.body
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
    """statement가 skill 호출인지 확인하고, skill 이름 반환"""
    # 전역 SKILL_NAMES 사용 (모듈 로드 시 skill_code*.py에서 동적 추출)

    def find_call(node):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in SKILL_NAMES:
                return node.func.id
            if isinstance(node.func, ast.Attribute) and node.func.attr in SKILL_NAMES:
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


def parse_code_bundle(code: str) -> CodeBundle:
    lines = code.split("\n")
    tree = ast.parse(code)
    ex = RunSkillExtractor()
    ex.visit(tree)
    if not ex.run_skill_body:
        raise ValueError("run_skill not found in policy code.")

    helper_src_parts = []
    for name in sorted(ex.helper_functions.keys()):
        node = ex.helper_functions[name]
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        helper_src_parts.append("\n".join(lines[start:end]))
    helper_src = "\n\n".join(helper_src_parts)

    run_stmt_texts = [get_statement_code(st, lines) for st in ex.run_skill_body]

    return CodeBundle(
        code=code,
        lines=lines,
        run_body=ex.run_skill_body,
        run_stmt_texts=run_stmt_texts,
        helper_nodes=ex.helper_functions,
        helper_hash=sha1(helper_src),
        code_hash=sha1(code),
    )