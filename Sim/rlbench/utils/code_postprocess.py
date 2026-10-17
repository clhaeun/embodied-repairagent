import ast
import re
from typing import Dict, Set
from termcolor import colored
import astor
import parso
import astunparse

CODE_HEADER = """
from env import setup_environment, shutdown_environment
from skill_code import *
import argparse
import importlib
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.dummy import Dummy
from pyrep.objects.proximity_sensor import ProximitySensor
from env import setup_environment, shutdown_environment
from rlbench.const import colors
"""

def _try_parse(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

# FOR INTERPRETER ERRORS
## extract policy code in code block
def extract_from_code_block(code: str):
    pattern = r'```python(?:\s*\n)?(.*?)```'
    matches = re.findall(pattern, code, flags=re.DOTALL)
    if len(matches) > 0:
        code_block = matches[0].strip()
        # # 대충 header 안들어있는지 확인
        # if not "from skill_code import pick, place, move" in code_block:
        #     return CODE_HEADER + "\n" + code_block
        return code_block, True
    else:
        return code, False

## non-code text
def _get_docstring_line_indices(code: str) -> Set[int]:
    """
    docstring line 번호 반환 (정상동작 확인완)
    """
    lines = code.splitlines()
    doc_lines = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _heuristic_triple_quote_blocks(code)

    def visit(node):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None and node.body:
                first = node.body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, (ast.Constant,)):
                    start = first.lineno - 1
                    end = getattr(first, "end_lineno", start) - 1
                    for i in range(start, end + 1):
                        doc_lines.add(i)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return doc_lines

def _heuristic_triple_quote_blocks(code: str) -> Set[int]:
    lines = code.splitlines()
    doc_lines = set()
    open_quote = None
    for idx, line in enumerate(lines):
        if open_quote:
            doc_lines.add(idx)
            if open_quote in line:
                # 닫힘 감지 (단순히 동일한 따옴표가 다시 나오면)
                # 여러 개 있을 경우도 고려: 시작+끝이 한 줄인 경우도 처리
                if line.count(open_quote) % 2 == 1:
                    open_quote = None
        else:
            m = re.search(r'("""|\'\'\')', line)
            if m:
                quote = m.group(1)
                open_quote = quote
                doc_lines.add(idx)
                # 시작과 끝이 같은 줄에 둘 다 있으면 바로 닫힘 처리
                if line.count(quote) >= 2:
                    open_quote = None
    return doc_lines

_keyword_start = re.compile(
    r'^(?:def|class|import|from|return|if|elif|else|for|while|try|except|with|as|lambda|assert|yield|async|await)\b'
)

def _is_code_like_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("#"):
        return True
    if _keyword_start.match(stripped):
        return True
    code_indicators = [
        "=", ":", "->", "(", ")", "[", "]", "{", "}", "+", "-", "*", "/", "%",
    ]
    if any(ind in line for ind in code_indicators):
        # print(colored(line, "green"))
        return True
    if re.match(r"\s*[A-Za-z_][A-Za-z0-9_]*\s*=", line):
        # print(colored(line, "blue"))
        return True
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", line):
        # print(colored(line, "yellow"))
        return True
    # print(colored(line, "red"))
    return False

def _is_prose_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    if _is_code_like_line(line):
        return False
    # words = stripped.split()
    # if len(words) >= 5:
    #     if any(c in stripped for c in [".", ",", ";", ":", "?"]):
    #         return True
    #     english_connectors = {"and", "or", "but", "because", "however", "therefore"}
    #     korean_connectors = {"그리고", "하지만", "그런데", "예를", "들면", "즉"}
    #     lower_words = {w.strip(".,!?").lower() for w in words}
    #     if lower_words & english_connectors:
    #         return True
    #     avg_len = sum(len(w) for w in words) / len(words)
    #     if avg_len >= 4 and len(words) >= 6:
    #         return True
    return True

def remove_non_code_text(code: str) -> str:
    """
    docstring을 보호하면서 LLM이 삽입한 불필요한 자연어(문장형 텍스트) 줄을 제거.
    """
    doc_lines = _get_docstring_line_indices(code)
    print("doc_lines: ", doc_lines)
    lines = code.splitlines()

    # 이미 파싱 가능하다면 docstring 줄은 삭제 대상에서 제외하고 prose만 제거
    if _try_parse(code):
        cleaned = []
        for idx, ln in enumerate(lines):
            if idx in doc_lines:
                cleaned.append(ln)
            elif _is_prose_line(ln):
                continue
            else:
                cleaned.append(ln)
        return "\n".join(cleaned)

    # 파싱 안 될 때: prose 후보들을 조합적으로 제거하되 docstring 줄은 건드리지 않음
    prose_indices = [i for i, ln in enumerate(lines) if i not in doc_lines and _is_prose_line(ln)]
    cleaned_lines = lines.copy()
    from itertools import combinations

    if len(prose_indices) <= 8:
        found = False
        for r in range(len(prose_indices), 0, -1):
            for combo in combinations(prose_indices, r):
                candidate = [ln for idx, ln in enumerate(lines) if idx not in combo]
                if _try_parse("\n".join(candidate)):
                    cleaned_lines = candidate
                    found = True
                    break
            if found:
                break
    else:
        # 그리디하게 하나씩 제거
        for i in prose_indices:
            if i >= len(cleaned_lines):
                continue
            if i in doc_lines:
                continue
            candidate = cleaned_lines[:i] + cleaned_lines[i+1:]
            if _try_parse("\n".join(candidate)):
                cleaned_lines = candidate

    # 여전히 파싱 안 되면, docstring 줄은 살리고 나머지 prose 줄은 무조건 제거
    if not _try_parse("\n".join(cleaned_lines)):
        fallback = []
        for idx, ln in enumerate(cleaned_lines):
            if idx in doc_lines:
                fallback.append(ln)
            elif _is_prose_line(ln):
                continue
            else:
                fallback.append(ln)
        cleaned_lines = fallback

    return "\n".join(cleaned_lines)

# FOR ENVIRONMENT ERRORS
## handle error: WrongObjectType
import ast

def fix_type_mismatches(code: str, type_map: Dict[str, str]) -> str:
    """
    주어진 코드 문자열에서 변수 이름별로 기대되는 생성자 타입(type_map)을 참고하여,
    실제 호출된 생성자가 다르면 기대 타입으로 교체한 코드를 반환한다.
    예: "tv_frame = Joint('tv_frame')" + {'tv_frame': 'Shape'} -> "tv_frame = Shape('tv_frame')"
    """
    class TypeFixer(ast.NodeTransformer):
        def visit_Assign(self, node):
            # 단순한 이름 대상(예: tv_frame = ...)만 처리
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                return self.generic_visit(node)
            
            # Extract the key_name from the constructor argument
            key_name = None
            if isinstance(node.value, ast.Call) and node.value.args:
                # Get the first argument if it's a string constant
                first_arg = node.value.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    key_name = first_arg.value
                elif isinstance(first_arg, ast.Str):  # For older Python versions
                    key_name = first_arg.s
            
            if not key_name:
                return self.generic_visit(node)
            
            try:
                expected_type = type_map.get(key_name)
            except Exception as e:
                print(e)
                return self.generic_visit(node)
            
            if expected_type and isinstance(node.value, ast.Call):
                func = node.value.func
                current_type = None
                if isinstance(func, ast.Name):
                    current_type = func.id
                elif isinstance(func, ast.Attribute):
                    current_type = func.attr  # module.Shape 같은 경우
                if current_type and current_type != expected_type:
                    # 생성자 이름을 교체 (네임스페이스가 있으면 attribute의 attr만 교체)
                    if isinstance(func, ast.Name):
                        node.value.func = ast.copy_location(
                            ast.Name(id=expected_type, ctx=ast.Load()), func
                        )
                    elif isinstance(func, ast.Attribute):
                        node.value.func = ast.copy_location(
                            ast.Attribute(value=func.value, attr=expected_type, ctx=ast.Load()), func
                        )
            return self.generic_visit(node)

    tree = ast.parse(code)
    fixer = TypeFixer()
    fixed_tree = fixer.visit(tree)
    ast.fix_missing_locations(fixed_tree)
    try:
        fixed_code = astor.to_source(fixed_tree)
    except AttributeError:
        raise RuntimeError("이 환경에서 astor.to_source를 사용할 수 없습니다.")
    return fixed_code

# tmp_code = """
# def get_object_positions(index: int):
#     a_obj = Joint('a')
#     b_obj = Shape('b')
#     return True

# def tmp2():
#     return False
# """

# tmp_dict = {
#     'a': 'Shape',
#     'b': 'Joint'
# }

# print(fix_type_mismatches(tmp_code, tmp_dict))

## handle error:  Handle does not exist
### 1. object 가져온 부분 다 찾기
### 2. dictionary에 없는데 작성한거 + dictionary에 있는데 안갖고온거 찾기
### 결국 LLM은 planning에만 집중할수 있도록 하는거..?

def adjust_indentation(code: str) -> str:
    """
    주어진 코드 문자열에서 각 줄의 indent를 정규화합니다.
    1) MODE(최빈 indent)를 base indent로 추정
    2) base indent를 제거하여 상대 indent만 남김
    3) 상대 indent를 4의 배수로 반올림
    """
    from collections import Counter

    lines = code.splitlines(keepends=True)
    non_empty = [ln for ln in lines if ln.strip()]
    if not non_empty:
        return code

    def lead(ln: str) -> int:
        return len(ln) - len(ln.lstrip(' '))

    # 코드 라인(주석 아닌)으로 base indent 계산
    code_lines = [ln for ln in non_empty
                  if not ln.lstrip().startswith('#')
                  and not ln.lstrip().startswith(('from ', 'import ', 'def '))]
    ref_lines = code_lines if code_lines else non_empty

    lead_counts = Counter(lead(ln) for ln in ref_lines)
    max_count = lead_counts.most_common(1)[0][1]
    base = min(indent for indent, count in lead_counts.items() if count == max_count)

    new_lines = []
    for line in lines:
        stripped = line.lstrip(' ')
        if not stripped.strip():
            new_lines.append(stripped)
            continue

        indent = len(line) - len(stripped)

        # base 제거 후 상대 indent 계산
        if indent >= base:
            relative = indent - base
        else:
            relative = 0

        # top-level (from/import/def)은 indent 0 유지
        if relative == 0 and stripped.startswith(("from ", "import ", "def ")):
            new_indent = 0
        else:
            # 상대 indent를 4의 배수로 반올림, 최소 4 (함수 body)
            if relative == 0:
                new_indent = 4
            elif relative % 4 == 0:
                new_indent = relative + 4
            else:
                new_indent = ((relative - 1) // 4 + 1) * 4 + 4

        new_lines.append(' ' * new_indent + stripped)

    return ''.join(new_lines)

def add_return_statement(code: str) -> str:
    tree = ast.parse(code)

    def _is_if_done(test: ast.AST) -> bool:
        # if done:
        if isinstance(test, ast.Name) and test.id == "done":
            return True

        # if (done == True) 또는 if (True == done)
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1:
            left, op, right = test.left, test.ops[0], test.comparators[0]
            is_true_const = lambda n: isinstance(n, ast.Constant) and n.value is True \
                                       or (isinstance(n, ast.Constant) and n.value is True)
            left_is_done = isinstance(left, ast.Name) and left.id == "done"
            right_is_done = isinstance(right, ast.Name) and right.id == "done"
            if isinstance(op, ast.Eq) and ((left_is_done and is_true_const(right)) or (right_is_done and is_true_const(left))):
                return True
            if isinstance(op, ast.Is) and ((left_is_done and is_true_const(right)) or (right_is_done and is_true_const(left))):
                return True

        # 그 외 (예: if not done:) 은 제외
        return False

    class _ForceReturn(ast.NodeTransformer):
        def visit_If(self, node: ast.If):
            self.generic_visit(node)
            if _is_if_done(node.test):
                if not node.body or not isinstance(node.body[-1], ast.Return):
                    # astunparse가 구버전 호환으로 처리할 수 있도록 Name 사용
                    node.body.append(
                        ast.Return(
                            value=ast.Name(id='True', ctx=ast.Load())
                        )
                    )
            return node

    new_tree = _ForceReturn().visit(tree)
    ast.fix_missing_locations(new_tree)
    # print("여기까진 됨.")
    return astunparse.unparse(new_tree)

def add_indent(text, indent_spaces=4):
    indent = ' ' * indent_spaces
    return '\n'.join(indent + line if line.strip() else line for line in text.splitlines())

def remove_is_triggered_calls(code: str) -> str:
    """
    Replace .is_triggered() calls with 'done', as ProximitySensor does not have this method.
    e.g. 'if success.is_triggered():' -> 'if done:'
    This prevents AttributeError: 'ProximitySensor' object has no attribute 'is_triggered'
    """
    # Replace any <var>.is_triggered() with 'done'
    code = re.sub(r'[A-Za-z_][A-Za-z0-9_]*\.is_triggered\(\)', 'done', code)
    return code

def postprocess(code: str, type_map: Dict[str, str] = None, initial_instruction: str = None):
    # initial_instruction 제외하고 postprocess 수행한 이후 다시 결합
    temp_interface = "def temp_interface():\n"

    code, is_code_block = extract_from_code_block(code)
    if is_code_block:
        code = add_indent(code)
    else:
        code = code.replace("```python", "").replace("```", "").strip()
    if initial_instruction:
        code = temp_interface + code
    code = adjust_indentation(code)
    # print("adjust_indentation Code:\n", code)
    code = remove_non_code_text(code)
    # print("remove_non_code_text Code:\n", code)

    # Remove is_triggered() calls
    code = remove_is_triggered_calls(code)

    # code = fix_type_mismatches(code, type_map)
    # code = add_return_statement(code)
    # print("add_return_statement Code:\n", code)
    if initial_instruction:
        code = code.replace(temp_interface, "")
        # code = CODE_HEADER + "\n" + initial_instruction + code
        code = initial_instruction + code
    return code

# with open('/workspace/rlbench_server/utils/tmp.py', 'r') as f:
#     initial_code = f.read()
# with open('/workspace/rlbench_server/tasks/ReachTarget.py', 'r') as f:
#     original_code = f.read()

# new_code = postprocess(original_code, initial_instruction=initial_code)
# print("Extracted Code Block:\n", new_code)