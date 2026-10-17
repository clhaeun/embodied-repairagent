import re
import os
import random

from colorama import Fore, Style
import torch
import numpy as np


def print_info(message):
    print(Fore.BLUE + message + Style.RESET_ALL)


def print_error(message):
    print(Fore.RED + message + Style.RESET_ALL)


def format_result(result_list):
    colored = []
    for r in result_list:
        if r == "full_success":
            colored.append(Fore.GREEN + r + Style.RESET_ALL)
        elif r == "partial_success":
            colored.append(Fore.YELLOW + r + Style.RESET_ALL)
        elif r == "fail":
            colored.append(Fore.RED + r + Style.RESET_ALL)
        else:
            colored.append(r)
    return "[" + ", ".join(colored) + "]"


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_ordered_list(response):
    lines = response.splitlines()
    items = []
    for line in lines:
        match = re.match(r"^\d+\.\s*(.*)", line.strip())
        if match:
            items.append(match.group(1))
    return items


def extract_unordered_list(response):
    return [line.strip("- ") for line in response.splitlines() if line.startswith("-")]


def extract_code_block(response):
    match = re.search(r"```(?:\w*\n)?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_section(text, header):
    header_pattern = re.escape(header)
    match = re.search(header_pattern, text, re.IGNORECASE)

    if not match:
        return ""

    start_pos = match.end()

    next_headers = [
        "Action Semantic:",
        "Precondition:",
        "Effect:",
        "* Prediction rationale:",
        "* Predicted Action:",
        "* Refinement rationale:",
        "* Refinement patch:",
    ]

    end_pos = len(text)
    for next_header in next_headers:
        if next_header == header:
            continue
        next_match = re.search(re.escape(next_header), text[start_pos:], re.IGNORECASE)
        if next_match:
            candidate_end = start_pos + next_match.start()
            end_pos = min(end_pos, candidate_end)

    content = text[start_pos:end_pos].strip()
    return content


def join_ordered_list(items):
    return "\n".join([f"{i + 1}. {item}" for i, item in enumerate(items)])


def join_unordered_list(items):
    return "\n".join([f"- {item}" for item in items])

def get_color(name):
    colors = {
        "red": (0.8, 0.2, 0.2),
        "green": (0.2, 0.8, 0.2),
        "blue": (0.2, 0.2, 0.8),
        "yellow": (0.95, 0.85, 0.2),
        "orange": (0.95, 0.55, 0.2),
        "purple": (0.6, 0.3, 0.8),
        "gray": (0.6, 0.6, 0.6),
        "silver": (0.75, 0.75, 0.75),
        "black": (0.1, 0.1, 0.1),
        "white": (0.95, 0.95, 0.95),
    }
    return colors.get(name.lower(), (0.5, 0.5, 0.5))