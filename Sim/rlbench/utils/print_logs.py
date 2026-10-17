from termcolor import cprint

def main_print(text):
    cprint(f"[Main]  {text}", "blue")

def error_print(text):
    cprint(f"[Error]  {text}", "red")

def start_print():
    cprint("---------- EXECUTES NEW TASK ----------", "red")

def success_print(text):
    cprint(f"[Success] {text}", "green")

def feedback_print(text):
    cprint(f"[Feedback_Start]\n{text}\n[Feedback_End]", "yellow")