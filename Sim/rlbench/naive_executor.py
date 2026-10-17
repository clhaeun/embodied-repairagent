# 20251215 작성

import argparse
import importlib
import numpy as np

from env import setup_environment, shutdown_environment

from utils.trigger_condition import SkillFailure, PathOutOfWorkspace
from utils.helper import object_names
from utils.feedback import Feedback, FeedbackWithError

from rlbench.gym import RLBenchEnv
from utils.code_postprocess import postprocess

def patched_close(self) -> None:
    pass

RLBenchEnv.close = patched_close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an arbitrary RLBench task by class name"
    )
    parser.add_argument(
        "--robot", "-r",
        default="panda",
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        help="RLBench task class name, e.g. PushButton, OpenDrawer, PutRubbishInBin, MyCustomTask"
    )
    args = parser.parse_args()
    robot_type = args.robot.lower()

    # rlbench.tasks 모듈에서 클래스 동적 로드
    tasks_module = importlib.import_module(f"rlbench.tasks")
    try:
        task_cls = getattr(tasks_module, args.task)
    except AttributeError:
        raise SystemExit(f"Error: '{args.task}' 라는 태스크 클래스를 찾을 수 없습니다.")
    
    # env setup
    env, task = setup_environment(task_cls, robot_type.lower())

    descriptions, obs = task.reset()
    
    objects, objects_dict, object_names = object_names(task)
    print(f"[Main] [Object_Start]\nObjects in the task: {objects}\n[Object_End]")
    
    timestep = 0

    while True:
        if env.is_shutdown():
            print("[Main] Environment is shutdown. Please restart the environment.")
            break
        
        cmd = input(">>> ")

        if cmd.strip().lower() in ("exit", "quit"):
            print("\n[Main] Exiting RLBench interactive console.")
            shutdown_environment(env)
            break

        if cmd.strip().lower() == "restart":
            print("\n[Main] Restarting the task...")
            descriptions, obs = task.reset()
            print("[Main] Task restarted successfully.")
            continue
         
        if cmd.strip().lower() == "run":
            timestep += 1

            module = importlib.import_module(f"tasks_{robot_type}.{args.task}")
            importlib.reload(module)
            run_task = getattr(module, "run_skill")
            run_task(env, task, descriptions, obs)

            # success 여부 확인
            success, terminate = task._task.success()
            if success:
                print("[Main] [Feedback] Task completed successfully! [Main]")
                shutdown_environment(env)
            else:
                print("[Main] [Feedback] Task not completed. [Main]")
    