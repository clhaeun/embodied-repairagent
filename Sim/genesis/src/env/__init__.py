from .env import GenesisEnv
from .lmp_wrapper import LMPWrapper
from .tasks import *


TASK_REGISTRY = {
    "composite": CompositeTask,
    "pick_place": PickPlaceTask,
    "put_in_hinge": PutInHingeTask,
    "put_in_prismatic": PutInPrismaticTask,
    "sweep": SweepTask,
    "pick_place_ball": PickPlaceBallTask,
    "pick_place_cube": PickPlaceCubeTask,
    "pick_place_cylinder": PickPlaceCylinderTask,
    "pick_place_inverse_ball": PickPlaceInverseBallTask,
    "pick_place_inverse_cube": PickPlaceInverseCubeTask,
    "pick_place_inverse_cylinder": PickPlaceInverseCylinderTask,
    "insert_slot": InsertSlotTask,
    "pour_liquid": PourLiquidTask,
}


def build_env():
    env = LMPWrapper(GenesisEnv())
    return env


def build_task(subtask_configs, variant=0):
    for subtask in subtask_configs:
        if isinstance(subtask["task"], str):
            subtask["task"] = TASK_REGISTRY[subtask["task"]]

    task = CompositeTask(subtask_configs, variant)
    return task
