# Embodied RepairAgent

A code-generation and recovery baseline for transferring robot manipulation skills across embodiments. Developed as part of the research reported in [*Efficient Skill Grounding via Code Refactoring with Small Language Models* (ICML 2026)](https://arxiv.org/abs/2606.07999).

This repository contains the **RepairAgent baseline**, not the paper's proposed RECENT framework.

## Approach

Given a skill written for a source robot and a target scene, the agent generates executable control code and evaluates it in simulation. Recovery distinguishes between:

- **Generate:** write an initial target-robot program.
- **Repair:** address execution errors or skill failures using the failure signal and current scene state.
- **Revise:** continue from the current environment state if the program finishes without completing the task.

The key distinction is between code execution and task completion: syntactically valid, successfully executed code does not necessarily achieve the intended robot state. The RLBench implementation includes AST-based code execution and post-processing.

## Implementation

| Component | Source |
| --- | --- |
| RLBench runner | [`rlbench_exp.py`](Sim/rlbench/rlbench_exp.py) |
| RLBench recovery loop | [`rlbench_loop.py`](Sim/rlbench/rlbench_loop.py), [`policy_provider.py`](Sim/rlbench/policy_provider.py) |
| Genesis runner and loop | [`genesis_experiment.py`](Sim/genesis/genesis_experiment.py), [`genesis_loop.py`](Sim/genesis/genesis_loop.py) |
| Inference servers | [RLBench](servers/rlbench_server.py), [Genesis](servers/genesis_server.py) |
| Archived evaluation summaries | [RLBench](results/rlbench_archived_runs.csv), [Genesis](results/genesis_archived_runs.csv) |

## Selected experiments

| Environment and transfer | Task success | Subtask success |
| --- | --- | --- |
| RLBench, Panda → Sawyer | 9/20 in each of three archived runs | 46/74, 43/74, 43/74 |
| Genesis, Panda → Robotiq | 7/12 for each of three recorded seeds | 47/64, 48/64, 48/64 |
| Genesis, Panda → Suction | 6/12 for each of three recorded seeds | 38/48, 38/48, 37/48 |

The archived runs used the `code_agent` execution mode with a remote LLM. In the **RLBench** records, seed numbers are inferred from folder names rather than verified in the result JSON; one launch script also disagrees with a folder label. The exact model checkpoint is not recorded in either archived summary. These results describe the baseline runs, **not** the paper's main-method performance.

## Dependencies and scope

The modified [RLBench fork](https://github.com/clhaeun/RLBench) is referenced as a Git submodule at `Sim/rlbench/RLBench`. RLBench/PyRep/CoppeliaSim, Genesis robot assets, and remote inference services are separate dependencies. This is an experimental research codebase rather than a packaged, standalone reproduction.
