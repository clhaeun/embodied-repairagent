# RepairAgent for Robot Skill Transfer

Code generation and repair for transferring robot manipulation skills between different robot embodiments. Experiments were conducted in RLBench and Genesis.

Given a source-robot skill and a target scene, the agent generates Python control code for the target robot. The generated code is executed in simulation and updated when an action fails or the task remains incomplete.

## Execution

The `code_agent` path uses three LLM requests:

- **Generate:** produce initial target-robot code from the source code and scene information.
- **Repair:** update the code after an execution error or skill failure, using the failure information and current scene.
- **Revise:** generate additional code when execution ends before the task is completed.

Execution uses a bounded repair loop. RLBench runs generated statements through an AST-based executor; Genesis applies the same generate–repair–revise workflow at the subtask level. The code also contains an `ours` branch, but the experimental runs listed below used `code_agent`.

## Code

| Component | Main files |
|---|---|
| RLBench experiment runner | [`rlbench_exp.py`](Sim/rlbench/rlbench_exp.py) |
| RLBench execution and repair | [`rlbench_loop.py`](Sim/rlbench/rlbench_loop.py), [`policy_provider.py`](Sim/rlbench/policy_provider.py) |
| Genesis experiment runner | [`genesis_experiment.py`](Sim/genesis/genesis_experiment.py) |
| Genesis subtask execution and repair | [`genesis_loop.py`](Sim/genesis/genesis_loop.py), [`policy_provider.py`](Sim/genesis/policy_provider.py) |
| Inference servers | [RLBench](servers/rlbench_server.py), [Genesis](servers/genesis_server.py) |
| Task code and prompts | `Sim/rlbench/tasks_*/`, `Sim/genesis/subtasks_*/`, `Sim/*/prompts/` |

## Experiments

The following figures are from the saved February 2026 experiment logs, not the final paper evaluation.

| Simulator | Transfer | Agent / execution mode | Task success | Subtask success |
|---|---|---|---|---|
| RLBench (Feb 11) | Panda → Sawyer | `code_agent`, `failure` | 9/20 in each of 3 runs | 46/74, 43/74, 43/74 |
| Genesis (Feb 27) | Panda → Robotiq | `code_agent`, failure-based repair | 7/12 for seeds 42, 123, 2024 | 47/64, 48/64, 48/64 |
| Genesis (Feb 27) | Panda → Suction | `code_agent`, failure-based repair | 6/12 for seeds 42, 123, 2024 | 38/48, 38/48, 37/48 |

All these runs used `remote_llm`. Both archived inference-server implementations default to `Qwen/Qwen2.5-Coder-7B`, but the result files do not identify which LLM checkpoint was actually loaded. The RLBench result files also omit the seed; their directory labels should not be treated as verified seeds.

See [RLBench results](results/rlbench_archived_runs.csv) and [Genesis results](results/genesis_archived_runs.csv) for the individual run records.

## Dependencies

- **RLBench / PyRep / CoppeliaSim:** the modified [RLBench fork](https://github.com/clhaeun/RLBench) is pinned as a Git submodule at `Sim/rlbench/RLBench`.
- **Genesis:** the experiment code uses `genesis-world`; robot models and simulator assets are external dependencies and are not bundled.
- **LLM server:** separate inference processes are implemented under `servers/`.

The source files were collected from the May 1, 2026 research backups. See [source versions](SOURCE_PROVENANCE.md) and [runtime notes](RUNTIME_NOTES.md) for their origins and environment details.

This code was developed and tested during a collaborative research project; some simulator integrations and supporting utilities were shared within the team.
