# Archived runtime notes

## Scope and sources

| Component | Source commit | Archived path |
|---|---|---|
| RLBench experiment code | `eb14553653e4ba6c4343ffb500a6a7e59268fd9a` | `Sim/rlbench/` |
| Genesis experiment code | `a20df7423e394a9248a33030c40a4e2958262c29` | `Sim/genesis/` |
| RLBench fork submodule | `5aba50f842bc73ab03e11a739bf9b67816d45e23` | `Sim/rlbench/RLBench/` |

The research archives contain joint work; this is a selective content copy, **not** a transplant of collaborators' commit history. Original filenames are preserved inside `Sim/`; root-level inference servers are separated into `servers/` because the two source branches each contained a different `server.py`.

## RLBench call graph

```text
Sim/rlbench/rlbench_exp.py
    -> rlbench_loop.run_task_episode()
        -> env / RLBench / PyRep
        -> policy_provider.PolicyProvider()
            -> remote_llm_socket_client.RemoteLLMSocketClient()
                -> servers/rlbench_server.py (run separately)
        -> task policy + skill_code*.py + utils + prompts
```

Example (not rerun or validated here):

```bash
git submodule update --init --recursive
cd Sim/rlbench
python -u rlbench_exp.py --mode failure -s panda -r sawyer --model code_agent --seed 42 --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results/reproduction_trial
```

Start the appropriate remote LLM service separately and install the dependencies in `requirements*.txt` and the pinned RLBench fork. Historical shell scripts may have source-machine absolute paths, addresses and mislabeled seeds. Read their contents before running them.

## Genesis call graph

```text
Sim/genesis/genesis_experiment.py
    -> genesis_loop.run_task_episode()
        -> GenesisEnv / src.env / tasks / robot assets
        -> policy_provider.PolicyProvider()
            -> prompts + remote_llm_socket_client + subtask implementation
                -> servers/genesis_server.py (run separately)
```

Example from archived Feb 27 run script, with historical external host deliberately replaced by localhost (not validated):

```bash
cd Sim/genesis
python -u genesis_experiment.py --output_dir ./results/test_robotiq123 -s panda -r robotiq --model code_agent --seed 123 --no_resume --remote_host 127.0.0.1 --remote_port 9000
```

## Missing dependencies and limitations

- **Genesis assets:** the original branch has roughly 199 MB under `Sim/genesis/assets/` (robot meshes, scene models, etc.). They are not included here because they are sizable third-party resources whose license/redistribution terms need checking. Some Genesis experiment modules require this exact directory, so the archived Genesis experiment cannot be assumed runnable until the required assets have been restored.
- **External simulators:** RLBench (submodule linked), CoppeliaSim/PyRep, and Genesis need a supported installation/environment.
- **Exact model versions:** archived JSON shows `model=code_agent` but not an unambiguous runtime LLM checkpoint. The inference server code contains model flags/defaults that do not prove which model was actually loaded.
- **Historical execution identity:** the May snapshots were committed after the February experiments. They are evidence of the backed-up implementation, **not proof that every line matches the execution-time working tree**.
- **Skill code distinction:** requested exclusion `skill_codes/` (top-level) is respected. Experiment-local `skill_code*.py` is necessary for Python imports and intentionally retained.
- **Experiment output:** full `results_*/`, prompts/logs/checkpoints/LLM-generated skill code are kept at source branches and are not copied; only aggregate CSVs are included here.
- **Rights and attribution:** source is collaborative and may incorporate third-party implementations. No blanket license is asserted, and no claim of sole authorship is made.
