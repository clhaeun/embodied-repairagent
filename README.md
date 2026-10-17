# Embodied RepairAgent — Robot Code Repair in Simulation

Research-code archive for iterative repair of generated robot-control programs in **RLBench** and **Genesis**. The work was performed in a collaborative research setting; this repository is a curated portfolio snapshot, **not a verified reproduction package** or a claim of sole authorship of every included module.

## Approach

1. Generate a task policy as Python code from a source-robot reference and environment observations.
2. Execute the policy in a target-robot simulator.
3. Detect execution or task-completion failures and extract error/scene context.
4. Request code repair or additional steps from the LLM, and retry within configured limits.
5. Record per-task/subtask outcomes and inference timings.

The historical experiment configurations use `model=code_agent` as an **agent mode**. This does **not** identify the LLM checkpoint, and the checkpoint used for the archived runs has not been confirmed.

## Repository contents

- `Sim/rlbench/`: archived RLBench experiment driver (`rlbench_exp.py`), AST-based execution and repair loop (`rlbench_loop.py`), LLM policy provider, robot primitive implementations, task definitions, prompts, utilities and run scripts.
- `Sim/rlbench/RLBench/`: Git submodule pointing to [clhaeun/RLBench](https://github.com/clhaeun/RLBench) at pinned commit `5aba50f8`. Initialize with `git submodule update --init --recursive`.
- `Sim/genesis/`: archived Genesis experiment driver (`genesis_experiment.py`), subtask execution loop (`genesis_loop.py`), LLM policy provider, task definitions, prompts, configuration and environment helpers.
- `servers/rlbench_server.py`: RLBench inference server copied from the RLBench home backup (original name: `server.py`).
- `servers/genesis_server.py`: Genesis inference server copied from the Genesis home backup (original name: `server.py`).
- `examples/oracle_skills/`: seven user-identified original robot task-policy examples.
- `results/`: condensed archival experimental summary CSVs, *not* full raw result sets.
- `SOURCE_PROVENANCE.md`, `ATTRIBUTION_REVIEW.md`: pinned origins, caveats, and collaboration attribution.
- `RUNTIME_NOTES.md`: current dependency and reproduction caveats.

The **top-level `skill_codes/` is intentionally excluded** as requested. The similarly named `Sim/rlbench/skill_code*.py` and `Sim/genesis/skill_code.py` are included because the simulation loops import their robot control primitives.

### How to retrieve the historical runtime

```bash
git clone --recurse-submodules https://github.com/clhaeun/embodied-repairagent.git
cd embodied-repairagent
# or, if already cloned:
git submodule update --init --recursive
```

**Not a verified turn-key reproduction:** The Genesis simulator scene/robot meshes and other original `Sim/genesis/assets/` resources (approximately 199 MB) have **not** been copied. They may be required at runtime and must be obtained from their original location with appropriate redistribution rights. Historical command scripts also contain environment-specific file paths and host addresses. The runtime, simulator/LLM checkpoints, model weights, and old environment are not independently available or tested. See `RUNTIME_NOTES.md`.

This is a curated collaborative research archive, **not a claim that every copied module was authored solely by the maintainer**. Keep original credits and review redistribution licenses before relying on these files.

## Archived experimental evidence (not final paper results)

| Simulator / transfer | Recorded runs | Recorded task success | Recorded subtask success |
|---|---|---|---|
| RLBench, Panda → Sawyer (Feb 11, 2026) | 3 runs (seeds unverified) | 9/20 in each run | 46/74, 43/74, 43/74 |
| Genesis, Panda → Robotiq (Feb 27, 2026) | seeds 42, 123, 2024 | 7/12 in each run | 47/64, 48/64, 48/64 |
| Genesis, Panda → Suction (Feb 27, 2026) | seeds 42, 123, 2024 | 6/12 in each run | 38/48, 38/48, 37/48 |

**Important qualifications:** The RLBench `final_result.json` files do not store the random seed. One archived script even labels an output directory `sawyertemp2024` but passes `--seed 123`. The exact LLM checkpoint and runtime commit are not recorded in the sampled results. May 1 code backups should not be treated as proven identical to the code running in February. These numbers are archived observations, **not** controlled reproducibility claims or final ICML paper metrics.

Evidence:
- [RLBench archival results](https://github.com/clhaeun/CaP_RepairAgent_new/tree/chaeun/home-backup-2026-05-01/Sim/rlbench/results_newtemp)
- [RLBench run script](https://github.com/clhaeun/CaP_RepairAgent_new/blob/chaeun/home-backup-2026-05-01/Sim/rlbench/run_newtemp.sh)
- [Genesis archival results](https://github.com/clhaeun/CaP_RepairAgent_new/tree/CaP_RepairAgent_genesis-home-backup-2026-05-01/Sim/genesis/results_0227_2)
- [Genesis run script](https://github.com/clhaeun/CaP_RepairAgent_new/blob/CaP_RepairAgent_genesis-home-backup-2026-05-01/Sim/genesis/run_exp0227.sh)

## Provenance, collaborators, and reproducibility

Selected historical files originate from [clhaeun/CaP_RepairAgent_new](https://github.com/clhaeun/CaP_RepairAgent_new), which includes contributions from other researchers and SkillCache-related work. The other researcher's separate `baseline/` directory is not republished here. A clean simulator environment, asset licenses, exact LLM model, and complete dependency set have **not** been independently verified. See the provenance and attribution notes before reusing code.

No general open-source license has been granted for this curated code archive; repository visibility alone does not establish permission to reuse third-party code.
