# Embodied RepairAgent — Robot Code Repair in Simulation

Research-code archive for iterative repair of generated robot-control programs in **RLBench** and **Genesis**. The work was performed in a collaborative research setting; this repository is a curated portfolio snapshot, **not a verified reproduction package** or a claim of sole authorship of every included module.

## Approach

1. Generate a task policy as Python code from a source-robot reference and environment observations.
2. Execute the policy in a target-robot simulator.
3. Detect execution or task-completion failures and extract error/scene context.
4. Request code repair or additional steps from the LLM, and retry within configured limits.
5. Record per-task/subtask outcomes and inference timings.

The historical experiment configurations use `model=code_agent` as an **agent mode**. This does **not** identify the LLM checkpoint, and the checkpoint used for the archived runs has not been confirmed.

## Code layout

- `rlbench/`: RLBench experiment driver, AST-based execution/repair loop, model-request client, environment utilities, and robot primitives (selected from the RLBench backup).
- `genesis/`: Genesis experiment driver, task/subtask loop, repair provider and support modules (selected from the Genesis backup).
- `examples/`: selected robot-skill examples reported as authored by the repository maintainer.
- `results/`: compact tables of *archived* results, without large raw traces or generated code.
- `SOURCE_PROVENANCE.md`: pinned source commits, limitations, and code-origin details.
- `ATTRIBUTION_REVIEW.md`: collaborative authorship and redistribution notes.

Some directories may remain incomplete pending attribution and dependency review. The files are historical snapshots and have **not been end-to-end tested** in a fresh environment.

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
