# Source provenance and reproducibility

The curated source candidates come from historical branches in [CaP_RepairAgent_new](https://github.com/clhaeun/CaP_RepairAgent_new). This repository does **not** import the original Git history.

| Scope | Snapshot commit | Description |
|---|---|---|
| RLBench | `eb14553653e4ba6c4343ffb500a6a7e59268fd9a` | 2026-05-01 home backup |
| Genesis | `a20df7423e394a9248a33030c40a4e2958262c29` | 2026-05-01 Genesis home backup |

The archived RLBench evaluation (Feb 11) precedes the May backup, whose working code changed during the intervening months. The Genesis main loop and policy provider were recorded in January commits preceding its Feb 27 evaluation, but the exact runtime tree remains unverified.

Archived evaluation output stores `code_agent` (program mode), **not** a confirmed LLM checkpoint. The RLBench final results do not record seed; source script `run_newtemp.sh` labels one output `sawyertemp2024` while supplying `--seed 123`.

Source folders and supporting evidence:
- [RLBench snapshot](https://github.com/clhaeun/CaP_RepairAgent_new/tree/chaeun/home-backup-2026-05-01/Sim/rlbench)
- [Genesis snapshot](https://github.com/clhaeun/CaP_RepairAgent_new/tree/CaP_RepairAgent_genesis-home-backup-2026-05-01/Sim/genesis)
- [RLBench results](https://github.com/clhaeun/CaP_RepairAgent_new/tree/chaeun/home-backup-2026-05-01/Sim/rlbench/results_newtemp)
- [Genesis results](https://github.com/clhaeun/CaP_RepairAgent_new/tree/CaP_RepairAgent_genesis-home-backup-2026-05-01/Sim/genesis/results_0227_2)

The source repository includes collaborator-authored modules. No clean end-to-end rerun or third-party licensing audit has been performed.
