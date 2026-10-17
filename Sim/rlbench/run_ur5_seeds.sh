#!/usr/bin/env bash
set -euo pipefail
cd ~/CSI_agent/CaP_RepairAgent_new/Sim/rlbench

python -u rlbench_exp.py --mode failure -r ur5 --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 42 --no_resume --output_dir ./results/ur5_seed42 \
  > logsur42.out 2>&1

python -u rlbench_exp.py --mode failure -r ur5 --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 123 --no_resume --output_dir ./results/ur5_seed123 \
  > logsur123.out 2>&1

python -u rlbench_exp.py --mode failure -r ur5 --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 2024 --no_resume --output_dir ./results/ur5_seed2024 \
  > logsur2024.out 2>&1
