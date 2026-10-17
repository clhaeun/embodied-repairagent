#!/usr/bin/env bash
set -euo pipefail
cd ~/CSI_agent/CaP_RepairAgent_new/Sim/rlbench

python -u rlbench_exp.py --mode failure -r sawyer --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 42 --no_resume --output_dir ./results/sawyer_seed42 \
  > logssaw42.out 2>&1

python -u rlbench_exp.py --mode failure -r sawyer --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 123 --no_resume --output_dir ./results/sawyer_seed123 \
  > logssaw123.out 2>&1

python -u rlbench_exp.py --mode failure -r sawyer --model code_agent --remote_host 127.0.0.1 --remote_port 9000 \
  --seed 2024 --no_resume --output_dir ./results/sawyer_seed2024 \
  > logssaw2024.out 2>&1
