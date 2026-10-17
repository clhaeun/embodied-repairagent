#!/usr/bin/env bash
set -euo pipefail
# cd ~/nas/CSI_agent/CaP_RepairAgent_new/Sim/rlbench
# cd /home/nas/chaeun/CSI_agent/CaP_RepairAgent_new/Sim/rlbench
cd /home/chaeun/CSI_agent/CaP_RepairAgent_new/Sim/rlbench

python3 -u rlbench_exp.py --mode failure -s panda -r sawyer --model code_agent --max_repairs 10 --remote_host 127.0.0.1 --remote_port 9000 \
  --no_resume --output_dir ./results/p_s4 \
  > p_s4.out 2>&1

python3 convert_task.py -t ./tasks_sawyer:sawyer > convert_task4.out 2>&1

python3 -u rlbench_exp.py --mode failure -s sawyer -r ur5 --model code_agent --max_repairs 10 --remote_host 127.0.0.1 --remote_port 9000 \
  --no_resume --output_dir ./results/s_u4 \
  > s_u4.out 2>&1

