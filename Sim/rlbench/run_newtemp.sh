#!/usr/bin/env bash
set -euo pipefail

# Run RLBench experiments sequentially and log outputs.
# Stop immediately if any command fails.

#python -u rlbench_exp.py --yaml_path ./static/all_tasks.yml --mode failure -s panda -r ur5 --model code_agent --max_repairs 5 --seed 42   --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/ur5temp42_all   > newtempur42_all.out 2>&1
#python -u rlbench_exp.py --yaml_path ./static/all_tasks.yml --mode failure -s panda -r ur5 --model code_agent --max_repairs 5 --seed 123  --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/ur5temp123_all  > newtempur123_all.out 2>&1
#python -u rlbench_exp.py --yaml_path ./static/all_tasks.yml --mode failure -s panda -r ur5 --model code_agent --max_repairs 5 --seed 2024 --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/ur5temp2024_all > newtempur2024_all.out 2>&1
#python -u rlbench_exp.py --mode failure -s panda -r ur5 --model code_agent --max_repairs 5 --seed 123 --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/ur5temp123 > newtempur123.out 2>&1

python -u rlbench_exp.py --mode failure -s panda -r sawyer --model code_agent --max_repairs 5 --seed 123 --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/sawyertemp123 > newtempsawyer123.out 2>&1
python -u rlbench_exp.py --mode failure -s panda -r sawyer --model code_agent --max_repairs 5 --seed 123 --remote_host 127.0.0.1 --remote_port 9000 --output_dir ./results_newtemp/sawyertemp2024 > newtempsawyer2024.out 2>&1


echo "All runs completed successfully."