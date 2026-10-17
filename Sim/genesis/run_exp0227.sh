#!/bin/bash

echo "=== 실험 시작 ==="

# robotiq 123
echo "[1/5] Robotiq 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results_0227_2/robotiq123 -s panda -r robotiq --model code_agent --remote_host 115.145.173.12 --remote_port 9000 --seed 123 --no_resume > 0227_ro123.out


# robotiq 2024
echo "[2/5] Robotiq 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results_0227_2/robotiq2024 -s panda -r robotiq --model code_agent --remote_host 115.145.173.12 --remote_port 9000 --seed 2024 --no_resume > 0227_ro2024.out


# suction 42
echo "[3/5] suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results_0227_2/suction42 -s panda -r suction --model code_agent --remote_host 115.145.173.12 --remote_port 9000 --seed 42 --no_resume > 0227_su42.out


# suction 123
echo "[4/5] suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results_0227_2/suction123 -s panda -r suction --model code_agent --remote_host 115.145.173.12 --remote_port 9000 --seed 123 --no_resume > 0227_su123.out


# suction 2024
echo "[5/5] Suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results_0227_2/suction2024 -s panda -r suction --model code_agent --remote_host 115.145.173.12 --remote_port 9000 --seed 2024 --no_resume > 0227_su2024.out



echo "=== 모든 실험이 완료되었습니다. ==="