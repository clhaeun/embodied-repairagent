#!/bin/bash

echo "=== 실험 시작 ==="

# 1. Robotiq 실험 (Viewer off)
# echo "[1/3] Robotiq 실험 진행 중..."
# python -u genesis_experiment.py --output_dir ./results/robotiq42 -s panda -r robotiq --model code_agent --seed 42 --no_resume > exp_robotiq42.out

# echo "[1/3] Robotiq 실험 진행 중..."
# python -u genesis_experiment.py --output_dir ./results/robotiq123 -s panda -r robotiq --model code_agent --seed 123 --no_resume > exp_robotiq123.out

# echo "[1/3] Robotiq 실험 진행 중..."
# python -u genesis_experiment.py --output_dir ./results/robotiq2024 -s panda -r robotiq --model code_agent --seed 2024 --no_resume > exp_robotiq2024.out


# 2. Suction 실험 (Viewer off)
# echo "[2/2] Suction 실험 진행 중..."
# python -u genesis_experiment.py --output_dir ./results/suction42 -s panda -r suction --model code_agent --seed 42 --no_resume > exp_suction42.out

echo "[2/2] Suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_suction123 -s panda -r suction --model code_agent --seed 123 --no_resume > exp1_suction123.out

echo "[2/2] Suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_suction2024 -s panda -r suction --model code_agent --seed 2024 --no_resume > exp1_suction2024.out

echo "=== 모든 1차 실험이 완료되었습니다. ==="


# 1. Robotiq 실험 (Viewer off)
echo "[1/3] Robotiq 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_robotiq42 -s panda -r robotiq --model code_agent --seed 42 --no_resume > exp1_robotiq42.out

echo "[2/3] Robotiq 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_robotiq123 -s panda -r robotiq --model code_agent --seed 123 --no_resume > exp1_robotiq123.out

echo "[3/3] Robotiq 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_robotiq2024 -s panda -r robotiq --model code_agent --seed 2024 --no_resume > exp1_robotiq2024.out


# 2. Suction 실험 (Viewer off)
echo "[2/2] Suction 실험 진행 중..."
python -u genesis_experiment.py --output_dir ./results/_suction42 -s panda -r suction --model code_agent --seed 42 --no_resume > exp1_suction42.out