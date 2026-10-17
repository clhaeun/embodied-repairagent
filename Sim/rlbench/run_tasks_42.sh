#!/bin/bash

# 실행할 Task 목록 정의 (요청하신 순서대로, MeatOffGrill 중복 포함)
tasks=(
    "PickAndLift" 
    "PushButton" 
    "ChangeChannel" 
    "CloseBox"
    "HitBallWithQueue"
    "LampOff"
    "LampOn" 
    "MeatOffGrill" 
    "MeatOnGrill" 
    "PhoneOnBase" 
    "PutKnifeOnChoppingBoard"
    "PutMoneyInSafe"
    "PutRubbishInBin"
    "PutUmbrellaInUmbrellaStand"
    "ReachAndDrag"
    "WipeDesk"
)

# 로그 파일 초기화 여부 (기존 로그 뒤에 붙이려면 이 줄을 삭제하세요)
# echo "New Batch Run Start: $(date)" > logslooptasks.out

echo "Starting sequential execution..."

for task in "${tasks[@]}"
do
    echo "==================================================" >> logslooptasks.out
    echo "Running Task: $task | Date: $(date)" >> logslooptasks.out
    echo "Command: python rlbench_loop.py --task $task -r sawyer --seed 42" >> logslooptasks.out
    echo "==================================================" >> logslooptasks.out

    # 파이썬 스크립트 실행
    # -u: 파이썬 출력을 버퍼링 없이 바로 파일에 쓰기 위함
    # 2>&1: 에러 메시지(stderr)도 표준 출력(stdout)과 함께 로그 파일에 저장
    python -u rlbench_loop.py \
        --task "$task" \
        --mode failure \
        --target_robot sawyer \
        --seed 42 \
        >> logslooptasks.out 2>&1
        
    echo "Finished: $task"
done

echo "All tasks completed. Check logslooptasks.out"