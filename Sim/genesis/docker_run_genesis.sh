# 이미지 빌드
cd /home/sera/workspace/Genesis
docker build -f docker/Dockerfile -t genesis-sim:latest .

# 컨테이너 실행
docker run -it --rm \
    --name genesis-container \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
    -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir \
    -e PULSE_SERVER=/mnt/wslg/PulseServer \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /mnt/wslg:/mnt/wslg \
    -v /home/sera/SkillCache/Sim/genesis:/workspace/genesis_exp \
    -e LOCAL_USER_ID=$(id -u) \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    genesis-sim:latest