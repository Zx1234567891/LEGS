#!/bin/bash
# LEGS Navigation Demo — PyBullet + NWM + MCTS
#
# Usage:
#   ./scripts/run_navigation_demo.sh                    # 默认: 离线MCTS模式, 室内场景
#   ./scripts/run_navigation_demo.sh --scene=maze       # 迷宫场景
#   ./scripts/run_navigation_demo.sh --with-server      # 启动Server + Dog
#   ./scripts/run_navigation_demo.sh --no-gui           # 无GUI模式
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/packages/legs_common/src:${PROJECT_ROOT}/packages/legs_dog/src:${PROJECT_ROOT}/packages/legs_server/src"

SCENE="indoor"
GOAL="4.0,0.0"
WITH_SERVER=false
NO_GUI=""
MAX_STEPS=2000

for arg in "$@"; do
    case $arg in
        --scene=*)  SCENE="${arg#*=}" ;;
        --goal=*)   GOAL="${arg#*=}" ;;
        --with-server) WITH_SERVER=true ;;
        --no-gui)   NO_GUI="--no-gui" ;;
        --max-steps=*) MAX_STEPS="${arg#*=}" ;;
    esac
done

echo "============================================"
echo "  LEGS Navigation Demo"
echo "  Scene:  $SCENE"
echo "  Goal:   $GOAL"
echo "  Server: $WITH_SERVER"
echo "============================================"

if [ "$WITH_SERVER" = true ]; then
    echo "[1/2] Starting gRPC Server (stub policy)..."
    python3 -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/packages/legs_server/src')
sys.path.insert(0, '${PROJECT_ROOT}/packages/legs_common/src')
from legs_server.model.nwm_infer import StubNWMPolicy
from legs_server.service.grpc_server import serve
policy = StubNWMPolicy(model_tag='demo-stub')
server, port = serve(policy=policy, bind_addr='0.0.0.0:50051')
print(f'Server running on port {port}')
server.wait_for_termination()
" &
    SERVER_PID=$!
    sleep 2
    echo "[2/2] Starting Dog (PyBullet)..."
    python3 -m legs_dog.main \
        --mode=sim \
        --sim-backend=pybullet \
        --scene="$SCENE" \
        --goal="$GOAL" \
        --server=localhost:50051 \
        --max-steps="$MAX_STEPS" \
        $NO_GUI

    kill $SERVER_PID 2>/dev/null || true
else
    echo "Starting Dog (PyBullet, offline MCTS mode)..."
    python3 -m legs_dog.main \
        --mode=sim \
        --sim-backend=pybullet \
        --scene="$SCENE" \
        --goal="$GOAL" \
        --offline \
        --use-mcts \
        --max-steps="$MAX_STEPS" \
        $NO_GUI
fi

echo "Done."
