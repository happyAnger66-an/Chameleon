#!/usr/bin/env bash
# pi05 TRT vs TVM 深层 profiling 命令集。
# 文档：docs/optimizer/pi05/trt_tvm_profile.md
#
#   bash scripts/profile_pi05_trt_tvm.sh              # 只打印
#   bash scripts/profile_pi05_trt_tvm.sh --run        # 全部执行
#   bash scripts/profile_pi05_trt_tvm.sh --run kv|trt|nsys|bench
#
# Jetson Thor（先 source scripts/tvm_thor.sh && export MLC_VLA_PY=<thor python3.12>）：
#   bash scripts/profile_pi05_trt_tvm.sh --run thor          # build 引擎 + bench
#   bash scripts/profile_pi05_trt_tvm.sh --run thor-deploy   # 仅 build 引擎
#   bash scripts/profile_pi05_trt_tvm.sh --run thor-bench    # 仅 bench（引擎已 build）

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${ROOT}/scripts/tvm_env.sh"

CHAM_PY="${CHAM_PY:-${ROOT}/models/openpi/.venv/bin/python}"
OUT="${OUT:-${ROOT}/output/pi05_libero_profile}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

RUN=0
MODE=all
if [[ "${1:-}" == "--run" ]]; then
  RUN=1
  MODE="${2:-all}"
fi

mkdir -p "$OUT"/{nsys,bench_kv,trt}

echo "== env =="
echo "  ROOT=$ROOT"
echo "  CHAM_PY=$CHAM_PY"
echo "  MLC_VLA_PY=$MLC_VLA_PY"
echo "  TVM_HOME=$TVM_HOME"
echo "  OUT=$OUT"
echo "  MODE=$MODE run=$RUN"
echo

_cmd() {
  echo "+ $*"
  if [[ "$RUN" -eq 1 ]]; then
    eval "$@"
  fi
}

run_kv() {
  echo "== 1a) mlc_vla.bench_kv (fp16, --no-cublas 基线) =="
  _cmd "\"$MLC_VLA_PY\" -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 --no-cublas | tee \"$OUT/bench_kv/fp16_steps_dlight.txt\""
  echo
  echo "== 1b) mlc_vla.bench_kv (fp16, 默认=cuBLAS, Phase B) =="
  _cmd "\"$MLC_VLA_PY\" -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 | tee \"$OUT/bench_kv/fp16_steps.txt\""
  echo
  echo "== 1c) mlc_vla.bench_kv (fp16, cuBLAS + --cuda-graph) =="
  _cmd "\"$MLC_VLA_PY\" -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 50 --cuda-graph | tee \"$OUT/bench_kv/fp16_steps_cg.txt\""
}

run_bench() {
  echo "== 2a) chameleon bench steps (逐步 denoise, tvm_loop/graph=off) =="
  _cmd "\"$CHAM_PY\" -m chameleon.cli bench --config configs/pi05/pi05_libero_bench_steps.yaml -v | tee \"$OUT/bench_kv/cham_steps.txt\""
  echo
  echo "== 2b) chameleon bench e2e (tvm_loop+CUDA Graph on) =="
  _cmd "\"$CHAM_PY\" -m chameleon.cli bench --config configs/pi05/pi05_libero_bench.yaml -v | tee \"$OUT/bench_kv/cham_e2e.txt\""
}

run_trt() {
  echo "== 3) chameleon trt-profile (llm + denoise) =="
  _cmd "\"$CHAM_PY\" -m chameleon.cli trt-profile --config configs/pi05/pi05_libero_trt_profile.yaml -v | tee \"$OUT/trt/trt_profile.log\""
}

# Jetson Thor（sm_101）：引擎设备相关，须在 Thor 本机 build 后再 bench。
# 前置：source scripts/tvm_thor.sh && export MLC_VLA_PY=<thor python3.12>
run_thor_deploy() {
  echo "== T1) Thor: build TRT engines (export+compile) =="
  _cmd "\"$CHAM_PY\" -m chameleon.cli workflow --config configs/pi05/pi05_libero_trt_deploy_thor.yaml"
}

run_thor_bench() {
  echo "== T2) Thor: chameleon bench TRT vs TVM (loop+CUDA Graph) =="
  _cmd "\"$CHAM_PY\" -m chameleon.cli bench --config configs/pi05/pi05_libero_bench_thor.yaml -v | tee \"$OUT/bench_kv/cham_thor.txt\""
}

run_nsys() {
  if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys not found; skip"
    return 0
  fi
  echo "== 4) nsys + bench_kv =="
  _cmd "nsys profile -t cuda,nvtx,osrt -s none --force-overwrite=true -o \"$OUT/nsys/tvm_bench_kv_fp16\" -- \"$MLC_VLA_PY\" -m mlc_vla.bench_kv --target cuda --dtype float16 --steps 10 --iters 20"
  if [[ "$RUN" -eq 1 ]]; then
    _cmd "nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \"$OUT/nsys/tvm_bench_kv_fp16.nsys-rep\" | tee \"$OUT/nsys/tvm_bench_kv_fp16_stats.txt\""
  else
    echo "+ nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum $OUT/nsys/tvm_bench_kv_fp16.nsys-rep"
  fi
}

case "$MODE" in
  all)
    run_kv
    echo
    run_bench
    echo
    run_trt
    echo
    run_nsys
    ;;
  kv) run_kv ;;
  bench) run_bench ;;
  trt) run_trt ;;
  nsys) run_nsys ;;
  thor-deploy) run_thor_deploy ;;
  thor-bench) run_thor_bench ;;
  thor)
    run_thor_deploy
    echo
    run_thor_bench
    ;;
  *)
    echo "unknown mode: $MODE (all|kv|bench|trt|nsys|thor|thor-deploy|thor-bench)" >&2
    exit 2
    ;;
esac

echo
if [[ "$RUN" -eq 0 ]]; then
  echo "Dry-run only. Re-run with: $0 --run [all|kv|bench|trt|nsys|thor|thor-deploy|thor-bench]"
else
  echo "Done. See $OUT and docs/optimizer/pi05/trt_tvm_profile.md"
fi
