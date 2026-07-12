#!/bin/bash
cd /home/zhangxa/codes/edgeLLM/Chamleon
source scripts/tvm_env.sh

# 1) nsys 采 bench_kv（约编译+跑一会儿）
bash scripts/profile_pi05_trt_tvm.sh --run nsys

# 2) 看 Top kernel / CUDA API
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum \
  output/pi05_libero_profile/nsys/tvm_bench_kv_fp16.nsys-rep | head -50

# 3)（可选）对照开 Graph 的捕获情况
OUT=output/pi05_libero_profile
nsys profile -t cuda,nvtx,osrt -s none --force-overwrite=true \
  -o "$OUT/nsys/tvm_bench_kv_fp16_cg" \
  -- "$MLC_VLA_PY" -m mlc_vla.bench_kv \
       --target cuda --dtype float16 --steps 10 --iters 20 --cuda-graph
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum "$OUT/nsys/tvm_bench_kv_fp16_cg.nsys-rep" | head -40