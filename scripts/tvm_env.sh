#!/bin/bash
# TVM / mlc-vla 环境（本机开发机示例；Thor 上请改成板端路径）。
#
# 新版 TVM 依赖独立包 tvm_ffi。确认存在其一：
#   $TVM_HOME/python/tvm_ffi
#   $TVM_HOME/3rdparty/tvm-ffi/python/tvm_ffi
# 若缺失，在 MLC_VLA_PY 对应解释器上安装（注意是 tvm-ffi 仓库根，不是 python/ 子目录）：
#   $MLC_VLA_PY -m pip install -e "$TVM_HOME/3rdparty/tvm-ffi"

export TVM_HOME=${TVM_HOME:-/home/zhangxa/codes/edgeLLM/tvm}
export MLC_VLA_HOME=${MLC_VLA_HOME:-/home/zhangxa/codes/edgeLLM/mlc-vla}
export MLC_VLA_PY=${MLC_VLA_PY:-/usr/bin/python3.12}
export TVM_LIBRARY_PATH=${TVM_LIBRARY_PATH:-$TVM_HOME/build/lib}
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:$MLC_VLA_HOME/python:${PYTHONPATH:-}

# 快速自检（可选）：
#   $MLC_VLA_PY -c "import tvm_ffi, tvm, mlc_vla; print('ok', tvm.__file__)"
