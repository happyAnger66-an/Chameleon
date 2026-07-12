export TVM_HOME=/srcs/tvm/
export TVM_LIBRARY_PATH=$TVM_HOME/build/lib
export MLC_VLA_HOME=/srcs/codes/mlc-vla/
export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/3rdparty/tvm-ffi/python:${PYTHONPATH:-}

# 快速自检（可选）：
#   $MLC_VLA_PY -c "import tvm_ffi, tvm, mlc_vla; print('ok', tvm.__file__)"