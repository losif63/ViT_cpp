# CHANGE /path/to/torch to the pytorch | libtorch installation path
export SYS_NAME=$(uname -s)
if [[ "$SYS_NAME" == "Linux" ]]; then
    export TORCH_PATH=/path/to/torch
elif [[ "$SYS_NAME" == "Darwin" ]]; then
    export TORCH_PATH=/path/to/torch
fi
