export SYS_NAME=$(uname -s)
if [[ "$SYS_NAME" == "Linux" ]]; then
    # export TORCH_PATH=/home/sslunder7/anaconda3/envs/fbgemm/lib/python3.8/site-packages/torch/include/torch/
    export TORCH_PATH=/home/sslunder7/libtorch
    export PATH_TO_CUDA=/home/sslunder7/anaconda3/envs/fbgemm/include/
elif [[ "$SYS_NAME" == "Darwin" ]]; then
    export TORCH_PATH=/Users/jaduksuh/anaconda3/envs/fbgemm/lib/python3.8/site-packages/torch/
fi
