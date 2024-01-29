# CHANGE /path/to/torch to the pytorch | libtorch installation path
export SYS_NAME=$(uname -s)
if [[ "$SYS_NAME" == "Linux" ]]; then
    export TORCH_PATH=/path/to/torch
elif [[ "$SYS_NAME" == "Darwin" ]]; then
    export TORCH_PATH=/path/to/torch
fi

mkdir data
cd data

mkdir CIFAR10 && mkdir CIFAR100
cd CIFAR10 && wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xvzf cifar-10-binary.tar.gz

cd ../CIFAR100 && wget https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar -xvzf cifar-100-binary.tar.gz

cd ../..
