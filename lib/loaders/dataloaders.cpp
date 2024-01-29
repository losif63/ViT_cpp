#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include "dataloaders.h"

CIFAR102Dataset::CIFAR102Dataset() {
    const char* filenames[6] ={
        "data/raw/CIFAR10/cifar-10-batches-bin/data_batch_1.bin",
        "data/raw/CIFAR10/cifar-10-batches-bin/data_batch_2.bin",
        "data/raw/CIFAR10/cifar-10-batches-bin/data_batch_3.bin",
        "data/raw/CIFAR10/cifar-10-batches-bin/data_batch_4.bin",
        "data/raw/CIFAR10/cifar-10-batches-bin/data_batch_5.bin",
        "data/raw/CIFAR10/cifar-10-batches-bin/test_batch.bin"
    };
    for (int i = 0; i < 6; i++) {
        std::ifstream file{filenames[i], std::ios::binary};
        char byte;
        file.read(&byte, 1);
        std::cout << (int)byte << std::endl;
    }
    
}

torch::Tensor CIFAR102Dataset::get(size_t index) {
    return torch::randn({1, 3});
}

std::optional<size_t> CIFAR102Dataset::size() const {
    return 0;
}
