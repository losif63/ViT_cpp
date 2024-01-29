#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include "dataloaders.h"

CIFAR102Dataset::CIFAR102Dataset(bool train) : train(train) {
    const char* filenames[6] ={
        "../data/raw/CIFAR10/cifar-10-batches-bin/data_batch_1.bin",
        "../data/raw/CIFAR10/cifar-10-batches-bin/data_batch_2.bin",
        "../data/raw/CIFAR10/cifar-10-batches-bin/data_batch_3.bin",
        "../data/raw/CIFAR10/cifar-10-batches-bin/data_batch_4.bin",
        "../data/raw/CIFAR10/cifar-10-batches-bin/data_batch_5.bin",
        "../data/raw/CIFAR10/cifar-10-batches-bin/test_batch.bin"
    };
    uint8_t row_label;
    uint8_t* row_data = (uint8_t*)malloc(3072);

    if(train) {
        for(int i = 0; i < 5; i++) {
            std::ifstream ifs;
            ifs.open(filenames[i], std::ifstream::in);
            for(int j = 0; j < 10000; j++) {
                ifs.read((char *)&row_label, 1);
                labels.push_back(row_label);
                ifs.read((char *)row_data, 3072);
                torch::Tensor new_data = torch::from_blob(
                    row_data, 
                    3072, 
                    c10::TensorOptions(c10::ScalarType::Byte)
                ).view({3, 32, 32});
                data.push_back(new_data);
            }
            ifs.close();
        }
    } else {
        std::ifstream ifs;
        ifs.open(filenames[5], std::ifstream::in);  
        for(int j = 0; j < 10000; j++) {
            ifs.read((char *)&row_label, 1);
            labels.push_back(row_label);
            ifs.read((char *)row_data, 3072);
            torch::Tensor new_data = torch::from_blob(
                row_data, 
                3072, 
                c10::TensorOptions(c10::ScalarType::Byte)
            ).view({3, 32, 32});
            data.push_back(new_data);
        }
        ifs.close();
    }
    
    free(row_data);
}

CIFARItem CIFAR102Dataset::get(size_t index) {
    return CIFARItem(
        this->data.at(index),
        this->labels.at(index)
    );
}

std::optional<size_t> CIFAR102Dataset::size() const {
    return this->data.size();
}