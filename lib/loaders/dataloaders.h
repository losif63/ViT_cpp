#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <torch/torch.h>

class CIFAR102Dataset : public torch::data::datasets::Dataset<CIFAR102Dataset, torch::Tensor> {
public:
    CIFAR102Dataset();
    torch::Tensor get(size_t index) override;
    std::optional<size_t> size() const override;

private:
};

#endif