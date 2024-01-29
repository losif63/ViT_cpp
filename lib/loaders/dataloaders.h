#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <torch/torch.h>

typedef torch::data::Example<torch::Tensor, uint8_t> CIFARItem;

class CIFAR102Dataset : public torch::data::datasets::Dataset<CIFAR102Dataset, CIFARItem> {
public:
    CIFAR102Dataset(bool train);
    CIFARItem get(size_t index) override;
    std::optional<size_t> size() const override;

private:
    bool train;
    std::vector<torch::Tensor> data;
    std::vector<uint8_t> labels;
};

#endif