#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <torch/torch.h>

typedef torch::data::Example<torch::Tensor, torch::Tensor> CIFARItem;

class CIFAR102Dataset : public torch::data::datasets::Dataset<CIFAR102Dataset, CIFARItem> {
public:
    CIFAR102Dataset(bool train);
    ~CIFAR102Dataset() override;
    CIFARItem get(size_t index) override;
    bool get_train();
    int get_num_classes();
    std::vector<torch::Tensor> get_data();
    std::vector<torch::Tensor> get_labels();
    c10::optional<size_t> size() const override;

private:
    bool train;
    const int num_classes = 10;
    std::vector<torch::Tensor> data;
    std::vector<torch::Tensor> labels;
};

#endif