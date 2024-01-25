#include <torch/torch.h>
#include "Classes.h"

struct FeedForwardImpl : torch::nn::Module {
    FeedForwardImpl(int dim, int hidden_dim, float dropout = 0.0)
        : net(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})),
            torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim)),
            torch::nn::GELU(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout)),
            torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim)),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout))
        )
    {
        register_module("net", net);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = net->forward(x);
        return x;
    }

    torch::nn::Sequential net;
};
TORCH_MODULE(FeedForward);

