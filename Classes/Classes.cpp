#include <cmath>
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

struct AttentionImpl : torch::nn::Module {
    
    AttentionImpl(int dim, int heads = 8, int dim_head = 8, float dropout = 0.0) 
        : norm(torch::nn::LayerNormOptions({dim})),
        attend(torch::nn::SoftmaxOptions(-1)),
        dropout(torch::nn::DropoutOptions(dropout)),
        to_qkv(torch::nn::LinearOptions(dim, dim_head * heads * 3).bias(false)),
        to_out(
            // TODO
        )
    {
        heads = heads;
        scale = pow((float)dim_head, -0.5);
    }
    int heads;
    float scale;
    torch::nn::LayerNorm norm;
    torch::nn::Softmax attend;
    torch::nn::Dropout dropout;
    torch::nn::Linear to_qkv;
    torch::nn::Sequential to_out;
};
TORCH_MODULE(Attention);