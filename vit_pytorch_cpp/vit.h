#ifndef VIT_H
#define VIT_H

#include <torch/torch.h>

class TORCH_API FeedForwardImpl : public torch::nn::Module {
public:
    FeedForwardImpl(int dim, int hidden_dim, float dropout = 0.0);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential net;
};
TORCH_MODULE(FeedForward);

class TORCH_API AttentionImpl : public torch::nn::Module {
public:
    AttentionImpl(int dim, int heads = 8, int dim_head = 8, float dropout = 0.0);
    torch::Tensor forward(torch::Tensor x);

private:
    int heads;
    float scale;
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Softmax attend{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear to_qkv{nullptr};
    torch::nn::Sequential to_out{nullptr};
};
TORCH_MODULE(Attention);

class TORCH_API TransformerImpl : public torch::nn::Module {
public:
    TransformerImpl(int dim, int depth, int heads, int dim_head, int mlp_dim, float dropout = 0.0);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::ModuleList layers{nullptr};
};
TORCH_MODULE(Transformer);

class TORCH_API RearrangeImpl : public torch::nn::Module {
public:
    RearrangeImpl(int p1, int p2);
    torch::Tensor forward(torch::Tensor x);

private:
    int p1, p2;
};
TORCH_MODULE(Rearrange);

class TORCH_API ViTImpl : public torch::nn::Module {
public:
    ViTImpl(
        std::vector<int> image_size, 
        std::vector<int> patch_size, 
        int num_classes, 
        int dim, 
        int depth, 
        int heads, 
        int mlp_dim, 
        char* pool = (char*)"cls",
        int channels = 3, 
        int dim_head = 64, 
        float dropout = 0.0, 
        float emb_dropout = 0.0
    );
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential to_patch_embedding{nullptr};
    torch::Tensor pos_embedding, cls_token;
    torch::nn::Dropout dropout{nullptr};
    Transformer transformer{nullptr};
    char* pool;
    torch::nn::Identity to_latent{nullptr};
    torch::nn::Linear mlp_head{nullptr};
};
TORCH_MODULE(ViT);

#endif