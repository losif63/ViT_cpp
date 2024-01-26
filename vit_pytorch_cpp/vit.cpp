#include <cmath>
#include <torch/torch.h>
#include <assert.h>
#include <string.h>
#include "vit.h"


FeedForwardImpl::FeedForwardImpl(int dim, int hidden_dim, float dropout)
{
    this->net = register_module("net", torch::nn::Sequential(
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})),
        torch::nn::Linear(dim, hidden_dim),
        torch::nn::GELU(),
        torch::nn::Dropout(dropout),
        torch::nn::Linear(hidden_dim, dim),
        torch::nn::Dropout(dropout)
    ));
}

torch::Tensor FeedForwardImpl::forward(torch::Tensor x) {
    x = this->net->forward(x);
    return x;
}

AttentionImpl::AttentionImpl(int dim, int heads, int dim_head, float dropout)
{
    int inner_dim = dim_head * heads;
    bool project_out = !((heads == 1) && (dim_head == dim));

    this->heads = heads;
    this->scale = pow((float)dim_head, -0.5);

    this->norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    this->attend = register_module("attend", torch::nn::Softmax(-1));
    this->dropout = register_module("dropout", torch::nn::Dropout(dropout));
    this->to_qkv = register_module("to_qkv", torch::nn::Linear(torch::nn::LinearOptions(dim, inner_dim * 3).bias(false)));
    if (project_out) {
        this->to_out = register_module("to_out", torch::nn::Sequential(
            torch::nn::Linear(inner_dim, dim),
            torch::nn::Dropout(dropout)
        ));
    } else {
        this->to_out = register_module("to_out", torch::nn::Sequential(torch::nn::Identity()));
    }
}

torch::Tensor AttentionImpl::forward(torch::Tensor x) {
    x = this->norm(x);
    auto qkv = this->to_qkv(x).chunk(3, -1);
    torch::Tensor q = qkv[0], k = qkv[1], v = qkv[2];
    q = q.view({q.size(0), heads, q.size(1), -1});
    k = k.view({k.size(0), heads, k.size(1), -1});
    v = v.view({v.size(0), heads, v.size(1), -1});
    
    torch::Tensor dots = torch::matmul(q, k.transpose(-1, -2)) * this->scale;
    torch::Tensor attn = this->attend(dots);
    attn = this->dropout(attn);
    torch::Tensor out = torch::matmul(attn, v);
    out = out.view({out.size(0), out.size(2), -1});
    return this->to_out->forward(out);
}

TransformerImpl::TransformerImpl(int dim, int depth, int heads, int dim_head, int mlp_dim, float dropout)
{
    this->norm = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    this->layers = register_module("layers", torch::nn::ModuleList());
    for (int i = 0; i < depth; i++) {
        torch::nn::ModuleList layer = torch::nn::ModuleList();
        layer->push_back(Attention(dim, heads, dim_head, dropout));
        layer->push_back(FeedForward(dim, mlp_dim, dropout));
        this->layers->push_back(layer);
    }
}

torch::Tensor TransformerImpl::forward(torch::Tensor x) {
    this->layers->size();
    for (size_t i = 0; i < this->layers->size(); i++) {
        torch::nn::ModuleListImpl elem = this->layers->at<torch::nn::ModuleListImpl>(i);
        x = elem.at<AttentionImpl>(0).forward(x) + x;
        x = elem.at<FeedForwardImpl>(1).forward(x) + x;
    }
    return this->norm(x);
}

RearrangeImpl::RearrangeImpl(int p1, int p2) {
    this->p1 = p1;
    this->p2 = p2;
}

torch::Tensor RearrangeImpl::forward(torch::Tensor x) {
    assert(x.size(2) % p1 == 0);
    assert(x.size(3) % p2 == 0);

    int b = x.size(0);
    int c = x.size(1);
    int h = x.size(2) / p1;
    int w = x.size(3) / p2;
    x = x.view({b, c, h, p1, w, p2});
    x = x.permute({0, 2, 4, 3, 5, 1}).reshape({b, h * w, p1 * p2 * c});
    return x;
}

ViTImpl::ViTImpl(
    std::vector<int> image_size, 
    std::vector<int> patch_size, 
    int num_classes, 
    int dim, 
    int depth, 
    int heads, 
    int mlp_dim, 
    char* pool,
    int channels, 
    int dim_head, 
    float dropout, 
    float emb_dropout
)
{
    int image_height = image_size.at(0);
    int image_width = image_size.at(1);
    int patch_height = patch_size.at(0);
    int patch_width = patch_size.at(1);

    assert((image_height % patch_height == 0) && (image_width % patch_width == 0));
    int num_patches = (image_height / patch_height) * (image_width / patch_width);
    int patch_dim = channels * patch_height * patch_width;
    
    assert((strcmp(pool, "cls") == 0) || (strcmp(pool, "mean") == 0));
    this->to_patch_embedding = register_module("to_patch_embedding", torch::nn::Sequential(
        Rearrange(patch_height, patch_width),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({patch_dim})),
        torch::nn::Linear(patch_dim, dim),
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim}))
    ));

    this->pos_embedding = register_parameter("pos_embedding", torch::randn({1, num_patches + 1, dim}));
    this->cls_token = register_parameter("cls_token", torch::randn({1, 1, dim}));
    this->dropout = register_module("dropout", torch::nn::Dropout(emb_dropout));
    this->transformer = register_module("transformer", Transformer(
        dim=dim, 
        depth=depth, 
        heads=heads, 
        dim_head=dim_head, 
        mlp_dim=mlp_dim, 
        dropout = dropout
    ));

    this->pool = pool;
    this->to_latent = register_module("to_latent", torch::nn::Identity());
    this->mlp_head = register_module("mlp_head", torch::nn::Linear(dim, num_classes));
}

torch::Tensor ViTImpl::forward(torch::Tensor x) {
    x = this->to_patch_embedding->forward(x);
    int b = x.size(0);
    int n = x.size(1);
    
    torch::Tensor cls_tokens = this->cls_token.repeat({b, 1, 1});
    std::cout << "REACHED HERE 1" << std::endl;
    x = torch::cat({cls_tokens, x}, 1);
    std::cout << "REACHED HERE 2" << std::endl;
    x += this->pos_embedding.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, (n + 1))});
    std::cout << "REACHED HERE 3" << std::endl;
    x = this->dropout(x);
    std::cout << "REACHED HERE 4" << std::endl;
    x = this->transformer(x);
    std::cout << "REACHED HERE 5" << std::endl;
    if (strcmp(this->pool, "mean") == 0) {
        x = x.mean(1);
    } else {
        x = x.index({torch::indexing::Slice(), 0});
    }
    std::cout << "REACHED HERE 6" << std::endl;
    x = this->to_latent(x);
    std::cout << "REACHED HERE 7" << std::endl;
    return this->mlp_head(x);
}