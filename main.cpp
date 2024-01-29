#include <iostream>
#include <torch/torch.h>
#include "lib/vit_pytorch_cpp/vit.h"

#define BATCH_SIZE 64
#define EPOCHS 20
#define LEARNING_RATE 3e-5
#define GAMMA 0.7

int main(void) {

    char* pool = "cls";
    ViT v = ViT(
        std::vector<int>({256, 256}),       // image_size
        std::vector<int>({32, 32}),         // patch_size
        1000,                               // num_classes
        1024,                               // dim
        6,                                  // depth
        16,                                 // heads
        2048,                               // mlp_dim
        pool,                               // pool
        3,                                  // channels
        64,                                 // dim_head       
        0.1,                                // dropout
        0.1                                 // emb_dropout
    );
    
    torch::Tensor img = torch::randn({1, 3, 256, 256});
    torch::Tensor predictions = v(img);


}
