#include <iostream>
#include <torch/torch.h>
#include "lib/vit_pytorch_cpp/vit.h"
#include "lib/loaders/dataloaders.h"
#include "lib/tqdm/tqdm.h"

#define BATCH_SIZE 64
#define EPOCHS 20
#define LEARNING_RATE 3e-5
#define GAMMA 0.7

int main(void) {

    torch::Device device(torch::kCPU);

    CIFAR102Dataset train = CIFAR102Dataset(true);
    CIFAR102Dataset test = CIFAR102Dataset(false);

    auto train_loader = torch::data::make_data_loader(
        train, 
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
    );

    auto test_loader = torch::data::make_data_loader(
        test,
        torch::data::DataLoaderOptions().batch_size(1)
    );

    char* pool = "cls";
    ViT model = ViT(
        std::vector<int>({32, 32}),         // image_size
        std::vector<int>({4, 4}),           // patch_size
        10,                                 // num_classes
        128,                                // dim
        6,                                  // depth
        16,                                 // heads
        512,                                // mlp_dim
        pool,                               // pool
        3,                                  // channels
        64,                                 // dim_head       
        0.1,                                // dropout
        0.1                                 // emb_dropout
    );
    model->to(device);
    
    // loss function, optimizer, scheduler
    auto criterion = torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::Adam(
        model->parameters(),
        torch::optim::AdamOptions(LEARNING_RATE)
    );
    auto scheduler = torch::optim::StepLR(optimizer, 1, GAMMA);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;

        std::cout << "RUNNING EPOCH " << epoch << "..." << std::endl;
        int batch_num = 0;
        for (std::vector<CIFARItem>& batch : *train_loader) {
            std::cout << "RUNNING BATCH_NUM  " << batch_num++ << "..." << std::endl;
            optimizer.zero_grad();
            std::vector<torch::Tensor> data_array = 
                std::vector<torch::Tensor>();
            std::vector<torch::Tensor> label_array = 
                std::vector<torch::Tensor>();
            for (int i = 0; i < batch.size(); i++) {
                data_array.push_back(batch.at(i).data);
                label_array.push_back(batch.at(i).target);
            }
            c10::ArrayRef<torch::Tensor> data_ref = 
                c10::ArrayRef<torch::Tensor>(
                    &data_array[0], 
                    batch.size()
                );
            c10::ArrayRef<torch::Tensor> label_ref = 
                c10::ArrayRef<torch::Tensor>(
                    &label_array[0], 
                    batch.size()
                );
            torch::Tensor data = torch::stack(data_ref);
            torch::Tensor label = torch::stack(label_ref);
            std::cout << data.sizes() << std::endl;
            std::cout << label.sizes() << std::endl;
            torch::Tensor output = model->forward(data);
            torch::Tensor loss = criterion(output, label);
            loss.backward();
            optimizer.step();
        }

    }

}
