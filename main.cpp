#include <iostream>
#include <torch/torch.h>
#include "lib/vit_pytorch_cpp/vit.h"
#include "lib/loaders/dataloaders.h"
// #include <torch/nn/parallel/data_parallel.h>

#define BATCH_SIZE 512
#define EPOCHS 100
#define LEARNING_RATE 1e-3
#define GAMMA 0.7

torch::Device device(torch::kCUDA, 0);
torch::Device device1(torch::kCUDA, 1);
torch::Device device2(torch::kCUDA, 2);
torch::Device device3(torch::kCUDA, 3);
torch::Device device4(torch::kCUDA, 4);
torch::Device device5(torch::kCUDA, 5);
torch::Device device6(torch::kCUDA, 6);
torch::Device device7(torch::kCUDA, 7);
std::vector<torch::Device> all_devices {
    device,
    device1,
    device2,
    device3,
    device4,
    device5,
    device6,
    device7
};

int main(void) {

    std::cout << "PyTorch version: "
    << TORCH_VERSION_MAJOR << "."
    << TORCH_VERSION_MINOR << "."
    << TORCH_VERSION_PATCH << std::endl;

    std::cout << "Initializing dataset..." << std::endl;
    CIFAR102Dataset train = CIFAR102Dataset(true);
    CIFAR102Dataset test = CIFAR102Dataset(false);
    std::cout << "Dataset initialized." << std::endl;

    std::cout << "Creating dataloader..." << std::endl;
    auto train_loader = torch::data::make_data_loader(
        train, 
        torch::data::DataLoaderOptions().batch_size(BATCH_SIZE)
    );
    auto test_loader = torch::data::make_data_loader(
        test,
        torch::data::DataLoaderOptions().batch_size(1)
    );
    std::cout << "Dataloader initialized." << std::endl;

    std::cout << "Setting up the ViT model..." << std::endl;
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
    
    std::cout << "ViT successfully created." << std::endl;
    
    std::cout << "Creating tools for learning..." << std::endl;
    // loss function, optimizer, scheduler
    auto criterion = torch::nn::CrossEntropyLoss();
    auto optimizer = torch::optim::AdamW(
        model->parameters(),
        torch::optim::AdamWOptions(LEARNING_RATE)
    );
    auto scheduler = torch::optim::StepLR(optimizer, 1, GAMMA);

    std::cout << "Learning environment setup complete. ";
    std::cout << "Beginning learning..." << std::endl;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double epoch_loss = 0.0;
        double epoch_accuracy = 0.0;

        int batch_num = 0;
        for (std::vector<CIFARItem>& batch : *train_loader) {
            // Reshape data from batch into single tensor
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
            torch::Tensor data = torch::stack(data_ref).to(device);
            torch::Tensor label = torch::stack(label_ref).to(device);

            // Forward the data & train the model
            model->zero_grad();
            // torch::Tensor output = torch::nn::parallel::data_parallel(
            //     model, 
            //     data
            // );
            // std::cout << output.sizes() << std::endl;
            torch::Tensor output = model->forward(data);
            torch::Tensor loss = criterion(output, label);
            loss.backward();
            optimizer.step();
            if(batch_num++ % 10 == 0)
            std::printf(
                "| [Epoch %2d/%2d] | [Batch %3d] | loss: %.4f |\n",
                epoch + 1,
                EPOCHS,
                batch_num,
                loss.item<float>()
            );
        }
    }
}
