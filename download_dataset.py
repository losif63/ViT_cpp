from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# cifar_10_training_data = datasets.CIFAR10(
#     root="data/CIFAR10/train",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# cifar_10_test_data = datasets.CIFAR10(
#     root="data/CIFAR10/test",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

# cifar_100_training_data = datasets.CIFAR100(
#     root="data/CIFAR100/train",
#     train=True,
#     download=True,
#     transform=ToTensor()
# )

# cifar_100_test_data = datasets.CIFAR100(
#     root="data/CIFAR100/test",
#     train=False,
#     download=True,
#     transform=ToTensor()
# )

imagenet_training_data = datasets.ImageNet(
    root='data/ImageNet/test',
    train=True,
    download=True,
    transform=ToTensor()
)

imagenet_test_data = datasets.ImageNet(
    root='data/ImageNet/test',
    train=False,
    download=True,
    transform=ToTensor()
)