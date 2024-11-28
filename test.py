from train import evaluate_model
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.densenet import densenet161

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128

if __name__ == '__main__':
    # 参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    ])

    model = resnet50(pretrained=True).to(device)

    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    evaluate_model(model, test_loader)
