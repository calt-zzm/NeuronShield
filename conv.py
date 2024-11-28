import torch
import torchattacks
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import vgg11, VGG11_Weights
import torch.nn.functional as F

import neuron
import utils


def count_anomaly_neurons_per_channel(layers_output, adv_layers_output, threshold=0.1, mask_ratio=0.2):
    """统计每个卷积层中每个通道的异常神经元数"""
    different_neurons_per_channel = {}  # 存储每层每个通道的异常神经元数

    # 调用 get_anomaly_neurons 函数，获取异常神经元和掩码
    different_neurons, _ = utils.get_anomaly_neurons(layers_output, adv_layers_output, threshold, mask_ratio)

    for key in layers_output.keys():
        if isinstance(layers_output[key], torch.Tensor):
            output_shape = layers_output[key].shape

            # 只统计卷积层的输出 (通常是 4D Tensor: [batch_size, channels, height, width])
            if len(output_shape) == 4:  # 形状 [batch_size, channels, height, width]
                batch_size, channels, height, width = output_shape
                diff_mean = torch.mean(torch.abs(layers_output[key] - adv_layers_output[key]), dim=0)  # 按 batch 求均值

                # 统计每个通道的异常神经元数
                channel_anomaly_neurons = []
                for c in range(channels):
                    # 获取当前通道的异常神经元索引
                    anomaly_neurons = torch.nonzero(diff_mean[c] > threshold).squeeze()
                    num_anomaly_neurons = len(anomaly_neurons)

                    # 保存每个通道的异常神经元数
                    channel_anomaly_neurons.append(num_anomaly_neurons)

                # 保存每层的通道异常神经元统计
                different_neurons_per_channel[key] = channel_anomaly_neurons

    return different_neurons_per_channel


def mask_top_10_percent_channels_by_weights(model, anomalies_per_channel):
    """通过修改权重来屏蔽每个卷积层中异常神经元数最多的前 10% 通道"""
    with torch.no_grad():  # 避免自动求导
        layer_dict = utils.get_model_layers(model)
        for name, layer in layer_dict.items():
            if "Conv2d" in name:  # 只对卷积层进行处理
                if name in anomalies_per_channel:
                    # 获取当前卷积层的权重
                    weight = layer.weight  # 形状为 [out_channels, in_channels, kernel_height, kernel_width]
                    out_channels = weight.shape[0]

                    # 获取该层每个通道的异常神经元数
                    channel_anomalies = anomalies_per_channel[name]

                    # 计算要屏蔽的通道数 (前 10%)
                    num_channels_to_mask = max(1, int(out_channels * 0.1))

                    # 找到异常神经元数最多的通道索引
                    sorted_channels = sorted(range(out_channels), key=lambda c: channel_anomalies[c], reverse=True)
                    channels_to_mask = sorted_channels[:num_channels_to_mask]

                    # 将这些通道的权重设为 0
                    for channel in channels_to_mask:
                        weight[channel, :, :, :] = 0  # 屏蔽该通道的所有卷积核

                    print(f"Masked {num_channels_to_mask} channels in layer {name}: {channels_to_mask}")


def evaluate_model(model, data_loader, device):
    """计算模型的预测准确率"""
    origin_model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1).to(device).eval()
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    atk = torchattacks.FGSM(origin_model, eps=0.1)
    for img_tensors, label_tensors in tqdm(data_loader):
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)
        adv_img_tensors = atk(img_tensors, label_tensors).to(device)
        # 前向传播
        outputs = model(adv_img_tensors)
        _, predicted = torch.max(outputs, 1)

        # 计算正确的预测数
        total += label_tensors.size(0)
        correct += (predicted == label_tensors).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy of the model: {accuracy:.2f}%')
    return accuracy


# 示例使用
def main():
    # 假设你已经有模型和数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1).to(device).eval()  # 加载模型
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256]),
        transforms.CenterCrop([224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集和数据加载器
    val_images_path = "D:/Program/safetyTest/ImageNet/images"
    val_labels_path = "D:/Program/safetyTest/ImageNet/ImageNet_val_labels.txt"
    dataset = neuron.ImageDataset(val_images_path, val_labels_path, transform=transform)
    data_loader = DataLoader(dataset, num_workers=4, batch_size=32, shuffle=False)

    atk = torchattacks.FGSM(model, eps=0.1)

    for img_tensors, label_tensors in tqdm(data_loader):
        img_tensors = img_tensors.to(device)
        adv_img_tensors = atk(img_tensors, label_tensors).to(device)

        # 获取原始样本和对抗样本的层输出
        layers_output = utils.get_layer_output(model, img_tensors)
        adv_layers_output = utils.get_layer_output(model, adv_img_tensors)

        # 统计每个卷积层中每个通道的异常神经元数
        anomalies_per_channel = count_anomaly_neurons_per_channel(layers_output, adv_layers_output, threshold=0.2,
                                                                  mask_ratio=0.1)
    # 通过修改权重来屏蔽每个卷积层中异常神经元数最多的前 10% 通道
    mask_top_10_percent_channels_by_weights(model, anomalies_per_channel)

    # 保存修改后的权重
    torch.save(model.state_dict(), "masked_model_weights.pth")
    print("Modified model weights saved to 'masked_model_weights.pth'")

    # 计算模型的预测准确率
    evaluate_model(model, data_loader, device)


if __name__ == "__main__":
    main()
