import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torchattacks
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import vgg11, VGG11_Weights, resnet18, ResNet18_Weights
from tqdm import tqdm
import utils


class ImageDataset(Dataset):
    """自定义数据集类，用于加载图像和标签"""

    def __init__(self, image_dir, labels_dir, transform=None):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        # 从标签文件中读取图像名称和标签
        labels_df = pd.read_csv(self.labels_dir, header=None, sep=" ", names=['filename', 'label'])
        img_name = labels_df.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_predictions(model, img_tensors):
    """获取模型的预测结果"""
    with torch.no_grad():
        logits = model(img_tensors)
        _, predicted = torch.max(torch.softmax(logits, dim=1), 1)
    return predicted


def process_batch(model, atk, img_tensors, label_tensors, device):
    """处理每个批次，返回预测结果和错误样本"""
    adv_tensors = atk(img_tensors, label_tensors).to(device)

    predicted = get_predictions(model, img_tensors)
    adv_predicted = get_predictions(model, adv_tensors)

    mismatched_indices = ~predicted.eq(adv_predicted)
    mismatched_img_tensors = img_tensors[mismatched_indices]
    mismatched_adv_tensors = adv_tensors[mismatched_indices]

    return predicted, adv_predicted, mismatched_img_tensors, mismatched_adv_tensors, mismatched_indices


def save_results(eps, ori_wrong_predictions, adv_wrong_predictions, masked_wrong_predictions):
    """保存结果到 JSON 文件"""
    output_data = {
        "origin wrong predictions": ori_wrong_predictions,
        "adversarial wrong predictions": adv_wrong_predictions,
        "masked wrong predictions": masked_wrong_predictions
    }
    with open(f'FGSM_{eps}_ImageNet_ResNet18.json', 'w') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)


def main():
    """主函数，设置设备、加载模型和数据，处理攻击"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256]),
        transforms.CenterCrop([224]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载预训练的模型
    # model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device).eval()
    # 数据集和数据加载器
    val_images_path = "D:/Program/safetyTest/ImageNet/images"
    val_labels_path = "D:/Program/safetyTest/ImageNet/ImageNet_val_labels.txt"
    dataset = ImageDataset(val_images_path, val_labels_path, transform=preprocess)
    data_loader = DataLoader(dataset, num_workers=4, batch_size=16, shuffle=False)
    # 设置攻击参数
    epsilon = [0.05]

    for eps in epsilon:
        ori_wrong_predictions = 0
        adv_wrong_predictions = 0
        masked_wrong_predictions = 0

        atk = torchattacks.FGSM(model, eps=eps)

        for img_tensors, label_tensors in tqdm(data_loader):
            img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)
            # 处理批次数据
            predicted, adv_predicted, mismatched_img_tensors, mismatched_adv_tensors, mismatched_indices = \
                process_batch(model, atk, img_tensors, label_tensors, device)

            # 统计错误预测
            ori_wrong_predictions += (predicted != label_tensors).sum().item()
            adv_wrong_predictions += (adv_predicted != label_tensors).sum().item()

            # 获取层输出与异常神经元
            layers_output = utils.get_layer_output(model, mismatched_img_tensors)
            adv_layer_output = utils.get_layer_output(model, mismatched_adv_tensors)
            different_neurons, masks = utils.get_anomaly_neurons(layers_output, adv_layer_output, threshold=0.1,
                                                                 mask_ratio=0.1)

            # 应用掩码并计算预测
            masked_outputs = utils.apply_masks(model, mismatched_img_tensors, masks)
            _, masked_predicted = torch.max(torch.softmax(masked_outputs['Linear-1'], dim=1), 1)
            masked_wrong_predictions += (masked_predicted != label_tensors[mismatched_indices]).sum().item()

        # 保存当前 epsilon 的结果
        save_results(eps, ori_wrong_predictions, adv_wrong_predictions, masked_wrong_predictions)


if __name__ == '__main__':
    main()
