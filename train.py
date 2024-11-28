import math
import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.models import resnet50
from tqdm import tqdm
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
batch_size = 128  # CIFAR-10 数据集较小，使用较小的 batch size
beta_repair = 0.01
gamma = 0.1


def generate_adversarial_examples(model, img_tensors, label_tensors, atk):
    """生成对抗样本"""
    model.eval()  # 评估模式，避免 BN 和 Dropout 的影响
    adv_tensors = atk(img_tensors, label_tensors).to(device)
    return adv_tensors


def compute_repair_loss(layer_outputs, adv_layer_outputs, anomalous_neurons):
    """计算异常神经元修正损失"""
    repair_loss = 0.0
    count = 0

    for i, (layer_output, adv_layer_output) in enumerate(zip(layer_outputs, adv_layer_outputs)):
        if i in anomalous_neurons:
            anomaly_indices = anomalous_neurons[i]
            diff = adv_layer_output[:, anomaly_indices] - layer_output[:, anomaly_indices]
            repair_loss += torch.sum(diff ** 2)
            count += len(anomaly_indices)

    if count > 0:
        repair_loss /= count
    return repair_loss


def fine_tune_model(model, anomaly_neurons, train_loader, val_loader, top_layers=5, learning_rate=1e-4, epochs=500,
                    patience=20):
    """
    针对异常神经元数最多的 top_layers 层进行训练，更新其权重，保持其他通道权重不变。
    """
    criterion = nn.CrossEntropyLoss()
    layer_dict = utils.get_model_layers(model)
    best_loss = math.inf
    step = 0
    early_stop_count = 0

    # 统计各层的异常神经元数量，并选出最多的 top_layers 层
    layer_anomaly_count = {key: len(neurons) for key, neurons in anomaly_neurons.items()}
    top_layers_sorted = sorted(layer_anomaly_count, key=layer_anomaly_count.get, reverse=True)[:top_layers]

    # 标记需要训练的层，冻结其他层
    for name, layer in layer_dict.items():
        if name in top_layers_sorted:  # 如果该层是异常神经元数最多的 top_layers 层之一
            for param in layer.parameters():
                param.requires_grad = True  # 解冻这些层的参数
        else:
            for param in layer.parameters():
                param.requires_grad = False  # 冻结其他层的参数

    # 过滤出可训练的参数
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = list(trainable_params)  # 确保非空

    if not trainable_params:
        raise ValueError("No trainable parameters found. Check your anomaly_neurons or layer names.")

    # 定义优化器只优化这些通道
    optimizer = optim.Adam(trainable_params, lr=learning_rate)

    # 进入整体训练循环，对所有解冻的层一起训练
    for epoch in range(epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for img_tensors, label_tensors in train_pbar:
            img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

            # 生成对抗样本（提前生成并复用）
            atk = torchattacks.PGD(model, steps=20, eps=8 / 255)
            adv_tensors = generate_adversarial_examples(model, img_tensors, label_tensors, atk)
            layer_outputs = utils.get_layer_output(model, img_tensors)
            adv_layer_outputs = utils.get_layer_output(model, adv_tensors)
            optimizer.zero_grad()
            outputs = model(img_tensors)
            adv_outputs = model(adv_tensors)

            # loss = criterion(adv_outputs, label_tensors) + criterion(outputs, label_tensors)
            loss_ce = criterion(adv_outputs, label_tensors) + criterion(outputs, label_tensors)
            loss_repair = compute_repair_loss(layer_outputs, adv_layer_outputs, anomaly_neurons)
            loss = loss_ce + beta_repair * math.exp(-gamma * epoch) * loss_repair
            print(loss_ce, loss_repair, loss)
            loss.backward()
            optimizer.step()

            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f"Epoch[{epoch + 1}/{epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)

        # 进行验证
        model.eval()
        loss_record = []
        for img_tensors, label_tensors in val_loader:
            img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

            # 生成对抗样本
            adv_tensors = generate_adversarial_examples(model, img_tensors, label_tensors, atk)

            with torch.no_grad():
                outputs = model(img_tensors)
                adv_outputs = model(adv_tensors)
                loss = criterion(adv_outputs, label_tensors) + criterion(outputs, label_tensors)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f"Epoch[{epoch + 1}/{epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}")

        # Early stopping 和模型保存
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), "fine_tuned_model.pth")
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= patience:
            print("\nModel is not improving, stop the training session.")
            break

    return model


def evaluate_model(model, data_loader):
    """使用给定数据集评估模型的准确率"""
    correct = 0
    total = 0
    adv_correct = 0
    model.eval()  # 评估模式
    atk = torchattacks.PGD(model, steps=20, eps=8 / 255)

    for img_tensors, label_tensors in tqdm(data_loader):
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

        # 生成对抗样本
        adv_tensors = generate_adversarial_examples(model, img_tensors, label_tensors, atk)

        outputs = model(img_tensors)
        _, predicted = torch.max(outputs, 1)
        adv_outputs = model(adv_tensors)
        _, adv_predicted = torch.max(adv_outputs, 1)
        total += label_tensors.size(0)
        correct += (predicted == label_tensors).sum().item()
        adv_correct += (adv_predicted == label_tensors).sum().item()
    accuracy = correct / total
    adv_accuracy = adv_correct / total
    print(f'Origin Accuracy: {accuracy * 100:.2f}%')
    print(f'Adv Accuracy: {adv_accuracy * 100:.2f}%')
    return accuracy


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载预训练的模型
    model = resnet50(weights=None).to(device)
    model.fc = torch.nn.Linear(2048, 10).to(device)
    model.load_state_dict(torch.load("resnet50-224-cifar10-95.55.pth", map_location=device))

    # 使用 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 数据加载器
    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=4, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=4, batch_size=batch_size, shuffle=False)

    # 存储所有异常神经元
    all_different_neurons = {}

    for img_tensors, label_tensors in tqdm(train_loader):
        img_tensors, label_tensors = img_tensors.to(device), label_tensors.to(device)

        # 生成对抗样本
        atk = torchattacks.PGD(model, steps=20, eps=8 / 255)
        adv_tensors = generate_adversarial_examples(model, img_tensors, label_tensors, atk)

        # 获取层输出和异常神经元
        layers_output = utils.get_layer_output(model, img_tensors)
        adv_layer_output = utils.get_layer_output(model, adv_tensors)
        different_neurons, _ = utils.get_anomaly_neurons(layers_output, adv_layer_output, threshold=0.2, mask_ratio=0.1)

        # 存储所有异常神经元
        all_different_neurons.update(different_neurons)

    # 针对异常神经元训练模型
    model = fine_tune_model(model, all_different_neurons, train_loader, val_loader, top_layers=5)

    # 评估模型
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    main()
