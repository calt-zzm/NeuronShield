import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_valid(module):
    is_vd = (isinstance(module, nn.Linear)
             or isinstance(module, nn.Conv2d)
             or isinstance(module, nn.ReLU)
             or isinstance(module, nn.Dropout)
             or isinstance(module, nn.LayerNorm)
             or isinstance(module, nn.Embedding)
             or isinstance(module, nn.MultiheadAttention)
             or isinstance(module, nn.BatchNorm2d)
             or isinstance(module, nn.MaxPool2d)
             )
    return is_vd


def iterate_module(name, module, name_list, module_list):
    is_vd = is_valid(module)
    if is_vd:
        return name_list + [name], module_list + [module]
    else:
        if len(list(module.named_children())):
            for child_name, child_module in module.named_children():
                name_list, module_list = \
                    iterate_module(child_name, child_module, name_list, module_list)
        return name_list, module_list


def get_model_layers(model):
    layer_dict = {}
    name_counter = {}
    for name, module in model.named_children():
        name_list, module_list = iterate_module(name, module, [], [])
        assert len(name_list) == len(module_list)
        for i, _ in enumerate(name_list):
            module = module_list[i]
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_dict['%s-%d' % (class_name, name_counter[class_name])] = module
    return layer_dict


def get_layer_output(model, data):
    name_counter = {}
    layer_output_dict = {}
    layer_dict = get_model_layers(model)

    with torch.no_grad():
        def hook(module, _, output):
            class_name = module.__class__.__name__
            if class_name not in name_counter.keys():
                name_counter[class_name] = 1
            else:
                name_counter[class_name] += 1
            layer_output_dict['%s-%d' % (class_name, name_counter[class_name])] = output

        hooks = []
        for layer, module in layer_dict.items():
            hooks.append(module.register_forward_hook(hook))
        try:
            _ = model(data.to(device))
        finally:
            for h in hooks:
                h.remove()
    return layer_output_dict


def get_total_neurons(model, img_tensors, layers_output):
    total_neurons = {}
    for key in layers_output.keys():
        if key not in total_neurons:
            output_tensor_tuple = get_layer_output(model, img_tensors)[key]
            if isinstance(output_tensor_tuple, tuple):
                total_neurons[key] = int(output_tensor_tuple[0].numel() / img_tensors.shape[0])
            else:
                total_neurons[key] = int(output_tensor_tuple.numel() / img_tensors.shape[0])
    return total_neurons


def get_anomaly_neurons(layers_output, adv_layer_output, threshold=0.1, mask_ratio=0.2):
    different_neurons = {}
    masks = {}
    for key in layers_output.keys():
        if isinstance(layers_output[key], tuple):
            diff = torch.abs(layers_output[key][0] - adv_layer_output[key][0])
        else:
            diff = torch.abs(layers_output[key] - adv_layer_output[key])

        diff_mean = torch.mean(diff, dim=0)
        diff_indices = torch.nonzero(diff_mean > threshold).squeeze()
        different_neurons[key] = diff_indices.tolist()

        # sorted_diff_indices = torch.argsort(diff_mean, descending=True)
        # num_to_mask = int(len(sorted_diff_indices) * mask_ratio)
        # mask = torch.ones_like(diff_mean)
        # if len(sorted_diff_indices.shape) == 1:
        #     mask[sorted_diff_indices[:num_to_mask]] = 0
        # elif len(sorted_diff_indices.shape) == 2:
        #     for idx in sorted_diff_indices[:num_to_mask]:
        #         mask[:, idx] = 0
        # elif len(sorted_diff_indices.shape) == 3:
        #     for idx in sorted_diff_indices[:num_to_mask]:
        #         mask[:, idx[0], idx[1]] = 0
        # masks[key] = mask
    return different_neurons, masks

# def get_anomaly_neurons(layers_output, adv_layer_output, contamination=0.1, mask_ratio=0.2):
#     different_neurons = {}
#     masks = {}
#
#     for key in layers_output.keys():
#         # 获取正常和对抗样本的神经元输出差异
#         if isinstance(layers_output[key], tuple):
#             diff = torch.abs(layers_output[key][0] - adv_layer_output[key][0])
#         else:
#             diff = torch.abs(layers_output[key] - adv_layer_output[key])
#
#         # 计算每个神经元的均值差异
#         diff_mean = torch.mean(diff, dim=0).cpu().numpy()
#
#         # 使用 Isolation Forest 检测异常神经元
#         isolation_forest = IsolationForest(contamination=contamination, random_state=2024)
#         diff_mean_reshaped = diff_mean.reshape(-1, 1)  # Isolation Forest 需要二维输入
#         isolation_forest.fit(diff_mean_reshaped)
#
#         # 预测异常值 (-1 是异常值，1 是正常值)
#         predictions = isolation_forest.predict(diff_mean_reshaped)
#         anomaly_indices = torch.nonzero(torch.tensor(predictions) == -1).squeeze()
#
#         # 将异常神经元索引保存进不同神经元字典
#         different_neurons[key] = anomaly_indices.tolist()
#     return different_neurons, masks


def apply_masks(model, img_tensors, masks):
    masked_outputs = {}
    with torch.no_grad():
        outputs = get_layer_output(model, img_tensors)
        for key in outputs.keys():
            if isinstance(outputs[key], tuple):
                masked_outputs[key] = outputs[key][0] * masks[key]
            else:
                masked_outputs[key] = outputs[key] * masks[key]
    return masked_outputs
