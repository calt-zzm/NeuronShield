# NeuronShield
This repository provides a comprehensive framework for analyzing, detecting, and mitigating adversarial attacks on neural networks. It includes utilities for identifying suspicious neurons, fine-tuning models to improve robustness, and evaluating the modified models. The repository is built with PyTorch and integrates popular adversarial attack libraries such as torchattacks.

## Features
### Adversarial Example Generation:
Supports Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks.
### Suspicious Neuron Detection:
Identifies neurons with abnormal activations in response to adversarial inputs.
Provides tools to analyze suspicious neurons at the layer and channel level.
### Fine-Tuning and Repair:
Fine-tunes specific layers to mitigate the effects of adversarial attacks.
Masks or modifies weights of channels with the highest anomaly scores.
### Evaluation:
Evaluates the model's performance on both clean and adversarial inputs.
Measures the impact of weights masking and fine-tuning on model accuracy.

# Repository Structure
## train.py
Implements training and fine-tuning of a model to repair suspicious neurons and improve adversarial robustness.
### Key functions:
fine_tune_model: Fine-tunes specific layers with high anomaly counts.
evaluate_model: Evaluates the model's accuracy on clean and adversarial datasets.
## test.py
Tests a pre-trained model's performance against adversarial attacks.
### Key components:
Loads pre-trained models (resnet18, resnet34, resnet50, etc.).
Evaluates adversarial and clean accuracy using the evaluate_model function from train.py.
## utils.py
Contains utility functions for model analysis and layer-wise operations.
### Key functions:
get_model_layers: Extracts all valid layers from a model.
get_layer_output: Captures intermediate outputs of specified layers.
get_anomaly_neurons: Identifies neurons with abnormal activations based on thresholds or statistical methods.
## neuron.py
Focuses on neuron-level anomaly detection and handling.
### Key components:
Custom ImageDataset for loading datasets with labels.
process_batch: Processes batches to identify misclassified or adversarial samples.
save_results: Saves the evaluation results in JSON format.
## conv.py
Handles channel-wise anomaly detection and masking in convolutional layers.
### Key functions:
count_anomaly_neurons_per_channel: Counts suspicious neurons in each channel of convolutional layers.
mask_top_10_percent_channels_by_weights: Masks the 10% most suspicious channels by setting their weights to zero.

# Example Workflow
Train a model and detect anomalies:
Use train.py to fine-tune a pre-trained ResNet50 on CIFAR-10.
Analyze the anomaly statistics printed during training.
Test the model's robustness:
Use test.py to evaluate the model on clean and adversarial inputs.
Perform channel-wise analysis:
Use conv.py to mask suspicious channels.
Re-evaluate the model using the modified weights.
