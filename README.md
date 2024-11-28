# NeuronShield
This repository provides a comprehensive framework for analyzing, detecting, and mitigating adversarial attacks on neural networks. It includes utilities for identifying anomalous neurons, fine-tuning models to improve robustness, and evaluating the modified models. The repository is built with PyTorch and integrates popular adversarial attack libraries such as torchattacks.

## Features
### Adversarial Example Generation:
Supports Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks.
### Anomalous Neuron Detection:
Identifies neurons with abnormal activations in response to adversarial inputs.
Provides tools to analyze anomalous neurons at the layer and channel level.
### Fine-Tuning and Repair:
Fine-tunes specific layers to mitigate the effects of adversarial attacks.
Masks or modifies weights of channels with the highest anomaly scores.
### Evaluation:
Evaluates the model's performance on both clean and adversarial inputs.
Measures the impact of weights masking and fine-tuning on model accuracy.
