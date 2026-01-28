# Generative-AI Based MRI Research - Warm-up Studies

This repository documents the preliminary studies, coding experiments, and warm-up tasks for the **TÜBİTAK 1001** project titled **"Generative-AI Based Magnetic Resonance Imaging of Brain Currents"**.

The primary focus of this repository is to explore the fundamentals of Diffusion Models, Fine-Tuning techniques, and Guidance mechanisms following the [Hugging Face Diffusion Course](https://huggingface.co/learn/diffusion-course/unit0/1).

## Project Overview

* **Project:** Generative-AI Based Magnetic Resonance Imaging of Brain Currents (TÜBİTAK 1001)
* **Role:** Researcher Student (Candidate)
* **Objective:** To gain a solid understanding of diffusion models and prepare for the core research phase.

## Study Curriculum & Progress

I am following the first two units of the Hugging Face Diffusion Course as requested.

### **Prerequisites: PyTorch Essentials**
- [x] **PyTorch in 60 Minutes:** Quick start guide for PyTorch.
- [x] **Intro to Deep Learning with PyTorch:** Fundamental deep learning concepts with PyTorch.
  - *Note: The `6_pytorch/` folder contains materials from [Udacity's Deep Learning with PyTorch course](https://www.udacity.com/course/deep-learning-pytorch--ud188).*

### **Unit 1: Introduction to Diffusion Models**
- [x] **Theory:** Understanding the logic behind Diffusion Models (Forward/Reverse Process).
- [x] **Implementation:** Building a basic diffusion model from scratch using PyTorch.
- [x] **Practice:** Exploring the `diffusers` library and noise schedulers.

### **Unit 2: Fine-Tuning & Guidance**
- [x] **Fine-Tuning:** Training an existing model on a custom dataset.
- [x] **Guidance:** Understanding how to control generation outputs.
- [x] **Conditioning:** Implementing class-conditioned generation.


## Repository Structure

```text
.
├── pytorch_prerequisites/   # PyTorch training notebooks
│   ├── 1_pytorch_in_60_mins/
│   │   ├── 1_tensors.ipynb
│   │   ├── 2_a_gentleIntroduction_to_torchautograd.ipynb
│   │   ├── 3_neural_networks.ipynb
│   │   └── 4_training_a_classifier.ipynb
│   └── 2_intro_deeplearning_withpytorch/
│       ├── 1_introduction_to_neural_networks.ipynb
│       ├── 2_gradient_descent.ipynb
│       ├── 3_neural_network_architecture.ipynb
│       ├── 4_analyzing_student_data.ipynb
│       ├── 5_training_optimization.ipynb
│       ├── 6_pytorch/
│       │   ├── Part 1 - Tensors in PyTorch (Exercises).ipynb
│       │   ├── Part 2 - Neural Networks in PyTorch (Exercises).ipynb
│       │   ├── Part 3 - Training Neural Networks (Exercises).ipynb
│       │   ├── Part 4 - Fashion-MNIST (Exercises).ipynb
│       │   ├── Part 5 - Inference and Validation (Exercises).ipynb
│       │   ├── Part 6 - Saving and Loading Models.ipynb
│       │   ├── Part 7 - Loading Image Data (Exercises).ipynb
│       │   ├── Part 8 - Transfer Learning (Exercises).ipynb
│       │   ├── fc_model.py
│       │   ├── helper.py
│       │   ├── checkpoint.pth
│       │   └── assets/
│       ├── 7_convolutional_neural_nteworks/
│       │   ├── 1_intro_to_cnn.ipynb
│       │   ├── 2_mnist_mlp_exercise.ipynb
│       │   ├── 3_conv_visualization.ipynb
│       │   ├── 4_cifar10_cnn_exercise.ipynb
│       │   └── data/
│       ├── 8_style_transfer/
│       │   ├── 1_intro_to_styletransfer.ipynb
│       │   ├── 2_Style_Transfer_Exercise.ipynb
│       │   └── images...
│       ├── 9_Recurrent _Neural_Networks/
│       │   ├── 1_intro_to_rnn.ipynb
│       │   ├── 2_Simple_RNN.ipynb
│       │   ├── 3_Character_Level_RNN_Exercise.ipynb
│       │   ├── 4_Sentiment_RNN_Exercise.ipynb
│       │   └── data/
│       ├── data.csv
│       ├── data2.csv
│       ├── student_data.csv
│       └── image.png, image2.png, image3.png, image4.png, image5.png
├── unit1_introduction/      # Notebooks and outputs for Unit 1
│   ├── 01_introduction_to_diffusers.ipynb
│   ├── 02_diffusion_models_from_scratch.ipynb
│   └── my_pipeline/         # Model outputs
├── unit2_finetuning/        # Notebooks and data for Unit 2
│   ├── 01_finetuning_and_guidance.ipynb
│   ├── 02_class_conditioned_diffusion_model_example.ipynb
│   └── outputs/
├── presentation/            # Summary presentation (PPTX/PDF)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation