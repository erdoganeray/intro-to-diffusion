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
- [ ] **Intro to Deep Learning with PyTorch:** Fundamental deep learning concepts with PyTorch.

### **Unit 1: Introduction to Diffusion Models**
- [ ] **Theory:** Understanding the logic behind Diffusion Models (Forward/Reverse Process).
- [ ] **Implementation:** Building a basic diffusion model from scratch using PyTorch.
- [ ] **Practice:** Exploring the `diffusers` library and noise schedulers.

### **Unit 2: Fine-Tuning & Guidance**
- [ ] **Fine-Tuning:** Training an existing model on a custom dataset.
- [ ] **Guidance:** Understanding how to control generation outputs.
- [ ] **Conditioning:** Implementing class-conditioned generation.


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
│       └── 1_introduction_to_neural_networks.ipynb
├── unit1_introduction/      # Notebooks and outputs for Unit 1
│   ├── 01_diffusion_intro.ipynb
│   └── outputs/
├── unit2_finetuning/        # Notebooks and data for Unit 2
│   ├── 01_finetuning.ipynb
│   └── outputs/
├── presentation/            # Summary presentation (PPTX/PDF)
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation