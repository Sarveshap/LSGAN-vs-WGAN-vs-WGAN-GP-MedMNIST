# 🧠 MedMNIST-GAN-Comparison: LSGAN vs WGAN vs WGAN-GP

This project compares three Generative Adversarial Network (GAN) variants — **LSGAN**, **WGAN**, and **WGAN-GP** — on the PneumoniaMNIST dataset from MedMNIST. The goal is to generate realistic chest X-ray images and evaluate them using quantitative metrics and visual inspection.

---

## 📊 Dataset: PneumoniaMNIST (MedMNIST)

- **Source**: https://medmnist.com/
- **Description**: 28x28 grayscale chest X-ray images
- **Task**: Binary classification — Pneumonia (1) vs Normal (0)
- **Total Samples**: 5,856 training | 1,496 validation | 1,498 test

---

## 🚀 GAN Models Implemented

1. **Least Squares GAN (LSGAN)**  
   - Uses least squares loss instead of binary cross-entropy  
   - Promotes stable gradients

2. **Wasserstein GAN (WGAN)**  
   - Replaces JS divergence with Wasserstein distance  
   - Uses weight clipping for Lipschitz continuity

3. **Wasserstein GAN with Gradient Penalty (WGAN-GP)**  
   - Adds gradient penalty to improve WGAN training  
   - Avoids weight clipping and improves convergence

All models use identical Generator and Discriminator architectures to ensure fair comparison.

---

## 🛠 Setup & Installation

### ✅ Install Requirements
```bash
pip install torch torchvision matplotlib numpy tensorboard medmnist scikit-learn
