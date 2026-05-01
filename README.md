# 🧠 TinyVGG CNN Implementation using PyTorch

A PyTorch implementation of the **TinyVGG** Convolutional Neural Network architecture, trained and evaluated on the **FashionMNIST** dataset for multi-class image classification.

---

## 📌 Overview

This project demonstrates how to build, train, and evaluate a lightweight CNN (TinyVGG) from scratch using PyTorch. It covers the full deep learning pipeline — from data loading and preprocessing to model training, evaluation, and prediction visualization.

---

## 🗂️ Dataset

- **Dataset:** [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist)
- **Classes (10):** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28×28 grayscale

---

## 🏗️ Model Architecture — TinyVGG

```
TinyVGGRep(
  (block_1): Sequential(
    Conv2d(1, 10, kernel_size=3, stride=1, padding=1)
    ReLU()
    Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    ReLU()
    MaxPool2d(kernel_size=2)
  )
  (block_2): Sequential(
    Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    ReLU()
    Conv2d(10, 10, kernel_size=3, stride=1, padding=1)
    ReLU()
    MaxPool2d(kernel_size=2)
  )
  (classifier): Sequential(
    Flatten()
    Linear(hidden_units → 10)
  )
)
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch Size | 32 |
| Optimizer | SGD |
| Learning Rate | 0.1 |
| Loss Function | CrossEntropyLoss |
| Accuracy Metric | MulticlassAccuracy (torchmetrics) |
| Device | GPU (CUDA) / CPU |

---

## 📦 Requirements

```bash
pip install torch torchvision torchmetrics tqdm matplotlib pandas
```

Or if running on Google Colab:

```bash
!pip install torchmetrics
```

> All other dependencies (PyTorch, torchvision, matplotlib, etc.) come pre-installed in Colab.

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SadeepM/Implementation-of-Tiny-VGG-Convolutional-Neural-Network-using-PyTorch.git
   ```

2. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter:
   ```
   Implementation_of_Tiny_VGG_CNN_using_PyTorch.ipynb
   ```

3. Run all cells from top to bottom.

---

## 📊 What's Covered

- ✅ Loading and exploring the FashionMNIST dataset
- ✅ Visualizing sample images
- ✅ Creating PyTorch `DataLoader` with mini-batches
- ✅ Building the TinyVGG model with `nn.Module`
- ✅ Device-agnostic code (GPU/CPU)
- ✅ Training and testing loop functions
- ✅ Tracking loss and accuracy per epoch
- ✅ Evaluating the model on test data
- ✅ Visualizing predictions vs ground truth labels

---

## 🛠️ Built With

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/)
- [Torchmetrics](https://torchmetrics.readthedocs.io/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://tqdm.github.io/)
- Google Colab (T4 GPU)

---

## 📁 Project Structure

```
📦 repo
 ┗ 📓 Implementation_of_Tiny_VGG_CNN_using_PyTorch.ipynb
 ┗ 📄 README.md
```