# 🔢 MNIST Handwritten Digit Recognizer with PyTorch CNN

A Convolutional Neural Network (CNN) classifier built and trained in **PyTorch** on the classic MNIST handwritten digits dataset (classes 0–9). The project includes a pre-trained model checkpoint and an image preprocessing inference pipeline to recognize digits from external test images.

---

## 📌 Features

- **Custom 3-Layer CNN Architecture**: Multi-stage 2D convolutions with ReLU activations designed for spatial feature extraction on $28 \times 28$ images.
- **Pre-trained Checkpoint Included**: Comes with `model_state.pt` ready for out-of-the-box inference without requiring GPU retraining.
- **Robust Image Preprocessing**: Automatically converts external RGB photos, screenshots, or drawings into normalized 1-channel MNIST tensors.
- **Command-Line Inference Tool**: Inspect any digit image with formatted prediction outputs and softmax confidence scores.

---

## 🧠 Model Architecture

```
Input (1 × 28 × 28)
  │
  ├──► Conv2D (1 ➔ 32, 3×3)  + ReLU  ──► (32 × 26 × 26)
  ├──► Conv2D (32 ➔ 64, 3×3) + ReLU  ──► (64 × 24 × 24)
  ├──► Conv2D (64 ➔ 64, 3×3) + ReLU  ──► (64 × 22 × 22)
  │
  ├──► Flatten (64 × 22 × 22 = 30,976 features)
  └──► Linear (30,976 ➔ 10 Classes)  ──► Softmax Probabilities
```

- **Loss Function**: `nn.CrossEntropyLoss`
- **Optimizer**: Adam ($\text{learning rate} = 10^{-3}$)
- **Batch Size**: 32

---

## 📁 Repository Structure

```
mnist-digit-recognizer-pytorch/
├── .gitignore               # Ignores Python caches and raw dataset downloads
├── README.md                # Network architecture & documentation
├── requirements.txt         # Dependencies (torch, torchvision, Pillow)
├── train.py                 # Training script (downloads MNIST, trains 20 epochs)
├── predict.py               # Inference pipeline accepting image CLI inputs
├── model_state.pt           # Pre-trained PyTorch model checkpoint (1.4 MB)
└── sample_images/           # Sample test digit images (2.png, eight.png, img_3.jpg...)
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/AureusBlaze/mnist-digit-recognizer-pytorch.git
cd mnist-digit-recognizer-pytorch
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Predictions on Test Images
You can pass any test image directly through `predict.py`:

```bash
python predict.py img_3.jpg
```
*Output:*
```
[*] Input Image     : img_3.jpg
[*] Predicted Digit : 3
[*] Confidence      : 99.82%
```

Other included test images:
```bash
python predict.py eight.png
python predict.py 2.png
python predict.py 4.png
```

### 4. Train from Scratch (Optional)
To retrain the neural network on the full MNIST training set:
```bash
python train.py
```
*(This automatically downloads MNIST and saves updated weights to `model_state.pt`).*

---

## 🛠️ Tech Stack
- **Python 3.x**
- **PyTorch (`torch`, `torch.nn`)**
- **Torchvision** (Transforms & dataset loaders)
- **Pillow (`PIL`)** (Image input handling)
