# MLP vs CNN: An Ablation Study on CIFAR-10

A controlled experimental study comparing Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures on the CIFAR-10 image classification benchmark. Five experiments isolate specific architectural properties — spatial invariance, shift robustness, model capacity, feature interpretability, and the role of pooling — to understand *why* CNNs outperform MLPs on image data.

---

## Experiments

| # | Experiment | What it Tests |
|---|---|---|
| 1 | Model Capacity Scaling | How MLP accuracy changes across 7 architectures (128 → 512×256×128 neurons) |
| 2 | Pixel Shuffling | Whether MLP relies on spatial pixel order or just pixel identity |
| 3 | MLP vs CNN Direct Comparison | Accuracy gap between flat and spatially-aware architectures |
| 4 | Feature Map Visualization | What patterns each CNN layer learns to detect |
| 5 | Image Shift Robustness | How a 2-pixel shift affects MLP vs CNN accuracy |
| 6 | Pooling Ablation | Effect of removing MaxPool layers from CNN |

---

## Results

### Experiment 1 — Model Capacity (MLP)

| Architecture | Parameters | Train Acc | Val Acc | Test Acc |
|---|---|---|---|---|
| [128] | ~394K | 0.6306 | 0.5075 | 0.5075 |
| [256] | ~787K | 0.6673 | 0.5267 | 0.5267 |
| [512] | ~1.6M | 0.6849 | 0.5079 | 0.5079 |
| [256, 128] | ~921K | 0.7028 | 0.5303 | 0.5303 |
| [512, 256] | ~1.9M | 0.7473 | 0.5367 | 0.5367 |
| [1024, 512] | ~3.8M | 0.7596 | 0.5275 | 0.5275 |
| [512, 256, 128] | ~2.0M | 0.7364 | 0.5290 | 0.5290 |

### Experiment 2 — Pixel Shuffling

| Condition | Test Accuracy |
|---|---|
| MLP (original pixels) | 0.5196 |
| MLP (shuffled pixels) | 0.5236 |
| Drop | -0.0040 |

### Experiment 3 — MLP vs CNN

| Model | Test Accuracy |
|---|---|
| MLP [512, 256] | 0.5196 |
| CNN (3 conv layers) | 0.7316 |

### Experiment 5 — Shift Robustness (2-pixel shift)

| Model | Original Acc | Shifted Acc | Drop |
|---|---|---|---|
| MLP | 0.5196 | 0.4182 | 0.1014 |
| CNN | 0.7316 | 0.6773 | 0.0543 |

### Experiment 6 — Pooling Ablation

| Model | Test Accuracy |
|---|---|
| CNN with MaxPool | 0.7316 |
| CNN without MaxPool | 0.6434 |

---

## Key Findings

- **MLP accuracy is nearly unchanged by pixel shuffling** — confirming it treats input as an unordered set of pixel values, with no spatial reasoning
- **CNN outperforms MLP by ~21.2%** on clean test data despite having a simpler architecture, due to local feature extraction via convolution
- **CNN accuracy drops less than MLP under 2-pixel shift**, demonstrating translation robustness from spatial weight sharing
- **Removing MaxPool layers degrades CNN accuracy**, showing pooling is essential for hierarchical feature aggregation and spatial downsampling
- **MLP capacity scaling shows diminishing returns** beyond [512, 256] — larger models overfit without improving test accuracy

---

## Architecture Details

### MLP
Input (3072) → Linear → ReLU → Linear → ReLU → ... → Linear (10)
- Input: CIFAR-10 images flattened to 1D vectors (3 × 32 × 32 = 3072)
- Activation: ReLU
- Output: 10-class logits (CrossEntropyLoss)
- Optimizer: Adam (lr=1e-3)

### CNN (SimpleCNN)
Conv2d(3→32) → ReLU → MaxPool   [32×32 → 16×16]

Conv2d(32→64) → ReLU → MaxPool  [16×16 → 8×8]

Conv2d(64→128) → ReLU → MaxPool [8×8 → 4×4]

Flatten → Linear(2048→256) → ReLU → Linear(256→10)
- Kernel size: 3×3 with padding=1 (preserves spatial dimensions before pooling)
- Pooling: MaxPool2d(kernel=2, stride=2)
- Optimizer: Adam (lr=1e-3)

---

## Dataset

**CIFAR-10** — 60,000 32×32 colour images across 10 classes

| Split | Size |
|---|---|
| Train | 40,000 |
| Validation | 10,000 |
| Test | 10,000 |

Classes: `plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

Preprocessing: normalized to mean=0.5, std=0.5 per channel.

---

## Visualizations

The notebook produces the following plots:

- Training/validation accuracy and loss curves for MLP and CNN
- Accuracy vs model capacity (number of parameters) for 7 MLP configs
- Bar chart: MLP original vs shuffled pixel accuracy
- Bar chart: MLP vs CNN test accuracy
- Feature maps from `conv1`, `conv2`, `conv3` for a sample test image
- Bar chart: MLP vs CNN accuracy before and after 2-pixel shift
- Bar chart: CNN with vs without MaxPool layers
- Grid of 10 misclassified test images with true and predicted labels
