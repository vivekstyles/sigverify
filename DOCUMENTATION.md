# Signature Verification System: Complete Architecture & Deep Learning Guide

This document provides a comprehensive technical breakdown of this Signature Verification project and explains every Deep Learning and Neural Network concept used from first principles.

---

## Table of Contents

1. [Project Overview & Problem Formulation](#1-project-overview--problem-formulation)
2. [Why Not Standard Classification? The Metric Learning Paradigm](#2-why-not-standard-classification-the-metric-learning-paradigm)
3. [Siamese Neural Network Architecture](#3-siamese-neural-network-architecture)
4. [Deep Dive into Neural Network Components](#4-deep-dive-into-neural-network-components)
   - [4.1 Convolutional Neural Networks (CNN)](#41-convolutional-neural-networks-cnn)
   - [4.2 Residual Connections (ResNet-18) & Solving Vanishing Gradients](#42-residual-connections-resnet-18--solving-vanishing-gradients)
   - [4.3 Transfer Learning & 1-Channel Adaptation](#43-transfer-learning--1-channel-adaptation)
   - [4.4 Global Average Pooling (GAP)](#44-global-average-pooling-gap)
   - [4.5 Fully Connected (Dense) Projection Head](#45-fully-connected-dense-projection-head)
   - [4.6 Activation Functions (ReLU)](#46-activation-functions-relu)
   - [4.7 Regularization: Dropout & Weight Decay](#47-regularization-dropout--weight-decay)
   - [4.8 L2 Normalization & Hyper-spherical Embeddings](#48-l2-normalization--hyper-spherical-embeddings)
5. [Loss Function: Contrastive Loss](#5-loss-function-contrastive-loss)
6. [Pair Generation & Mining Strategy](#6-pair-generation--mining-strategy)
7. [Data Augmentation Pipeline](#7-data-augmentation-pipeline)
8. [Optimization & Training Dynamics](#8-optimization--training-dynamics)
   - [8.1 Adam Optimizer](#81-adam-optimizer)
   - [8.2 Learning Rate Scheduling (ReduceLROnPlateau)](#82-learning-rate-scheduling-reducelronplateau)
   - [8.3 Early Stopping](#83-early-stopping)
9. [Biometric Evaluation Metrics: Distance, FAR, FRR & EER](#9-biometric-evaluation-metrics-distance-far-frr--eer)
10. [PyTorch Model Mechanics: State Dicts & `.pth` Checkpoints](#10-pytorch-model-mechanics-state-dicts--pth-checkpoints)
11. [Domain Shift & Practical Preprocessing Insights](#11-domain-shift--practical-preprocessing-insights)
12. [File-by-File Codebase Tour](#12-file-by-file-codebase-tour)
13. [CLI Reference & Quick Start](#13-cli-reference--quick-start)

---

## 1. Project Overview & Problem Formulation

### Offline vs. Online Signature Verification
- **Online Verification**: Captures dynamic temporal data during the act of signing using a digital stylus (pen velocity, acceleration, pen-up/down timings, angle, and pressure trajectory).
- **Offline Verification** (This Project): Analyzes static, 2D visual raster images (scanned documents, cheques, photographed forms). No temporal or pen-velocity data is available. The system must extract spatial, geometric, and stroke-morphology features strictly from pixels.

### Writer-Dependent vs. Writer-Independent Verification
- **Writer-Dependent**: A custom model is trained for *each individual user*. If 1,000 employees join an organization, 1,000 separate models must be trained and stored. This does not scale.
- **Writer-Independent (What We Built)**: A single universal feature encoder is trained. It learns what makes *any* handwriting genuine versus forged. It can verify signatures from people it has never seen during training without requiring fine-tuning or retraining.

---

## 2. Why Not Standard Classification? The Metric Learning Paradigm

### The Failure of Standard Softmax Classification
In classical deep learning image classification (e.g., Cat vs. Dog, or MNIST Digits 0–9):
1. The model outputs a probability distribution across fixed classes: $P(y = c \mid x) = \frac{e^{z_c}}{\sum_k e^{z_k}}$.
2. The number of output classes is fixed at compile time (e.g., `num_classes = 6`).

**Why this fails for signature verification:**
- If user #7 arrives, a standard classifier has no output neuron for them. The architecture breaks.
- Classification encourages the network to memorize class-specific identity markers rather than general stroke fluidity, curvature variations, and tremor indicators.

### Metric Learning (Distance-Based Representation)
Instead of predicting an identity, we train an **Encoder** $f_\theta(x)$ that maps an image $x$ into a $D$-dimensional continuous vector space ($\mathbb{R}^{128}$):

$$x \in \mathbb{R}^{1 \times 155 \times 220} \xrightarrow{f_\theta} \mathbf{e} \in \mathbb{R}^{128}$$

The objective is geometric:
- **Intra-class compactness**: If $x_1$ and $x_2$ are signatures written by the **same** person, their embeddings $\mathbf{e}_1$ and $\mathbf{e}_2$ must be placed very close together: $\|\mathbf{e}_1 - \mathbf{e}_2\|_2 \approx 0$.
- **Inter-class separability**: If $x_1$ and $x_3$ are from **different** signers (or one is a skilled forgery), their embeddings must be pushed far apart: $\|\mathbf{e}_1 - \mathbf{e}_3\|_2 \ge m$, where $m$ is a safety margin.

---

## 3. Siamese Neural Network Architecture

A **Siamese Network** consists of two identical subnetworks that **share the exact same parameters and weights** ($\theta$).

```
Image A (1x155x220) ────► [ Shared Encoder f_θ ] ────► Embedding A (128-d) ──┐
                                                                              ├──► Euclidean Distance ──► Loss / Decision
Image B (1x155x220) ────► [ Shared Encoder f_θ ] ────► Embedding B (128-d) ──┘
                               ▲
                        (Shared Weights)
```

### Why Weight Sharing is Crucial:
1. **Symmetry**: $D(f(A), f(B)) = D(f(B), f(A))$. The order in which images are compared does not matter.
2. **Consistency**: Two identical images will map to the exact same point in latent space, yielding a distance of 0.
3. **Parameter Efficiency**: We only store and update one set of weights (~11.3 million parameters).

---

## 4. Deep Dive into Neural Network Components

### 4.1 Convolutional Neural Networks (CNN)
A Convolutional layer slides small learnable filters (kernels, e.g., $3 \times 3$ or $7 \times 7$) across the image.

$$\mathbf{S}(i, j) = (\mathbf{I} * \mathbf{K})(i, j) = \sum_m \sum_n \mathbf{I}(i - m, j - n) \mathbf{K}(m, n)$$

- **Early Layers**: Learn low-level visual features: edge angles, ink stroke orientation, stroke endpoints.
- **Middle Layers**: Combine edges into curves, loops, line intersections, and stroke crossings.
- **Deep Layers**: Capture signature gestalt: proportional loop scales, slant angles, global baseline consistency.

### 4.2 Residual Connections (ResNet-18) & Solving Vanishing Gradients
In deep standard networks, backpropagating gradients through many stacked layers causes the gradient to exponentially shrink towards zero (the **vanishing gradient problem**):

$$\frac{\partial \mathcal{L}}{\partial W_1} = \frac{\partial \mathcal{L}}{\partial a_L} \prod_{l=2}^{L} \frac{\partial a_l}{\partial a_{l-1}} \frac{\partial a_1}{\partial W_1} \to 0$$

ResNet solves this with **Identity Shortcut Connections**:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

```
           x ───────────────┐ (Identity Shortcut)
           │                │
       [ Conv ]             │
           │                │
        [ ReLU ]            │
           │                │
       [ Conv ]             │
           │                │
           ▼                ▼
         F(x)  ──────► [ + ] ──► ReLU(F(x) + x)
```

During backpropagation:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left( \frac{\partial \mathcal{F}}{\partial \mathbf{x}} + 1 \right)$$

The $+ 1$ term guarantees that gradients can flow backwards unimpeded through the skip connection directly to earlier layers, enabling stable training without degradation.

### 4.3 Transfer Learning & 1-Channel Adaptation
Training an 11-million parameter model from scratch on only 210 images would cause catastrophic overfitting. We initialize our backbone with **ImageNet-pretrained weights**, transferring broad knowledge of edges, textures, and geometry.

#### Converting 3 Channels (RGB) to 1 Channel (Grayscale)
Pretrained ResNet expects 3-channel input ($R, G, B$ with kernel shape `[64, 3, 7, 7]`). Signatures are grayscale ($1$ channel with kernel shape `[64, 1, 7, 7]`).

In `model.py`, we adapt the first convolutional layer by averaging the weights across the RGB channels:

```python
original_conv = backbone.conv1
self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
  self.conv1.weight = nn.Parameter(
      original_conv.weight.mean(dim=1, keepdim=True)
  )
```
This preserves the pretrained spatial edge detectors while accepting 1-channel signature inputs.

### 4.4 Global Average Pooling (GAP)
Traditional CNNs flatten 2D feature maps (`H x W x C`) into a huge 1D vector, requiring millions of weights in the subsequent dense layer.
Instead, ResNet uses **Global Average Pooling (GAP)**:
- Takes the feature map of shape `(Batch, 512, 5, 7)`.
- Averages each $5 \times 7$ slice down to a single number:

$$\text{GAP}(c) = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W A_{c, i, j}$$

- Outputs a vector of shape `(Batch, 512)`.
- **Advantage**: Spatial translation invariance and zero additional parameters, drastically reducing overfitting risk.

### 4.5 Fully Connected (Dense) Projection Head
The 512-dimensional output from the ResNet backbone contains generic visual representations. We append a non-linear projection MLP to distill these into signature-specific biometric features:

$$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{256 \times 512}$$
$$\mathbf{h}_2 = \text{Dropout}(\mathbf{h}_1, p=0.3)$$
$$\mathbf{z} = \mathbf{W}_2 \mathbf{h}_2 + \mathbf{b}_2, \quad \mathbf{W}_2 \in \mathbb{R}^{128 \times 256}$$

### 4.6 Activation Functions (ReLU)
We use the Rectified Linear Unit:

$$\text{ReLU}(z) = \max(0, z)$$

- **Sparsity**: Zeroes out negative activations, creating sparse representations.
- **Computational Efficiency**: Evaluated with a single threshold check without exponential functions.
- **Gradient Retention**: Derivative is $1$ for all positive values, preventing gradient saturation.

### 4.7 Regularization: Dropout & Weight Decay
Given our small dataset, two regularization mechanisms prevent memorization:
1. **Dropout ($p=0.3$)**: During training, 30% of intermediate neurons are randomly set to 0 in each forward pass. This prevents co-adaptation—no single neuron can become a single point of failure.
2. **Weight Decay ($10^{-5}$)**: Adds an $L_2$ penalty to the objective function:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{contrastive}} + \frac{\lambda}{2} \sum_{w} w^2$$

This penalizes large weights and encourages smooth, simpler decision boundaries.

### 4.8 L2 Normalization & Hyper-spherical Embeddings
Every embedding vector $\mathbf{z} \in \mathbb{R}^{128}$ is projected onto the surface of a unit hypersphere ($\|\mathbf{e}\|_2 = 1$):

$$\mathbf{e} = \frac{\mathbf{z}}{\|\mathbf{z}\|_2} = \frac{\mathbf{z}}{\sqrt{\sum_{i=1}^{128} z_i^2}}$$

#### Why Normalization is Essential:
1. **Bounded Distances**: The Euclidean distance between any two unit vectors is bounded strictly between $0$ and $2$:

$$d(\mathbf{e}_1, \mathbf{e}_2)^2 = \|\mathbf{e}_1 - \mathbf{e}_2\|_2^2 = \|\mathbf{e}_1\|^2 + \|\mathbf{e}_2\|^2 - 2 \mathbf{e}_1 \cdot \mathbf{e}_2 = 2 - 2 \cos(\theta)$$

2. **Invariance to Stroke Intensity**: Prevents the network from encoding irrelevant global brightness or ink concentration into vector length.

---

## 5. Loss Function: Contrastive Loss

We train using **Contrastive Loss** (formulated by Yann LeCun et al.):

$$\mathcal{L}(x_1, x_2, y) = y \cdot d^2 + (1 - y) \cdot \max(0, m - d)^2$$

Where:
- $d = \|f_\theta(x_1) - f_\theta(x_2)\|_2$ (Euclidean distance between embeddings)
- $y = 1$ for **Positive Pairs** (genuine signatures from same signer)
- $y = 0$ for **Negative Pairs** (skilled forgery or different signers)
- $m = 1.0$ is the **Margin** parameter

### Mathematical Behavior:
- **Case 1: Same Signer ($y = 1$)**:
  $$\mathcal{L} = d^2$$
  The loss acts like an elastic spring pulling the two points together until $d \to 0$.
- **Case 2: Different Signers or Forgery ($y = 0$)**:
  $$\mathcal{L} = \max(0, m - d)^2$$
  - If $d \ge m$ (already at or beyond the margin), $\mathcal{L} = 0$. The optimizer leaves the pair alone.
  - If $d < m$ (incorrectly close), $\mathcal{L} = (m - d)^2$. The loss pushes the points apart until distance is at least $m$.

```
Positive Pairs (y=1):
  [ e1 ] <========= PULL =========> [ e2 ]   (Target: d -> 0)

Negative Pairs (y=0, d < margin):
  [ e1 ] <--------- PUSH ---------> [ e2 ]   (Target: d >= margin)
```

---

## 6. Pair Generation & Mining Strategy

In `dataset.py`, we construct balanced batches of 1,000 pairs per epoch divided into three categories:

1. **Positive Pairs (33.3%)**: Genuine signature $A$ + Genuine signature $B$ from user $U_i$ ($y = 1.0$).
2. **Skilled Forgery Negative Pairs (33.3%)**: Genuine signature $A$ from user $U_i$ + Forged signature attempting to imitate $U_i$ ($y = 0.0$).
   - *Why this matters*: Teaches the network subtle differences in micro-tremors, line fluency, and loop shapes that a forger cannot perfectly replicate.
3. **Cross-User Negative Pairs (33.3%)**: Genuine signature from user $U_i$ + Genuine signature from user $U_j$ ($y = 0.0$).
   - *Why this matters*: Teaches the network macro-level differences in general handwriting style and signature geometry.

At the start of every epoch, `reshuffle()` dynamically samples a new set of pairs so the network never sees the exact same training pairs twice.

---

## 7. Data Augmentation Pipeline

To prevent 11M parameters from overfitting on ~150 training images, we apply stochastic affine perturbations on the fly during training:

```python
transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((175, 240)),  # Slight upscale
    transforms.RandomCrop((155, 220)),  # Random translation crop
    transforms.RandomAffine(
        degrees=5,  # Slight rotation (-5° to +5°)
        translate=(0.05, 0.05),  # Spatial shift
        scale=(0.9, 1.1),  # Size scaling (90% to 110%)
        shear=3,  # Slant angle variation
    ),
    transforms.RandomPerspective(distortion_scale=0.05, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.01, 0.05)),  # Simulates pen dropouts
])
```

---

## 8. Optimization & Training Dynamics

### 8.1 Adam Optimizer
We optimize parameters using **Adam** (Adaptive Moment Estimation):
- Maintains an exponentially decaying average of past gradients ($m_t$, first moment = momentum).
- Maintains an exponentially decaying average of past squared gradients ($v_t$, second moment = RMSprop adaptive scaling).

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

- Initial learning rate: $\alpha = 10^{-4}$.

### 8.2 Learning Rate Scheduling (ReduceLROnPlateau)
When validation loss stops improving for 5 consecutive epochs (`patience=5`), the scheduler decays the learning rate by a factor of 0.5:

$$\alpha_{\text{new}} = \alpha_{\text{old}} \times 0.5$$

This allows coarse updates early on, followed by fine-grained convergence into sharper local minima.

### 8.3 Early Stopping
Training terminates automatically if validation loss fails to achieve a new best score for 10 consecutive epochs (`patience=10`). In our run, training stopped at epoch 25, retaining the checkpoint from epoch 15 (best validation loss: `0.1501`).

---

## 9. Biometric Evaluation Metrics: Distance, FAR, FRR & EER

In biometric verification, systems make binary decisions based on a distance threshold $\tau$:
- If $d(\mathbf{e}_1, \mathbf{e}_2) < \tau \implies$ **MATCH (Genuine)**
- If $d(\mathbf{e}_1, \mathbf{e}_2) \ge \tau \implies$ **NO MATCH (Forgery / Impostor)**

### Error Rates
1. **False Acceptance Rate (FAR)**: The probability that an impostor/forged signature is mistakenly accepted as genuine:
   $$\text{FAR}(\tau) = \frac{\text{False Positives}}{\text{Total Negative Pairs}} = P(d < \tau \mid y = 0)$$
2. **False Rejection Rate (FRR)**: The probability that a genuine signature is mistakenly rejected:
   $$\text{FRR}(\tau) = \frac{\text{False Negatives}}{\text{Total Positive Pairs}} = P(d \ge \tau \mid y = 1)$$

### Equal Error Rate (EER)
- As threshold $\tau$ increases: FAR increases, FRR decreases.
- As threshold $\tau$ decreases: FAR decreases, FRR increases.
- **EER** is the operating point where $\text{FAR}(\tau) = \text{FRR}(\tau)$. It is the standard single-number benchmark for biometric accuracy.

```
Error Rate
   ▲
1.0│     \  FRR            /  FAR
   │      \               /
   │       \             /
EER│────────\───[●]─────/─────────
   │         \ /   \   /
   │          X     \ /
0.0└─────────/───────X────────────► Threshold τ
                    τ* (Optimal)
```

In our validation on unseen signers (`u07`, `u09`):
- **Optimal Threshold ($\tau^*$)**: `0.4729`
- **EER**: `20.87%`
- **Verification Accuracy at EER**: `79.2%`

---

## 10. PyTorch Model Mechanics: State Dicts & `.pth` Checkpoints

### What is a `state_dict`?
In PyTorch, a model's `state_dict` is a native Python dictionary mapping every layer string name to its learnable parameter tensor:
```python
{
    'conv1.weight': tensor([[[[ ... ]]]]),          # shape: [64, 1, 7, 7]
    'bn1.weight': tensor([ ... ]),                  # shape: [64]
    'bn1.bias': tensor([ ... ]),                    # shape: [64]
    'layer1.0.conv1.weight': tensor([ ... ]),       # shape: [64, 64, 3, 3]
    ...
    'embedding_head.3.weight': tensor([ ... ]),     # shape: [128, 256]
    'embedding_head.3.bias': tensor([ ... ])        # shape: [128]
}
```

### Why Instantiate with `pretrained=False` First?
In `verify.py`:
```python
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
encoder = SignatureEncoder(embedding_dim=128, pretrained=False)  # Step 1
encoder.load_state_dict(checkpoint["encoder_state_dict"])  # Step 2
```
1. **Step 1 (`SignatureEncoder(..., pretrained=False)`)**: Constructs the empty PyTorch computational graph in memory. Memory is allocated for all tensors with random weights. We set `pretrained=False` so PyTorch skips downloading generic ImageNet weights.
2. **Step 2 (`encoder.load_state_dict(...)`)**: Copies our fine-tuned weights directly into the allocated memory addresses of that computational graph.

### What is Saved in `checkpoints/best_encoder.pth`?
```python
torch.save(
    {
        "epoch": 15,
        "encoder_state_dict": model.encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": 0.1501,
        "embedding_dim": 128,
    },
    checkpoint_path,
)
```

---

## 11. Domain Shift & Practical Preprocessing Insights

### The "White Collapse" Problem (Observed During Testing)
When evaluating digital pen signatures (`ref1-grayscale.png`, `sub1-grayscale.png`, `sub2-grayscale.png`), both genuine and forged pairs yielded very low distances (~0.07):

```
dist(ref1, sub1) = 0.077   (Genuine pair)
dist(ref1, sub2) = 0.083   (Impostor pair!)
dist(ref1, blank_white) = 0.304  (< threshold 0.4729!)
```

### Root Causes:
1. **Resolution & Stroke Discrepancy**:
   - Training images are $1080 \times 1729$ paper scans with thick, continuous, bleeding ballpoint ink strokes.
   - Test images were $193 \times 115$ digital sketches drawn with a 1-pixel-wide mouse tool on a large white canvas.
2. **Background Dominance**:
   - When resized to $155 \times 220$, the 1-pixel line was interpolated and washed out. Over **99.2%** of all pixels in the tensor were pure white ($1.0$).
   - Because 99.2% of the tensor was empty white space, any two such images appeared virtually identical to the CNN.

### Standard Production Remediation Pipeline:
For real-world deployment across mixed scan/digital sources:
1. **Otsu Binarization**: Automatically separate ink from background:
   $$\text{threshold } T = \arg\max \sigma_B^2(T)$$
2. **Bounding Box Auto-Crop**: Detect all coordinates where $\text{pixel} < T$, and crop tightly around the signature bounding box:
   $$\text{BBox} = [x_{\min}, y_{\min}, x_{\max}, y_{\max}]$$
3. **Morphological Dilation**: For thin digital pen strokes, dilate with a $3 \times 3$ kernel to restore physical pen stroke thickness:
   $$\mathbf{I}_{\text{dilated}} = \mathbf{I} \oplus \mathbf{K}$$
4. **Contrast Inversion**: Invert colors so background = $0.0$ and ink = $1.0$. Because $0 \times W = 0$, convolution kernels will only activate on stroke pixels, ignoring background proportions.

---

## 12. File-by-File Codebase Tour

| File | Purpose | Key Functions / Classes |
| :--- | :--- | :--- |
| [`model.py`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/model.py) | Defines the neural network and loss function. | `SignatureEncoder`, `ContrastiveLoss`, `SiameseNetwork` |
| [`dataset.py`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/dataset.py) | Custom PyTorch dataset with contrastive pair mining and augmentation. | `SignaturePairDataset`, `load_user_images` |
| [`train.py`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/train.py) | Full training loop, validation, EER computation, and plot generation. | `train_one_epoch`, `validate`, `find_eer`, `plot_loss_curves` |
| [`verify.py`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/verify.py) | Standalone inference script for verifying any two image paths. | `load_encoder`, `preprocess_image`, `verify_signatures` |
| [`main.py`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/main.py) | Unified command-line interface entry point. | Dispatches to `train` or `verify` subcommands |
| [`requirements.txt`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/requirements.txt) | Project dependencies. | PyTorch, torchvision, Pillow, scikit-learn, matplotlib |

---

## 13. CLI Reference & Quick Start

### 1. Verify Two Signatures
```powershell
# Run from c:\Users\iamvi\MyProjects\POC\sigverify\sigverify\
python main.py verify --img1 "imgs\imgs\a.png" --img2 "imgs\imgs\b.png"
```

Sample output:
```text
==================================================
Signature Verification Result
==================================================
Image 1: imgs\imgs\a.png
Image 2: imgs\imgs\b.png
--------------------------------------------------
Euclidean Distance : 0.0758
Cosine Similarity  : 0.9971
Threshold          : 0.4729
Confidence         : 92.0%
--------------------------------------------------
>>> MATCH -- Signatures likely belong to the SAME person
==================================================
```

### 2. Verify with Custom Threshold Override
```powershell
python main.py verify --img1 "path/to/sig1.png" --img2 "path/to/sig2.png" --threshold 0.40
```

### 3. Re-train the Encoder
```powershell
python main.py train --epochs 50 --batch-size 32 --lr 1e-4 --pairs 1000 --margin 1.0
```

Trained checkpoints and visual diagnostics will be output to the [`checkpoints/`](file:///c:/Users/iamvi/MyProjects/POC/sigverify/sigverify/checkpoints/) directory:
- `best_encoder.pth`: Model weights
- `threshold.pth`: Saved optimal threshold and validation EER
- `loss_curve.png`: Epoch-by-epoch loss curve
- `distance_distribution.png`: Genuine vs. forgery separation histogram
