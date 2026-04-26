# MLXSTMBot

A 149M-parameter **xLSTM** language model trained on Apple Silicon via **knowledge distillation** from Llama 3.2-1B-Instruct. Built entirely in Swift using the [MLX](https://github.com/ml-explore/mlx-swift) framework.

```
Teacher  ─────────────────────────────────────────►  Llama 3.2-1B-Instruct (4-bit)
                    KL divergence + Cross-Entropy
Student  ─────────────────────────────────────────►  xLSTM  [mLSTM × 5  +  sLSTM × 1]
```

---

## Contents

- [Overview](#overview)
- [Architecture](#architecture)
  - [sLSTM](#slstm)
  - [mLSTM](#mlstm)
  - [Full model](#full-model)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Generating training data](#generating-training-data)
- [Training](#training)
- [Inference](#inference)
- [Configuration](#configuration)
- [How distillation works](#how-distillation-works)
- [Model statistics](#model-statistics)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

MLXSTMBot implements the **xLSTM architecture** (Beck et al., 2024) natively in Swift and trains it using **knowledge distillation**:

1. A frozen Llama 3.2-1B teacher generates soft probability distributions over the vocabulary.
2. The xLSTM student learns to match those distributions (KL divergence) alongside predicting the correct next tokens (cross-entropy).
3. The result is a compact recurrent model that runs efficiently on Apple Silicon without a KV-cache or quadratic attention complexity.

Unlike transformer-based models, xLSTM processes tokens sequentially with a fixed-size state, making inference memory cost **O(1)** in sequence length rather than **O(n)**.

---

## Architecture

### sLSTM

Scalar LSTM with **exponential gating** and a numerically stabilised normalizer state. Each gate operates in log-space to prevent overflow, with a running maximum `m_t` that keeps all exponents non-positive.

```
m_t  =  max(z_f + m_{t-1},  z_i)
i′_t =  exp(z_i - m_t)              # input gate,  always in (0, 1]
f′_t =  exp(z_f + m_{t-1} - m_t)   # forget gate, always in (0, 1]
n_t  =  f′_t · n_{t-1} + i′_t      # normalizer state, init = 1.0
c_t  =  f′_t · c_{t-1} + i′_t · tanh(W_c · [x; h])
h_t  =  σ(W_o · [x; h]) · (c_t / max(n_t, ε))
```

The sLSTM block also includes a **SwiGLU feed-forward network** (GELU-gated linear unit) and post-norm with residual connection.

### mLSTM

Matrix LSTM with an outer-product **matrix memory** `C_t ∈ ℝ^{d×d}`. Instead of a scalar cell state, the model maintains a full matrix that can store richer associative information.

```
q_t  =  W_q · conv_causal(x)        # query from left-padded depthwise conv
k_t  =  W_k · x / √d                # key, length-normalised
v_t  =  W_v · x                     # value
C_t  =  f_t · C_{t-1} + i_t · (v_t ⊗ k_t^T)   # matrix memory update
n_t  =  f_t · n_{t-1} + i_t · k_t
denom     =  max(|n_t^T q_t|, 1)
retrieved =  GroupNorm(C_t q_t / denom)
h_t  =  σ(W_o · x) · GELU(W_skip · x) · retrieved
```

The causal convolution on the query path uses **left-only padding** (`kernelSize - 1` zeros on the left, `padding: 0` on the convolution) to ensure no future context leaks into past positions.

### Full model

```
Token IDs  ──►  Embedding (128 256 × 512)
               │
               ├──►  xLSTMBlock [mLSTM]  ──►  preNorm + residual
               ├──►  xLSTMBlock [mLSTM]  ──►  preNorm + residual
               ├──►  xLSTMBlock [mLSTM]  ──►  preNorm + residual
               ├──►  xLSTMBlock [mLSTM]  ──►  preNorm + residual
               ├──►  xLSTMBlock [mLSTM]  ──►  preNorm + residual
               └──►  xLSTMBlock [sLSTM]  ──►  preNorm + SwiGLU FFN + residual
               │
               LayerNorm (512)
               │
               LM Head (512 → 128 256)
               │
              Logits
```

Every block applies **pre-norm** (LayerNorm before the recurrent layer) and a **residual connection** around the LSTM output. The mLSTM blocks use an up-projection (`hiddenDim → hiddenDim × 2`) that is split into a recurrent path and a gating path, with a down-projection combining both back to `hiddenDim`.

---

## Project structure

```
MLXSTMBot/
├── Sources/
│   └── MLXSTMBot/
│       ├── main.swift               # Entry point — mode dispatch
│       ├── xLSTM.swift              # Top-level model
│       ├── xLSTMBlock.swift         # Block wrapper, LayerState enum
│       ├── sLSTM.swift              # Scalar LSTM implementation
│       ├── mLSTM.swift              # Matrix LSTM implementation
│       ├── FeedForward.swift        # SwiGLUFeedForward, MLPFeedForward
│       ├── LSTMUtils.swift          # Validation helpers, numerical utils
│       ├── ChatDataProvider.swift   # JSON data loader + batch generator
│       ├── DistillationTrainer.swift# Training loop, KL + CE loss, AdamW
│       └── TrainingMain.swift       # Pipeline orchestration
├── generate_training_data.py        # Training data generator (Python)
├── build.sh                         # Build script
├── train.sh                         # Training script
├── training_data.json               # Generated training data (git-ignored)
├── xLSTM_Final.safetensors          # Saved model weights (git-ignored)
└── Package.swift                    # Swift Package Manager manifest
```

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Mac hardware | Apple Silicon (M1) | M2 Pro / M3 / M4 |
| macOS | 14.0 Sonoma | 15.x Sequoia |
| Xcode | 15.0 | 16.x |
| Swift | 5.9 | 5.10 |
| RAM | 8 GB | 16 GB |
| Free disk space | 4 GB | 8 GB |
| Python (data gen only) | 3.9 | 3.12 |

The teacher model (`mlx-community/Llama-3.2-1B-Instruct-4bit`) is approximately **800 MB** and is downloaded automatically by `MLXLMCommon` on first run.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MLXSTMBot.git
cd MLXSTMBot
```

### 2. Resolve Swift packages

```bash
swift package resolve
```

The project depends on:

- [`mlx-swift`](https://github.com/ml-explore/mlx-swift) — core tensor library for Apple Silicon
- [`mlx-swift-examples`](https://github.com/ml-explore/mlx-swift-examples) — provides `MLXLLM`, `MLXLMCommon`
- [`swift-transformers`](https://github.com/huggingface/swift-transformers) — tokenizer support

### 3. Build

```bash
./build.sh
```

A successful build prints:

```
=== Building MLXSTMBot ===
Build complete!
=== Build succeeded ===
```

---

## Generating training data

The included Python script pulls from four public HuggingFace datasets and produces a `training_data.json` file in the format expected by `ChatDataProvider`.

### Install Python dependencies

```bash
pip install datasets tqdm
```

### Generate data

```bash
# Quick smoke-test (≈ 2 min)
python generate_training_data.py --samples 2000

# Recommended first real run (≈ 10 min)
python generate_training_data.py --samples 10000

# Full production run (≈ 45 min)
python generate_training_data.py --samples 50000
```

**Data sources and allocation:**

| Source | Share | Character |
|---|---|---|
| Wikipedia (20220301.en) | 35% | Factual, encyclopedic prose |
| OpenWebText (Skylion007) | 35% | Diverse web language |
| BookCorpus | 20% | Long-form narrative |
| Anthropic HH-RLHF (helpful) | 10% | Conversational / instruction phrasing |

The script applies a quality gate (length bounds, punctuation check, ASCII ratio), exact deduplication, and a final shuffle to interleave sources before writing.

**Key options:**

```
--samples   N     Number of samples to generate (default: 5000)
--output    PATH  Output file path (default: training_data.json)
--seed      N     Random seed for reproducibility (default: 42)
--min-chars N     Minimum characters per sample (default: 80)
--max-chars N     Maximum characters per sample (default: 2048)
```

The `--max-chars` default of 2048 corresponds to roughly 512 tokens, which comfortably covers the `sequenceLength = 256` training window in `TrainingConfig`.

---

## Training

### Run training

```bash
./train.sh
```

This builds the project (if needed) and launches the full distillation pipeline. You will see output like:

```
🚀 MLXSTMBot - xLSTM Knowledge Distillation Training
============================================================
📚 Initializing Models...
✓ Student xLSTM model initialized
  - Vocabulary size: 128256
  - Hidden dimension: 512
  - Number of layers: 6
Loading teacher model: mlx-community/Llama-3.2-1B-Instruct-4bit
✓ Teacher model loaded successfully
Teacher model frozen — parameters excluded from gradient graph
🏃 Starting Training Loop...
Step    1 | Loss: 4.40 | Steps/sec: 0.02 | Progress: 0.1%
Step    2 | Loss: 4.31 | Steps/sec: 0.35 | Progress: 0.2%
...
```

Step 1 is slow due to MLX graph compilation. From step 2 onward, speed stabilises.

**What a healthy loss curve looks like:**

| Steps | Expected loss range |
|---|---|
| 1 | 4.2 – 4.6 |
| 50 | 3.5 – 4.0 |
| 200 | 3.0 – 3.5 |
| 500 | 2.5 – 3.2 |
| 1000 | 2.2 – 3.0 |

If the loss does not decline after 50 steps, check that your `training_data.json` has at least 500 samples and is not the 10-sample placeholder generated by `createSampleData`.

### Architecture test mode

Validates all components without loading the teacher model. Useful after code changes:

```bash
swift run MLXSTMBot test
```

### Interactive mode

Loads the Llama teacher and demonstrates inference directly:

```bash
swift run MLXSTMBot
```

### Saved artefacts

After training completes:

| File | Contents |
|---|---|
| `xLSTM_Final.safetensors` | Final student model weights |
| `training_stats.json` | Loss history and training metadata |
| `checkpoint_N.safetensors` | Checkpoints saved every 10 steps |

---

## Inference

Load saved weights and run the student model standalone:

```swift
import MLX
import MLXNN

// Load model
let model = try xLSTM(
    vocabSize: 128256,
    hiddenDim: 512,
    blockSpec: [.mLSTM, .mLSTM, .mLSTM, .mLSTM, .mLSTM, .sLSTM]
)
let weights = try MLX.loadArrays(url: URL(fileURLWithPath: "xLSTM_Final.safetensors"))
try model.update(parameters: ModuleParameters.unflattened(weights))

// Autoregressive generation — O(1) memory per step
let promptTokens = MLXArray([1, 450, 3817], dtype: .int32)  // your tokenised prompt
let generated = model.generateSingle(
    promptTokens: promptTokens,
    maxLength: 100,
    temperature: 0.8,
    topK: 50
)
// Decode generated token IDs with your tokenizer
```

**State-based single-token loop** (for streaming / interactive use):

```swift
var states = model.initialStates()
let tokenId = MLXArray([Int32(startToken)])

for _ in 0..<maxSteps {
    let (logits, newStates) = try model(tokenId, states: states)
    let nextToken = MLXRandom.categorical(logits / temperature)
    states = newStates
    // stream nextToken to output
}
```

---

## Configuration

All training hyperparameters live in the `TrainingConfig` struct inside `TrainingMain.swift`:

```swift
struct TrainingConfig {
    var teacherModelId  = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    var vocabSize       = 128256   // must match teacher tokenizer
    var hiddenDim       = 512
    var numLayers       = 6
    var batchSize       = 4
    var sequenceLength  = 256
    var numIterations   = 1000
    var learningRate    : Float = 1e-4
    var dataPath        = "training_data.json"
    var outputPath      = "xLSTM_Final.safetensors"
}
```

**Distillation hyperparameters** in `DistillationTrainer.init`:

```swift
temperature        = 2.0   // softens the teacher's probability distribution
distillationWeight = 0.7   // weight on KL divergence loss
groundTruthWeight  = 0.3   // weight on cross-entropy loss
```

**Tuning guidance:**

| Goal | Adjustment |
|---|---|
| Faster training on M1 (8 GB RAM) | `batchSize = 2`, `sequenceLength = 128` |
| Better convergence | `learningRate = 3e-4` for first 200 steps, then `1e-4` |
| More teacher signal | `distillationWeight = 0.9`, `groundTruthWeight = 0.1` |
| Sharper soft targets | `temperature = 1.5` (lower = sharper) |
| Larger model | `hiddenDim = 768`, add more mLSTM blocks to `blockSpec` |

---

## How distillation works

```
Input tokens
     │
     ├──► Teacher (Llama, frozen) ──► soft logits P_T  ─────────────────┐
     │                                                                    │
     └──► Student (xLSTM)         ──► soft logits P_S  ──► KL(P_T ‖ P_S)┤
                                  └──► hard logits     ──► CE(y, P_S)   ┤
                                                                         │
                                  Total loss = 0.3 · CE + 0.7 · KL ◄───┘
                                       │
                                  AdamW, lr=1e-4, grad clip=1.0
                                       │
                                  Student weights updated
```

The teacher's soft probability distribution over all 128,256 tokens carries richer signal than a one-hot label — it encodes which tokens are semantically similar and which are plausible alternatives. The student learns this "dark knowledge" through the KL term, while the CE term anchors it to the correct ground-truth token.

The teacher's parameters are **frozen** (`module.freeze(recursive: true)`) and never receive gradients. Only the student's 149M parameters are updated.

---

## Model statistics

| Component | Parameters |
|---|---|
| Token embedding | 65.7M |
| mLSTM blocks × 5 | 14.4M |
| sLSTM block × 1 | 3.1M |
| LM head | 65.7M |
| **Total** | **~149M** |

| Metric | Value |
|---|---|
| Memory (fp16, inference) | ~0.30 GB |
| Memory (fp32, training) | ~0.60 GB |
| Vocabulary size | 128,256 (Llama 3 tokenizer) |
| Hidden dimension | 512 |
| Matrix memory size (mLSTM) | 512 × 512 = 262,144 elements |
| Inference complexity | O(1) in sequence length |
| Attention complexity | None — fully recurrent |

---

## Troubleshooting

**Loss is NaN from step 1**
The `n_t` normalizer in `sLSTM` must be initialised to `1.0`, not `0.0`. Check `sLSTM.initialState(batchSize:)`.

**Build fails with "module not found: MLXLLM"**
Run `swift package resolve` first. Xcode may also need a Clean Build Folder (`⇧⌘K`) to pick up new package versions.

**Teacher model download hangs**
`MLXLMCommon` downloads from HuggingFace Hub. If your network blocks it, set:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Steps/sec stays below 0.05 after step 5**
The sequential timestep loop in `sLSTM`/`mLSTM` (`processSequence`) is the bottleneck. Reduce `sequenceLength` to 128 or `batchSize` to 2 as a short-term fix. A scan-based parallel implementation would resolve this permanently.

**`training_data.json` not found**
Run the data generator first:
```bash
python generate_training_data.py --samples 5000
```

**Checkpoint files filling up disk**
Checkpoints are saved every 10 steps by default. Increase `saveEvery` in `TrainingConfig` or delete old checkpoints:
```bash
rm checkpoint_*.safetensors
```

---

## References

```
Beck, M., Pöppel, K., Spanring, M., Auer, A., Prudnikova, O., Kopp, M.,
Klambauer, G., Brandstetter, J., & Hochreiter, S. (2024).
xLSTM: Extended Long Short-Term Memory.
arXiv:2405.04517. https://arxiv.org/abs/2405.04517

Hinton, G., Vinyals, O., & Dean, J. (2015).
Distilling the Knowledge in a Neural Network.
arXiv:1503.02531. https://arxiv.org/abs/1503.02531

Apple MLX Team (2024).
MLX: An array framework for Apple Silicon.
https://github.com/ml-explore/mlx-swift
```
