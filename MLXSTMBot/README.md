# MLXSTMBot - Full xLSTM Architecture Implementation

A complete implementation of the xLSTM (Extended Long Short-Term Memory) architecture optimized for Apple Silicon using MLX, with integrated knowledge distillation training infrastructure.

## Architecture Overview

This implementation provides a full xLSTM architecture with comprehensive training capabilities:

### Core LSTM Components
- **sLSTM (Scalar LSTM)**: Uses exponential gating and normalizer states for improved stability
- **mLSTM (Matrix LSTM)**: Uses matrix memory with covariance-based updates for enhanced information storage

### xLSTM Block Wrapper
- **Pre-layer RMS Normalization**: Normalizes inputs before LSTM processing
- **Residual Connections**: Implements `x = x + Layer(x)` pattern
- **Feed-Forward Networks**: Gated Linear Units (GLU) for sLSTM blocks
- **Flexible Block Types**: Supports both sLSTM and mLSTM configurations

### Full xLSTM Model
- **Token Embedding**: Maps vocabulary to hidden dimensions
- **Alternating Architecture**: [mLSTM, sLSTM, mLSTM, sLSTM, ...] pattern
- **Language Modeling Head**: Projects back to vocabulary size
- **Autoregressive Generation**: Supports text generation with temperature and top-k sampling

### Training Infrastructure
- **ChatDataProvider**: Loads and tokenizes JSON conversational data
- **DistillationTrainer**: Knowledge distillation from teacher LLM to student xLSTM
- **TrainingMain**: Complete training pipeline with checkpointing and monitoring

## Key Features

✅ **MLX Optimization**: Fully optimized for Apple Silicon unified memory  
✅ **Knowledge Distillation**: Train xLSTM using Llama-3.2-1B-Instruct as teacher  
✅ **Proper State Management**: Handles complex state propagation across layers  
✅ **Comprehensive Validation**: Input validation and error handling throughout  
✅ **Numerical Stability**: Epsilon additions and value clamping for stable training  
✅ **Modular Design**: Clean separation between components for easy extension  
✅ **Memory Efficient**: Minimal tensor copies and optimal memory layouts  
✅ **Training Monitoring**: Loss tracking, checkpointing, and progress reporting  

## Usage

### Training Mode (Knowledge Distillation)
```bash
./MLXSTMBot train
```

This will:
1. Load Llama-3.2-1B-Instruct-4bit as the teacher model
2. Initialize xLSTM student model with matching vocabulary (128,256 tokens)
3. Run knowledge distillation training for 1000 iterations
4. Save the final model to `xLSTM_Final.safetensors`

### Test Mode (Architecture Validation)
```bash
./MLXSTMBot test
```

Runs comprehensive tests of the xLSTM architecture components.

### Interactive Mode (Teacher Model Demo)
```bash
./MLXSTMBot
```

Demonstrates the teacher model inference capabilities.

## Training Configuration

Default training parameters:
- **Teacher Model**: Llama-3.2-1B-Instruct-4bit
- **Student Vocabulary**: 128,256 tokens (matches teacher)
- **Hidden Dimension**: 512
- **Number of Layers**: 6
- **Batch Size**: 4
- **Sequence Length**: 256
- **Learning Rate**: 1e-4
- **Temperature**: 2.0 (for KL divergence)
- **Loss Weights**: 70% distillation, 30% ground truth

## Model Statistics

For the default configuration:
- **Total Parameters**: ~67M
- **Embedding Parameters**: ~65M
- **LSTM Parameters**: ~9M
- **Feed-Forward Parameters**: ~5M
- **Language Model Head**: ~65M

## Training Components

### 1. ChatDataProvider Class
```swift
let dataProvider = try ChatDataProvider(
    jsonPath: "training_data.json",
    tokenizer: tokenizer,
    shuffle: true
)

let (inputs, targets) = try dataProvider.nextBatch(
    batchSize: 4,
    seqLen: 256
)
```

**Features:**
- Loads JSON data in format `[{"content": "..."}]`
- Handles tokenization using MLXLLM tokenizers
- Provides batched data with proper padding/truncation
- Supports shuffling and epoch management

### 2. DistillationTrainer Class
```swift
let trainer = DistillationTrainer(
    studentModel: xlstm,
    teacherModel: llama,
    learningRate: 1e-4,
    temperature: 2.0
)

let loss = try trainer.trainingStep(inputs: inputs, targets: targets)
```

**Features:**
- AdamW optimizer with configurable learning rate
- Combined loss: Cross-entropy + KL divergence
- Teacher model in eval mode (no gradients)
- Proper state management for recurrent models
- Loss monitoring and validation steps

### 3. TrainingMain Class
```swift
let trainingMain = TrainingMain()
await trainingMain.runTraining()
```

**Features:**
- Complete training pipeline
- Model loading and initialization
- Progress monitoring and logging
- Checkpoint saving every 100 steps
- Final model export to SafeTensors format

## Data Format

Training data should be in JSON format:
```json
[
    {"content": "Hello, how are you today?"},
    {"content": "I'm doing well, thank you for asking."},
    {"content": "Can you explain machine learning?"},
    {"content": "Machine learning is a subset of AI..."}
]
```

## File Structure
```
MLXSTMBot/
├── sLSTM.swift              # Scalar LSTM implementation
├── mLSTM.swift              # Matrix LSTM implementation  
├── xLSTMBlock.swift         # Block wrapper with normalization
├── xLSTM.swift              # Main architecture
├── RMSNorm.swift            # RMS normalization layer
├── FeedForward.swift        # GLU and MLP implementations
├── LSTMUtils.swift          # Shared utilities
├── ChatDataProvider.swift   # Data loading and tokenization
├── DistillationTrainer.swift # Knowledge distillation trainer
├── TrainingMain.swift       # Main training pipeline
├── main.swift               # Application entry point
└── README.md               # This file
```

## Mathematical Formulations

### sLSTM (Exponential Gating)
```
i_t = exp(W_i * [x_t; h_{t-1}] + b_i)  (exponential input gate)
f_t = exp(W_f * [x_t; h_{t-1}] + b_f)  (exponential forget gate)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c * [x_t; h_{t-1}] + b_c)
n_t = f_t ⊙ n_{t-1} + i_t  (normalizer state)
h_t = o_t ⊙ (c_t / (n_t + ε))  (normalized output)
```

### mLSTM (Matrix Memory)
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)  (covariance update)
h_t = o_t ⊙ tanh(C_t @ q_t)  (query-based retrieval)
```

### Knowledge Distillation Loss
```
L_total = α * L_CE(student, targets) + β * L_KL(student, teacher)
L_KL = KL(softmax(teacher_logits/T) || softmax(student_logits/T)) * T²
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Xcode 15.0+
- MLX Swift framework
- MLXLLM framework
- Swift 5.9+

## Building and Running

```bash
# Build the project
xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot build

# Run training
./MLXSTMBot train

# Run tests
./MLXSTMBot test

# Run interactive demo
./MLXSTMBot
```

## Training Output

The training process will generate:
- `xLSTM_Final.safetensors` - Final trained model weights
- `training_stats.json` - Training statistics and loss history
- `checkpoint_*.safetensors` - Periodic training checkpoints
- Console logs with progress and loss information

## Performance Monitoring

Training progress is displayed every 10 steps:
```
Step  100 | Loss: 2.3456 | Avg Loss: 2.4123 | Steps/sec: 1.23 | Progress: 10.0%
Step  200 | Loss: 2.1234 | Avg Loss: 2.2456 | Steps/sec: 1.25 | Progress: 20.0%
```

## Future Enhancements

- [ ] Advanced sampling strategies (nucleus sampling, beam search)
- [ ] Gradient checkpointing for memory efficiency
- [ ] Multi-head attention integration
- [ ] Distributed training support
- [ ] Model quantization and compression
- [ ] Advanced tokenization strategies
- [ ] Curriculum learning implementation
- [ ] Evaluation metrics and benchmarking

## References

- Beck, M., et al. "xLSTM: Extended Long Short-Term Memory" (2024)
- Hinton, G., et al. "Distilling the Knowledge in a Neural Network" (2015)
- MLX Framework Documentation
- Apple Silicon Optimization Guidelines

---

**Ready for knowledge distillation training on Apple Silicon!** 🚀