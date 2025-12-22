# MLXSTMBot - Full xLSTM Architecture Implementation

A complete implementation of the xLSTM (Extended Long Short-Term Memory) architecture optimized for Apple Silicon using MLX.

## Architecture Overview

This implementation provides a full xLSTM architecture with the following components:

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

## Key Features

✅ **MLX Optimization**: Fully optimized for Apple Silicon unified memory  
✅ **Proper State Management**: Handles complex state propagation across layers  
✅ **Comprehensive Validation**: Input validation and error handling throughout  
✅ **Numerical Stability**: Epsilon additions and value clamping for stable training  
✅ **Modular Design**: Clean separation between components for easy extension  
✅ **Memory Efficient**: Minimal tensor copies and optimal memory layouts  

## Model Statistics

For a typical configuration (vocab_size=1000, hidden_dim=128, num_layers=4):
- **Total Parameters**: ~1.1M
- **Embedding Parameters**: ~128K
- **LSTM Parameters**: ~589K  
- **Feed-Forward Parameters**: ~262K
- **Language Model Head**: ~128K

## Usage Example

```swift
// Create xLSTM model
let xlstm = try xLSTM(
    vocabSize: 1000,
    hiddenDim: 128,
    numLayers: 4,
    includeFeedForward: true
)

// Single token processing
let tokenIds = MLX.zeros([batchSize], dtype: .int32) + 1
let (logits, states) = try xlstm(tokenIds)

// Sequence processing
let tokenSequence = MLX.zeros([batchSize, sequenceLength], dtype: .int32) + 1
let (sequenceLogits, finalStates) = try xlstm.processSequence(tokenSequence)

// Text generation
let prompt = MLX.zeros([1, promptLength], dtype: .int32) + 1
let generated = try xlstm.generate(
    prompt: prompt,
    maxLength: 50,
    temperature: 1.0,
    topK: 50
)
```

## Architecture Components

### Files Structure
- `sLSTM.swift` - Scalar LSTM implementation with exponential gating
- `mLSTM.swift` - Matrix LSTM implementation with covariance updates  
- `xLSTMBlock.swift` - Block wrapper with normalization and residual connections
- `xLSTM.swift` - Main architecture with embedding and language modeling head
- `RMSNorm.swift` - Root Mean Square normalization layer
- `FeedForward.swift` - Gated Linear Unit and MLP implementations
- `LSTMUtils.swift` - Shared utilities and validation functions

### Mathematical Formulations

#### sLSTM (Exponential Gating)
```
i_t = exp(W_i * [x_t; h_{t-1}] + b_i)  (exponential input gate)
f_t = exp(W_f * [x_t; h_{t-1}] + b_f)  (exponential forget gate)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_c * [x_t; h_{t-1}] + b_c)
n_t = f_t ⊙ n_{t-1} + i_t  (normalizer state)
h_t = o_t ⊙ (c_t / (n_t + ε))  (normalized output)
```

#### mLSTM (Matrix Memory)
```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)  (covariance update)
h_t = o_t ⊙ tanh(C_t @ q_t)  (query-based retrieval)
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Xcode 15.0+
- MLX Swift framework
- Swift 5.9+

## Building and Running

```bash
# Build the project
xcodebuild -project MLXSTMBot.xcodeproj -scheme MLXSTMBot build

# Run the comprehensive test suite
./MLXSTMBot
```

## Test Results

The implementation passes comprehensive tests including:
- Individual LSTM component functionality
- xLSTM block processing with residual connections
- Full architecture text generation capabilities
- State management across multiple layers
- Numerical stability validation
- Error handling and edge cases

## Future Enhancements

- [ ] Advanced sampling strategies (nucleus sampling, beam search)
- [ ] Gradient checkpointing for memory efficiency
- [ ] Multi-head attention integration
- [ ] Training loop implementation
- [ ] Model serialization/deserialization
- [ ] Performance benchmarking suite

## References

- Beck, M., et al. "xLSTM: Extended Long Short-Term Memory" (2024)
- MLX Framework Documentation
- Apple Silicon Optimization Guidelines

---

**Ready for training and text generation on Apple Silicon!** 🚀