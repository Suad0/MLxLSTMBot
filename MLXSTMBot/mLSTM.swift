//
//  mLSTM.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Matrix Long Short-Term Memory (mLSTM) implementation with matrix memory
/// 
/// Mathematical formulation (Beck et al. 2024):
/// - q_t = W_q Conv1d_causal(x_l)
/// - k_t = W_k x_l / sqrt(d)
/// - v_t = W_v x_l
/// - z_i = W_i x_l + b_i
/// - z_f = W_f x_l + b_f
/// - m_t = max(z_f + m_{t-1}, z_i)
/// - i_t = exp(z_i - m_t)
/// - f_t = exp(z_f + m_{t-1} - m_t)
/// - n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t
/// - C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
/// - denom = max(|n_t^T q_t|, 1)
/// - retrieved = GroupNorm(C_t q_t / denom)
/// - o_t = sigmoid(W_o x_l) ⊙ GeLU(W_skip x_l)
/// - h_t = o_t ⊙ retrieved
public class mLSTM: Module {
    
    // MARK: - Properties
    
    public let inputDim: Int
    public let hiddenDim: Int
    
    public let inputProjection: Linear
    public let forgetProjection: Linear
    public let outputProjection: Linear
    public let skipProjection: Linear
    public let keyProjection: Linear
    public let valueProjection: Linear
    public let queryProjection: Linear
    public let groupNorm: GroupNorm
    public let conv1dCausal: Conv1d // Depthwise causal conv
    
    // MARK: - Initialization
    
    public init(inputDim: Int, hiddenDim: Int) throws {
        try LSTMUtils.validateDimensions(inputDim: inputDim, hiddenDim: hiddenDim)
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        
        // Linear mapping from inputDim (e.g. x_l) to hiddenDim
        self.inputProjection = Linear(inputDim, hiddenDim)
        self.forgetProjection = Linear(inputDim, hiddenDim)
        self.outputProjection = Linear(inputDim, hiddenDim)
        self.skipProjection = Linear(inputDim, hiddenDim)
        self.keyProjection = Linear(inputDim, hiddenDim)
        self.valueProjection = Linear(inputDim, hiddenDim)
        self.queryProjection = Linear(inputDim, hiddenDim)
        
        // GroupNorm for retrieved projection
        self.groupNorm = GroupNorm(groupCount: hiddenDim / 64 > 0 ? hiddenDim / 64 : 1, dimensions: hiddenDim)
        
        // Depthwise causal conv
        self.conv1dCausal = Conv1d(inputChannels: inputDim, outputChannels: inputDim, kernelSize: 4, stride: 1, padding: 3, groups: inputDim)
        
        super.init()
    }
    
    // MARK: - State Management
    
    public func initialState(batchSize: Int) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        try LSTMUtils.validateBatchSize(batchSize)
        
        let hiddenStateShape = [batchSize, hiddenDim]
        
        // State tuple: (h, C, n, m) — 4 tensors
        let h_t = LSTMUtils.createTensor(shape: hiddenStateShape, value: 0.0)
        let C_t = LSTMUtils.createTensor(shape: [batchSize, hiddenDim, hiddenDim], value: 0.0) // Bug B4 fix
        let n_t = LSTMUtils.createTensor(shape: hiddenStateShape, value: 0.0) // Bug B2 fix
        let m_t = LSTMUtils.createTensor(shape: hiddenStateShape, value: 0.0)
        
        return (h_t, C_t, n_t, m_t)
    }
    
    // MARK: - Forward Pass
    
    public func callAsFunction(_ input: MLXArray, state: (MLXArray, MLXArray, MLXArray, MLXArray)) throws -> (MLXArray, (MLXArray, MLXArray, MLXArray, MLXArray)) {
        let (h_prev, C_prev, n_prev, m_prev) = state
        
        // Comprehensive input validation
        try LSTMUtils.validateInputTensor(input)
        
        guard input.ndim == 2 else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "2D tensor [batch_size, input_dim]",
                actual: "\(input.ndim)D tensor with shape \(input.shape)"
            )
        }
        
        let batchSize = input.shape[0]
        let inputDimActual = input.shape[1]
        
        guard inputDimActual == inputDim else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "input_dim = \(inputDim)",
                actual: "input_dim = \(inputDimActual)"
            )
        }
        
        // Validate state tensor shapes
        try LSTMUtils.validateStateTensor(h_prev, expectedShape: [batchSize, hiddenDim])
        try LSTMUtils.validateMatrixMemory(C_prev, batchSize: batchSize, hiddenDim: hiddenDim)
        try LSTMUtils.validateSameDevice([input, h_prev, C_prev, n_prev, m_prev])
        
        // Causal conv on a single timestep: padding with 0s or keeping state is tricky in callAsFunction. 
        // For accurate single step auto-regressive generation, tracking a conv state buffer is ideal, 
        // but since we only have single x_t here, we simulate it or treat it as a point-wise application.
        // We'll apply it directly on the expanded input.
        let x_expanded = input.reshaped([batchSize, 1, inputDim])
        var convOut = conv1dCausal(x_expanded)
        convOut = convOut[0..., 0..<1, 0...] // Truncate padded parts to sequence length 1
        let x_conv = convOut.squeezed(axis: 1)
        
        // Query, Key, Value
        let q_t = queryProjection(x_conv)
        let k_t = keyProjection(input) / MLX.sqrt(MLXArray(Float(hiddenDim)))
        let v_t = valueProjection(input)
        
        // Gates
        let z_i = inputProjection(input)
        let z_f = forgetProjection(input)
        
        // m_t = max(z_f + m_{t-1}, z_i)
        let m_t = MLX.maximum(z_f + m_prev, z_i)
        
        // i_t = exp(z_i - m_t)
        // f_t = exp(z_f + m_{t-1} - m_t)
        let i_t = MLX.exp(z_i - m_t)
        let f_t = MLX.exp(z_f + m_prev - m_t)
        
        // n_t = f_t ⊙ n_{t-1} + i_t ⊙ k_t
        let n_t = f_t * n_prev + i_t * k_t
        
        // C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
        let v_expanded = v_t.expandedDimensions(axis: -1) // [batch, d, 1]
        let k_expanded = k_t.expandedDimensions(axis: -2) // [batch, 1, d]
        let outer_product = MLX.matmul(v_expanded, k_expanded) // [batch, d, d]
        let f_expanded = f_t.expandedDimensions(axis: -1)
        let i_expanded = i_t.expandedDimensions(axis: -1)
        let C_t = f_expanded * C_prev + i_expanded * outer_product
        
        // Retrieval denom = max(|n_t^T q_t|, 1)
        let n_expanded = n_t.expandedDimensions(axis: -2) // [batch, 1, d]
        let q_expanded = q_t.expandedDimensions(axis: -1) // [batch, d, 1]
        let n_dot_q = MLX.matmul(n_expanded, q_expanded).squeezed(axis: -1) // [batch, 1]
        let denom = MLX.maximum(MLX.abs(n_dot_q), MLXArray(1.0)) // Bug B3
        
        // retrieved = GroupNorm(C_t q_t / denom)
        let C_q = MLX.matmul(C_t, q_expanded).squeezed(axis: -1) // [batch, d]
        let retrieved_raw = C_q / denom
        let retrieved = groupNorm(retrieved_raw)
        
        // o_t = sigmoid(W_o x_l) ⊙ GeLU(W_skip x_l)
        let skip_gate = MLXNN.gelu(skipProjection(input))
        let o_t = MLX.sigmoid(outputProjection(input)) * skip_gate
        
        // h_t = o_t ⊙ retrieved
        let h_t = o_t * retrieved
        
        let newState = (h_t, C_t, n_t, m_t)
        return (h_t, newState)
    }
}

// MARK: - Convenience Extensions

extension mLSTM {
    
    public func initialState() -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        do {
            return try initialState(batchSize: 1)
        } catch {
            fatalError("Failed to create initial state: \(error)")
        }
    }
    
    public func processSequence(_ sequence: MLXArray, initialState: (MLXArray, MLXArray, MLXArray, MLXArray)? = nil) throws -> (MLXArray, (MLXArray, MLXArray, MLXArray, MLXArray)) {
        try LSTMUtils.validateSequenceTensor(sequence)
        try LSTMUtils.validateInputTensor(sequence)
        
        let batchSize = sequence.shape[0]
        let sequenceLength = sequence.shape[1]
        let inputDimActual = sequence.shape[2]
        
        try LSTMUtils.validateSequenceLength(sequenceLength)
        
        guard inputDimActual == inputDim else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "input_dim = \(inputDim)",
                actual: "input_dim = \(inputDimActual)"
            )
        }
        
        var currentState: (MLXArray, MLXArray, MLXArray, MLXArray)
        if let initialState = initialState {
            currentState = initialState
        } else {
            currentState = try self.initialState(batchSize: batchSize)
        }
        
        if sequenceLength == 0 {
            let emptyOutputs = LSTMUtils.createTensor(shape: [batchSize, 0, hiddenDim])
            return (emptyOutputs, currentState)
        }
        
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(sequenceLength)
        
        // Process each timestep sequentially
        // For performance, sequences could be processed with graph trace, but loop fits autograd memory dynamically
        for t in 0..<sequenceLength {
            let timestepInput = sequence[0..., t, 0...] // [batch_size, input_dim]
            let (output, newState) = try callAsFunction(timestepInput, state: currentState)
            outputs.append(output)
            currentState = newState
        }
        
        let allOutputs = stacked(outputs, axis: 1) // [batch_size, sequence_length, hidden_dim]
        return (allOutputs, currentState)
    }
}
