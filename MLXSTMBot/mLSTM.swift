//
//  mLSTM.swift
//  MLXSTMBot
//
//  Created by Kiro on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Matrix Long Short-Term Memory (mLSTM) implementation with matrix memory
/// 
/// This class implements the mLSTM variant that uses matrix memory structures
/// with covariance-based updates for enhanced information storage and retrieval.
/// It inherits from MLXNN.Module to integrate seamlessly with the MLX framework.
///
/// Mathematical formulation:
/// - i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)  (input gate)
/// - f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)  (forget gate)
/// - o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)  (output gate)
/// - k_t = W_k * x_t + U_k * h_{t-1} + b_k  (key vector)
/// - v_t = W_v * x_t + U_v * h_{t-1} + b_v  (value vector)
/// - q_t = W_q * x_t + U_q * h_{t-1} + b_q  (query vector)
/// - C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)  (covariance update)
/// - h_t = o_t ⊙ tanh(C_t @ q_t)  (query-based retrieval)
public class mLSTM: Module {
    
    // MARK: - Properties
    
    /// Input dimension
    public let inputDim: Int
    
    /// Hidden state dimension
    public let hiddenDim: Int
    
    /// Linear projection for input gate computation
    public let inputProjection: Linear
    
    /// Linear projection for forget gate computation
    public let forgetProjection: Linear
    
    /// Linear projection for output gate computation
    public let outputProjection: Linear
    
    /// Linear projection for key vector generation
    public let keyProjection: Linear
    
    /// Linear projection for value vector generation
    public let valueProjection: Linear
    
    /// Linear projection for query vector generation
    public let queryProjection: Linear
    
    // MARK: - Initialization
    
    /// Initializes the mLSTM module with specified dimensions
    /// - Parameters:
    ///   - inputDim: Dimension of input features
    ///   - hiddenDim: Dimension of hidden state
    /// - Throws: LSTMError.invalidDimension if dimensions are invalid
    public init(inputDim: Int, hiddenDim: Int) throws {
        // Validate dimensions
        try LSTMUtils.validateDimensions(inputDim: inputDim, hiddenDim: hiddenDim)
        
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        
        // Initialize linear projection layers
        // Each projection maps from (input + hidden) dimensions to hidden dimensions
        let totalInputDim = inputDim + hiddenDim
        
        self.inputProjection = Linear(totalInputDim, hiddenDim)
        self.forgetProjection = Linear(totalInputDim, hiddenDim)
        self.outputProjection = Linear(totalInputDim, hiddenDim)
        self.keyProjection = Linear(totalInputDim, hiddenDim)
        self.valueProjection = Linear(totalInputDim, hiddenDim)
        self.queryProjection = Linear(totalInputDim, hiddenDim)
        
        super.init()
    }
    
    // MARK: - State Management
    
    /// Creates initial state tensors for the mLSTM
    /// 
    /// Returns a tuple (h_t, C_t) where:
    /// - h_t: hidden state initialized to zeros [batch_size, hidden_dim]
    /// - C_t: matrix memory initialized to identity matrix [batch_size, hidden_dim, hidden_dim]
    ///
    /// - Parameter batchSize: Size of the batch dimension
    /// - Returns: Tuple of initial state tensors (h_t, C_t)
    /// - Throws: LSTMError.invalidDimension if batch size is invalid
    public func initialState(batchSize: Int) throws -> (MLXArray, MLXArray) {
        // Validate batch size
        try LSTMUtils.validateBatchSize(batchSize)
        
        let hiddenStateShape = [batchSize, hiddenDim]
        
        // Initialize hidden state with zeros
        let h_t = LSTMUtils.createTensor(shape: hiddenStateShape, value: 0.0)
        
        // Initialize matrix memory with identity matrix
        let C_t = LSTMUtils.createIdentityMatrix(batchSize: batchSize, dim: hiddenDim)
        
        return (h_t, C_t)
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass computation for mLSTM
    ///
    /// Implements the mLSTM forward pass with:
    /// - Standard sigmoid gating for input, forget, and output gates
    /// - Covariance update rule: C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
    /// - Query-based retrieval: h_t = o_t ⊙ tanh(C_t @ q_t)
    ///
    /// - Parameters:
    ///   - input: Input tensor of shape [batch_size, input_dim]
    ///   - state: Tuple of (h_t, C_t) where h_t is [batch_size, hidden_dim] and C_t is [batch_size, hidden_dim, hidden_dim]
    /// - Returns: Tuple of (output, new_state) where output is [batch_size, hidden_dim] and new_state is (h_t, C_t)
    public func callAsFunction(_ input: MLXArray, state: (MLXArray, MLXArray)) -> (MLXArray, (MLXArray, MLXArray)) {
        let (h_prev, C_prev) = state
        
        do {
            // Comprehensive input validation
            try LSTMUtils.validateInputTensor(input)
            
            // Validate input tensor shape (should be 2D for single timestep)
            guard input.ndim == 2 else {
                throw LSTMUtils.LSTMError.invalidTensorShape(
                    expected: "2D tensor [batch_size, input_dim]",
                    actual: "\(input.ndim)D tensor with shape \(input.shape)"
                )
            }
            
            let batchSize = input.shape[0]
            let inputDimActual = input.shape[1]
            
            // Validate input dimension matches expected
            guard inputDimActual == inputDim else {
                throw LSTMUtils.LSTMError.invalidTensorShape(
                    expected: "input_dim = \(inputDim)",
                    actual: "input_dim = \(inputDimActual)"
                )
            }
            
            // Validate state tensor shapes
            let expectedHiddenShape = [batchSize, hiddenDim]
            try LSTMUtils.validateStateTensor(h_prev, expectedShape: expectedHiddenShape)
            try LSTMUtils.validateMatrixMemory(C_prev, batchSize: batchSize, hiddenDim: hiddenDim)
            
            // Validate device consistency
            try LSTMUtils.validateSameDevice([input, h_prev, C_prev])
            
            // Validate state stability
            try LSTMUtils.validateStateStability([h_prev, C_prev])
            
            // Concatenate input and previous hidden state
            let combined = MLX.concatenated([input, h_prev], axis: -1)
            
            // Compute gates using standard sigmoid activation
            // i_t = sigmoid(W_i * x_t + U_i * h_{t-1} + b_i)
            let i_t = MLX.sigmoid(inputProjection(combined))
            
            // f_t = sigmoid(W_f * x_t + U_f * h_{t-1} + b_f)
            let f_t = MLX.sigmoid(forgetProjection(combined))
            
            // o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)
            let o_t = MLX.sigmoid(outputProjection(combined))
            
            // Compute key, value, and query vectors
            // k_t = W_k * x_t + U_k * h_{t-1} + b_k
            let k_t = keyProjection(combined)
            
            // v_t = W_v * x_t + U_v * h_{t-1} + b_v
            let v_t = valueProjection(combined)
            
            // q_t = W_q * x_t + U_q * h_{t-1} + b_q
            let q_t = queryProjection(combined)
            
            // Compute outer product: v_t ⊗ k_t^T
            // v_t shape: [batch_size, hidden_dim]
            // k_t shape: [batch_size, hidden_dim]
            // Result shape: [batch_size, hidden_dim, hidden_dim]
            let v_expanded = v_t.expandedDimensions(axis: -1)  // [batch_size, hidden_dim, 1]
            let k_expanded = k_t.expandedDimensions(axis: -2)  // [batch_size, 1, hidden_dim]
            let outer_product = MLX.matmul(v_expanded, k_expanded)  // [batch_size, hidden_dim, hidden_dim]
            
            // Expand gates for broadcasting with matrix memory
            // f_t and i_t shape: [batch_size, hidden_dim]
            // Need to expand to [batch_size, hidden_dim, 1] for proper broadcasting
            let f_expanded = f_t.expandedDimensions(axis: -1)  // [batch_size, hidden_dim, 1]
            let i_expanded = i_t.expandedDimensions(axis: -1)  // [batch_size, hidden_dim, 1]
            
            // Covariance update rule: C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v_t ⊗ k_t^T)
            let C_t = f_expanded * C_prev + i_expanded * outer_product
            
            // Query-based retrieval: C_t @ q_t
            // C_t shape: [batch_size, hidden_dim, hidden_dim]
            // q_t shape: [batch_size, hidden_dim]
            // Need to expand q_t to [batch_size, hidden_dim, 1] for matmul
            let q_expanded = q_t.expandedDimensions(axis: -1)  // [batch_size, hidden_dim, 1]
            let retrieval = MLX.matmul(C_t, q_expanded)  // [batch_size, hidden_dim, 1]
            let retrieval_squeezed = retrieval.squeezed(axis: -1)  // [batch_size, hidden_dim]
            
            // Final hidden state: h_t = o_t ⊙ tanh(C_t @ q_t)
            let h_t = o_t * MLX.tanh(retrieval_squeezed)
            
            // Validate output stability before returning
            let newState = (h_t, C_t)
            try LSTMUtils.validateStateStability([h_t, C_t])
            
            return (h_t, newState)
            
        } catch let error as LSTMUtils.LSTMError {
            // Re-throw LSTM-specific errors
            fatalError("mLSTM forward pass error: \(error.localizedDescription)")
        } catch {
            // Handle unexpected errors
            fatalError("Unexpected error in mLSTM forward pass: \(error)")
        }
    }
}

// MARK: - Convenience Extensions

extension mLSTM {
    
    /// Creates initial state with default batch size of 1
    /// - Returns: Initial state tuple for single sample
    public func initialState() -> (MLXArray, MLXArray) {
        do {
            return try initialState(batchSize: 1)
        } catch {
            fatalError("Failed to create initial state: \(error)")
        }
    }
    
    /// Process a full sequence of inputs
    /// 
    /// This convenience method processes a 3D input tensor representing a sequence
    /// by iterating through timesteps and applying the forward pass.
    ///
    /// - Parameters:
    ///   - sequence: Input sequence [batch_size, sequence_length, input_dim]
    ///   - initialState: Optional initial state. If nil, uses default initial state
    /// - Returns: Tuple of (outputs, final_state) where outputs contains all hidden states
    ///            and final_state is the last state tuple
    /// - Throws: LSTMError for invalid inputs
    public func processSequence(_ sequence: MLXArray, initialState: (MLXArray, MLXArray)? = nil) throws -> (MLXArray, (MLXArray, MLXArray)) {
        // Validate sequence tensor
        try LSTMUtils.validateSequenceTensor(sequence)
        try LSTMUtils.validateInputTensor(sequence)
        
        let batchSize = sequence.shape[0]
        let sequenceLength = sequence.shape[1]
        let inputDimActual = sequence.shape[2]
        
        // Validate sequence length
        try LSTMUtils.validateSequenceLength(sequenceLength)
        
        guard inputDimActual == inputDim else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "input_dim = \(inputDim)",
                actual: "input_dim = \(inputDimActual)"
            )
        }
        
        // Handle edge case: empty sequence
        if sequenceLength == 0 {
            let emptyOutputs = LSTMUtils.createTensor(shape: [batchSize, 0, hiddenDim])
            let finalState: (MLXArray, MLXArray)
            if let initialState = initialState {
                finalState = initialState
            } else {
                finalState = try self.initialState(batchSize: batchSize)
            }
            return (emptyOutputs, finalState)
        }
        
        // Use provided initial state or create default
        var currentState: (MLXArray, MLXArray)
        if let initialState = initialState {
            // Validate provided initial state
            let expectedHiddenShape = [batchSize, hiddenDim]
            try LSTMUtils.validateStateTensor(initialState.0, expectedShape: expectedHiddenShape)
            try LSTMUtils.validateMatrixMemory(initialState.1, batchSize: batchSize, hiddenDim: hiddenDim)
            try LSTMUtils.validateStateStability([initialState.0, initialState.1])
            currentState = initialState
        } else {
            currentState = try self.initialState(batchSize: batchSize)
        }
        
        // Collect outputs for each timestep
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(sequenceLength)
        
        // Process each timestep
        for t in 0..<sequenceLength {
            let timestepInput = sequence[0..., t, 0...]  // [batch_size, input_dim]
            let (output, newState) = callAsFunction(timestepInput, state: currentState)
            outputs.append(output)
            currentState = newState
        }
        
        // Stack outputs along sequence dimension
        let allOutputs = stacked(outputs, axis: 1)  // [batch_size, sequence_length, hidden_dim]
        
        return (allOutputs, currentState)
    }
}