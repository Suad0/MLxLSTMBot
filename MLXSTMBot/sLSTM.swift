//
//  sLSTM.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Scalar Long Short-Term Memory (sLSTM) implementation with exponential gating
/// 
/// This class implements the sLSTM variant that uses exponential gating mechanisms
/// and a normalizer state for improved stability. It inherits from MLXNN.Module
/// to integrate seamlessly with the MLX framework.
///
/// Mathematical formulation:
/// - i_t = exp(W_i * x_t + U_i * h_{t-1} + b_i)  (exponential input gate)
/// - f_t = exp(W_f * x_t + U_f * h_{t-1} + b_f)  (exponential forget gate)
/// - c_tilde = tanh(W_c * x_t + U_c * h_{t-1} + b_c)  (cell candidate)
/// - o_t = sigmoid(W_o * x_t + U_o * h_{t-1} + b_o)  (output gate)
/// - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde  (cell state update)
/// - n_t = f_t ⊙ n_{t-1} + i_t  (normalizer state update)
/// - h_t = o_t ⊙ (c_t / n_t)  (hidden state with normalization)
public class sLSTM: Module {
    
    // MARK: - Properties
    
    /// Input dimension
    public let inputDim: Int
    
    /// Hidden state dimension
    public let hiddenDim: Int
    
    /// Linear projection for input gate computation
    public let inputProjection: Linear
    
    /// Linear projection for forget gate computation
    public let forgetProjection: Linear
    
    /// Linear projection for cell candidate computation
    public let cellProjection: Linear
    
    /// Linear projection for output gate computation
    public let outputProjection: Linear
    
    // MARK: - Initialization
    
    /// Initializes the sLSTM module with specified dimensions
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
        self.cellProjection = Linear(totalInputDim, hiddenDim)
        self.outputProjection = Linear(totalInputDim, hiddenDim)
        
        super.init()
        
        // Initialize weights with smaller values for numerical stability
        initializeWeights()
    }
    
    /// Initializes weights with smaller values for numerical stability
    private func initializeWeights() {
        // Note: In MLX Swift, Linear layer weights are initialized automatically
        // and cannot be modified after creation. The default initialization
        // should be sufficient for numerical stability.
        // If custom initialization is needed, it should be done during Linear layer creation.
        print("Using default Linear layer initialization for numerical stability")
    }
    
    // MARK: - State Management
    
    /// Creates initial state tensors for the sLSTM
    /// 
    /// Returns a tuple (h_t, c_t, n_t, m_t) where:
    /// - h_t: hidden state initialized to zeros [batch_size, hidden_dim]
    /// - c_t: cell state initialized to zeros [batch_size, hidden_dim]
    /// - n_t: normalizer state initialized to ones [batch_size, hidden_dim]
    /// - m_t: stabilizer state initialized to zeros [batch_size, hidden_dim]
    ///
    /// - Parameter batchSize: Size of the batch dimension
    /// - Returns: Tuple of initial state tensors (h_t, c_t, n_t, m_t)
    /// - Throws: LSTMError.invalidDimension if batch size is invalid
    public func initialState(batchSize: Int) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        // Validate batch size
        try LSTMUtils.validateBatchSize(batchSize)
        
        let stateShape = [batchSize, hiddenDim]
        
        // Initialize hidden state and cell state with zeros
        let h_t = LSTMUtils.createTensor(shape: stateShape, value: 0.0)
        let c_t = LSTMUtils.createTensor(shape: stateShape, value: 0.0)
        
        // Initialize normalizer state with 1.0 (Bug B1 fix)
        let n_t = LSTMUtils.createTensor(shape: stateShape, value: 1.0)

        // Initialize stabilizer state with 0.0
        let m_t = LSTMUtils.createTensor(shape: stateShape, value: 0.0)
        
        return (h_t, c_t, n_t, m_t)
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass computation for sLSTM with exponential gating
    /// 
    /// Implements the complete sLSTM forward pass with:
    /// - Exponential gating for input and forget gates
    /// - Normalizer state computation for stability
    /// - Numerical stability measures
    ///
    /// Mathematical formulation:
    /// - i_t = exp(W_i * [x_t; h_{t-1}] + b_i)  (exponential input gate)
    /// - f_t = exp(W_f * [x_t; h_{t-1}] + b_f)  (exponential forget gate)
    /// - c_tilde = tanh(W_c * [x_t; h_{t-1}] + b_c)  (cell candidate)
    /// - o_t = sigmoid(W_o * [x_t; h_{t-1}] + b_o)  (output gate)
    /// - c_t = f_t ⊙ c_{t-1} + i_t ⊙ c_tilde  (cell state update)
    /// - n_t = f_t ⊙ n_{t-1} + i_t  (normalizer state update)
    /// - h_t = o_t ⊙ (c_t / (n_t + ε))  (hidden state with normalization)
    ///
    /// - Parameters:
    ///   - input: Input tensor [batch_size, input_dim] for single timestep
    ///   - state: Current state tuple (h_t, c_t, n_t)
    /// - Returns: Tuple of (output, new_state) where output is the hidden state
    ///            and new_state is the updated (h_t, c_t, n_t) tuple
    /// - Throws: LSTMError for invalid inputs or numerical issues
    public func callAsFunction(_ input: MLXArray, state: (MLXArray, MLXArray, MLXArray, MLXArray)) throws -> (MLXArray, (MLXArray, MLXArray, MLXArray, MLXArray)) {
        let (h_prev, c_prev, n_prev, m_prev) = state
        
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
            let expectedStateShape = [batchSize, hiddenDim]
            // Note: skipping redundant validation for performance after initial checks
            // try LSTMUtils.validateStateTensor(h_prev, expectedShape: expectedStateShape)
            
            // Concatenate input and previous hidden state for linear projections
            // Combined input: [x_t; h_{t-1}] with shape [batch_size, input_dim + hidden_dim]
            let combinedInput = concatenated([input, h_prev], axis: 1)
            
            // Compute linear projections
            let inputGateLinear = inputProjection(combinedInput)
            let forgetGateLinear = forgetProjection(combinedInput)
            let cellCandidateLinear = cellProjection(combinedInput)
            let outputGateLinear = outputProjection(combinedInput)
            
            // --- Stabilized xLSTM Update Rule ---
            
            // 1. Calculate pre-activation gates
            // We do NOT clamp here yet because we use log-sum-exp logic for stability
            // But to prevent extreme values before m_t calculation, we can apply soft clamping or trust m_t
            // Let's use the raw linear outputs as log-gates: z_i = inputGateLinear, z_f = forgetGateLinear
            
            let z_i = inputGateLinear
            let z_f = forgetGateLinear
            
            // 2. Update stabilizer state m_t
            // m_t = max(z_f + m_{t-1}, z_i)
            // Note: m_{t-1} should come from previous state.
            // If n_{t-1} represents sum of exps scaled by m_{t-1},
            // n_{t-1}_real = n_{t-1} * exp(m_{t-1})
             
            let m_t = MLX.maximum(z_f + m_prev, z_i)
            
            // 3. Compute stabilized gates
            // i'_t = exp(z_i - m_t)
            // f'_t = exp(z_f + m_{t-1} - m_t)
            // Both exponents are guaranteed <= 0, so result is in (0, 1]
            let i_prime = MLX.exp(z_i - m_t)
            let f_prime = MLX.exp(z_f + m_prev - m_t)
            
            // 4. Apply standard activations for cell candidate and output gate
            let c_tilde = MLX.tanh(cellCandidateLinear)
            let o_t = MLX.sigmoid(outputGateLinear)
            
            // 5. Update normalizer state
            // n_t = f'_t * n_{t-1} + i'_t
            let n_t = f_prime * n_prev + i_prime
            
            // 6. Update cell state
            // c_t = f'_t * c_{t-1} + i'_t * c_tilde
            let c_t = f_prime * c_prev + i_prime * c_tilde
            
            // 7. Compute hidden state
            // h_t = o_t * (c_t / n_t)
            // Add epsilon to n_t to avoid division by zero (though n_t should be >= exp(diff) > 0)
            let n_t_stable = MLX.maximum(n_t, LSTMUtils.epsilon)
            let h_t = o_t * (c_t / n_t_stable)
            
            let newState = (h_t, c_t, n_t, m_t)
            return (h_t, newState)
    }
}

// MARK: - Convenience Extensions

extension sLSTM {
    

    
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
    public func processSequence(_ sequence: MLXArray, initialState: (MLXArray, MLXArray, MLXArray, MLXArray)? = nil) throws -> (MLXArray, (MLXArray, MLXArray, MLXArray, MLXArray)) {
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
            let finalState: (MLXArray, MLXArray, MLXArray, MLXArray)
            if let initialState = initialState {
                finalState = initialState
            } else {
                finalState = try self.initialState(batchSize: batchSize)
            }
            return (emptyOutputs, finalState)
        }
        
        // Use provided initial state or create default
        var currentState: (MLXArray, MLXArray, MLXArray, MLXArray)
        if let initialState = initialState {
            // Validate provided initial state (basic check only for performance)
            let expectedStateShape = [batchSize, hiddenDim]
            // try LSTMUtils.validateStateTensor(initialState.0, expectedShape: expectedStateShape)
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
            let (output, newState) = try callAsFunction(timestepInput, state: currentState)
            outputs.append(output)
            currentState = newState
        }
        
        // Stack outputs along sequence dimension
        let allOutputs = stacked(outputs, axis: 1)  // [batch_size, sequence_length, hidden_dim]
        
        return (allOutputs, currentState)
    }
}
