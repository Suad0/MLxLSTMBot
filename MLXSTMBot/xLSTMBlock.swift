//
//  xLSTMBlock.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Layer state type for xLSTM blocks
public enum LayerState {
    case sLSTM(h: MLXArray, c: MLXArray, n: MLXArray)
    case mLSTM(h: MLXArray, C: MLXArray)
    
    /// Extract hidden state from any layer type
    public var hiddenState: MLXArray {
        switch self {
        case .sLSTM(let h, _, _):
            return h
        case .mLSTM(let h, _):
            return h
        }
    }
}

/// xLSTM Block Type
public enum xLSTMBlockType {
    case sLSTM
    case mLSTM
}

/// xLSTM Block wrapper that can contain either sLSTM or mLSTM
/// 
/// This class implements the complete xLSTM block with:
/// - Pre-layer normalization using RMSNorm
/// - Residual connections around the LSTM layer
/// - Optional feed-forward network (for sLSTM blocks)
/// - Proper state management for autoregressive generation
public class xLSTMBlock: Module {
    
    // MARK: - Properties
    
    /// Block type (sLSTM or mLSTM)
    public let blockType: xLSTMBlockType
    
    /// Input/hidden dimension
    public let hiddenDim: Int
    
    /// Pre-layer normalization
    public let preNorm: RMSNorm
    
    /// sLSTM layer (if block type is sLSTM)
    public let sLSTMLayer: sLSTM?
    
    /// mLSTM layer (if block type is mLSTM)
    public let mLSTMLayer: mLSTM?
    
    /// Feed-forward network (only for sLSTM blocks)
    public let feedForward: GatedFeedForward?
    
    /// Post-FFN normalization (only for sLSTM blocks with FFN)
    public let postNorm: RMSNorm?
    
    // MARK: - Initialization
    
    /// Initializes an xLSTM block
    /// - Parameters:
    ///   - blockType: Type of LSTM block (sLSTM or mLSTM)
    ///   - hiddenDim: Hidden dimension
    ///   - inputDim: Input dimension (for the first layer, otherwise equals hiddenDim)
    ///   - includeFeedForward: Whether to include FFN (only applies to sLSTM blocks)
    /// - Throws: LSTMError for invalid configurations
    public init(blockType: xLSTMBlockType, hiddenDim: Int, inputDim: Int? = nil, includeFeedForward: Bool = true) throws {
        self.blockType = blockType
        self.hiddenDim = hiddenDim
        
        let actualInputDim = inputDim ?? hiddenDim
        
        // Validate dimensions
        try LSTMUtils.validateDimensions(inputDim: actualInputDim, hiddenDim: hiddenDim)
        
        // Initialize pre-layer normalization
        self.preNorm = RMSNorm(normalizedShape: hiddenDim)
        
        // Initialize LSTM layer based on block type
        switch blockType {
        case .sLSTM:
            self.sLSTMLayer = try sLSTM(inputDim: actualInputDim, hiddenDim: hiddenDim)
            self.mLSTMLayer = nil
            
            // Add feed-forward network for sLSTM blocks
            if includeFeedForward {
                self.feedForward = GatedFeedForward(inputDim: hiddenDim)
                self.postNorm = RMSNorm(normalizedShape: hiddenDim)
            } else {
                self.feedForward = nil
                self.postNorm = nil
            }
            
        case .mLSTM:
            self.mLSTMLayer = try mLSTM(inputDim: actualInputDim, hiddenDim: hiddenDim)
            self.sLSTMLayer = nil
            
            // mLSTM blocks typically don't include FFN
            self.feedForward = nil
            self.postNorm = nil
        }
        
        super.init()
    }
    
    // MARK: - State Management
    
    /// Creates initial state for this block
    /// - Parameter batchSize: Batch size
    /// - Returns: Initial layer state
    /// - Throws: LSTMError for invalid batch size
    public func initialState(batchSize: Int) throws -> LayerState {
        try LSTMUtils.validateBatchSize(batchSize)
        
        switch blockType {
        case .sLSTM:
            guard let sLSTM = sLSTMLayer else {
                fatalError("sLSTM layer not initialized")
            }
            let (h, c, n) = try sLSTM.initialState(batchSize: batchSize)
            return .sLSTM(h: h, c: c, n: n)
            
        case .mLSTM:
            guard let mLSTM = mLSTMLayer else {
                fatalError("mLSTM layer not initialized")
            }
            let (h, C) = try mLSTM.initialState(batchSize: batchSize)
            return .mLSTM(h: h, C: C)
        }
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass for xLSTM block
    /// - Parameters:
    ///   - input: Input tensor [batch_size, hidden_dim] for single timestep
    ///   - state: Current layer state
    /// - Returns: Tuple of (output, new_state)
    public func callAsFunction(_ input: MLXArray, state: LayerState) -> (MLXArray, LayerState) {
        // Validate input dimensions
        guard input.ndim == 2 && input.shape[1] == hiddenDim else {
            fatalError("Invalid input shape. Expected [batch_size, \(hiddenDim)], got \(input.shape)")
        }
        
        // Validate state type matches block type
        switch (blockType, state) {
        case (.sLSTM, .sLSTM), (.mLSTM, .mLSTM):
            break // Valid combination
        default:
            fatalError("State type \(state) doesn't match block type \(blockType)")
        }
        
        // Pre-layer normalization
        let normalizedInput = preNorm(input)
        
        // LSTM forward pass with residual connection
        let (lstmOutput, newLSTMState): (MLXArray, LayerState)
        
        switch blockType {
        case .sLSTM:
            guard let sLSTM = sLSTMLayer else {
                fatalError("sLSTM layer not initialized")
            }
            
            if case .sLSTM(let h, let c, let n) = state {
                let (output, (newH, newC, newN)) = sLSTM(normalizedInput, state: (h, c, n))
                lstmOutput = output
                newLSTMState = .sLSTM(h: newH, c: newC, n: newN)
            } else {
                fatalError("Invalid state type for sLSTM block")
            }
            
        case .mLSTM:
            guard let mLSTM = mLSTMLayer else {
                fatalError("mLSTM layer not initialized")
            }
            
            if case .mLSTM(let h, let C) = state {
                let (output, (newH, newC)) = mLSTM(normalizedInput, state: (h, C))
                lstmOutput = output
                newLSTMState = .mLSTM(h: newH, C: newC)
            } else {
                fatalError("Invalid state type for mLSTM block")
            }
        }
        
        // First residual connection: x + LSTM(norm(x))
        let afterLSTM = input + lstmOutput
        
        // Apply feed-forward network if present (only for sLSTM blocks)
        let finalOutput: MLXArray
        if let ffn = feedForward, let postNorm = postNorm {
            // Post-normalization before FFN
            let normalizedAfterLSTM = postNorm(afterLSTM)
            
            // FFN forward pass
            let ffnOutput = ffn(normalizedAfterLSTM)
            
            // Second residual connection: x + FFN(norm(x))
            finalOutput = afterLSTM + ffnOutput
        } else {
            finalOutput = afterLSTM
        }
        
        return (finalOutput, newLSTMState)
    }
    
    /// Process a sequence through the block
    /// - Parameters:
    ///   - sequence: Input sequence [batch_size, sequence_length, hidden_dim]
    ///   - initialState: Optional initial state
    /// - Returns: Tuple of (outputs, final_state)
    /// - Throws: LSTMError for invalid inputs
    public func processSequence(_ sequence: MLXArray, initialState: LayerState? = nil) throws -> (MLXArray, LayerState) {
        // Validate sequence tensor
        try LSTMUtils.validateSequenceTensor(sequence)
        
        let batchSize = sequence.shape[0]
        let sequenceLength = sequence.shape[1]
        let inputDimActual = sequence.shape[2]
        
        guard inputDimActual == hiddenDim else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "hidden_dim = \(hiddenDim)",
                actual: "input_dim = \(inputDimActual)"
            )
        }
        
        // Use provided initial state or create default
        var currentState: LayerState
        if let initialState = initialState {
            currentState = initialState
        } else {
            currentState = try self.initialState(batchSize: batchSize)
        }
        
        // Handle empty sequence
        if sequenceLength == 0 {
            let emptyOutputs = LSTMUtils.createTensor(shape: [batchSize, 0, hiddenDim])
            return (emptyOutputs, currentState)
        }
        
        // Process each timestep
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(sequenceLength)
        
        for t in 0..<sequenceLength {
            let timestepInput = sequence[0..., t, 0...]  // [batch_size, hidden_dim]
            let (output, newState) = callAsFunction(timestepInput, state: currentState)
            outputs.append(output)
            currentState = newState
        }
        
        // Stack outputs along sequence dimension
        let allOutputs = stacked(outputs, axis: 1)  // [batch_size, sequence_length, hidden_dim]
        
        return (allOutputs, currentState)
    }
}

// MARK: - Convenience Extensions

extension xLSTMBlock {
    
    /// Creates initial state with default batch size of 1
    /// - Returns: Initial layer state for single sample
    public func initialState() -> LayerState {
        do {
            return try initialState(batchSize: 1)
        } catch {
            fatalError("Failed to create initial state: \(error)")
        }
    }
}
