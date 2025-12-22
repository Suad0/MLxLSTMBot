//
//  FeedForward.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Gated Linear Unit (GLU) Feed-Forward Network
/// 
/// Implements a GLU-based feed-forward network commonly used in transformer architectures.
/// The GLU applies a gating mechanism to control information flow.
/// 
/// Mathematical formulation:
/// - gate = sigmoid(W_gate * x + b_gate)
/// - value = W_value * x + b_value
/// - output = gate ⊙ value
/// - final_output = W_out * output + b_out
public class GatedFeedForward: Module {
    
    // MARK: - Properties
    
    /// Input dimension
    public let inputDim: Int
    
    /// Hidden dimension (typically 4x input dimension)
    public let hiddenDim: Int
    
    /// Gate projection layer
    public let gateProjection: Linear
    
    /// Value projection layer
    public let valueProjection: Linear
    
    /// Output projection layer
    public let outputProjection: Linear
    
    // MARK: - Initialization
    
    /// Initializes the gated feed-forward network
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - hiddenDim: Hidden dimension (default: 4x input dimension)
    public init(inputDim: Int, hiddenDim: Int? = nil) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim ?? (inputDim * 4)
        
        // Initialize projection layers
        self.gateProjection = Linear(inputDim, self.hiddenDim)
        self.valueProjection = Linear(inputDim, self.hiddenDim)
        self.outputProjection = Linear(self.hiddenDim, inputDim)
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass for gated feed-forward network
    /// - Parameter input: Input tensor of shape [..., inputDim]
    /// - Returns: Output tensor of same shape as input
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Compute gate and value projections
        let gate = MLX.sigmoid(gateProjection(input))
        let value = valueProjection(input)
        
        // Apply gating mechanism
        let gated = gate * value
        
        // Final output projection
        return outputProjection(gated)
    }
}

/// Standard Multi-Layer Perceptron (MLP) Feed-Forward Network
/// 
/// Implements a standard MLP with ReLU activation for comparison with GLU.
/// 
/// Mathematical formulation:
/// - hidden = ReLU(W_1 * x + b_1)
/// - output = W_2 * hidden + b_2
public class MLPFeedForward: Module {
    
    // MARK: - Properties
    
    /// Input dimension
    public let inputDim: Int
    
    /// Hidden dimension
    public let hiddenDim: Int
    
    /// First linear layer
    public let linear1: Linear
    
    /// Second linear layer
    public let linear2: Linear
    
    // MARK: - Initialization
    
    /// Initializes the MLP feed-forward network
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - hiddenDim: Hidden dimension (default: 4x input dimension)
    public init(inputDim: Int, hiddenDim: Int? = nil) {
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim ?? (inputDim * 4)
        
        // Initialize linear layers
        self.linear1 = Linear(inputDim, self.hiddenDim)
        self.linear2 = Linear(self.hiddenDim, inputDim)
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass for MLP feed-forward network
    /// - Parameter input: Input tensor of shape [..., inputDim]
    /// - Returns: Output tensor of same shape as input
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // First layer with ReLU activation (max(0, x))
        let hidden = MLX.maximum(linear1(input), MLXArray(0.0))
        
        // Second layer (output)
        return linear2(hidden)
    }
}
