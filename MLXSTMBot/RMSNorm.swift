//
//  RMSNorm.swift
//  MLXSTMBot
//
//  Created by Kiro on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Root Mean Square Layer Normalization (RMSNorm)
/// 
/// RMSNorm is a simplified version of LayerNorm that normalizes using only the root mean square
/// of the input activations, without centering (no mean subtraction).
/// 
/// Mathematical formulation:
/// - RMS(x) = sqrt(mean(x^2) + ε)
/// - y = (x / RMS(x)) * γ
/// 
/// Where γ is a learnable scale parameter.
public class RMSNorm: Module {
    
    // MARK: - Properties
    
    /// Dimension to normalize over (typically the last dimension)
    public let normalizedShape: Int
    
    /// Small epsilon for numerical stability
    public let eps: Float
    
    /// Learnable scale parameter
    public let weight: MLXArray
    
    // MARK: - Initialization
    
    /// Initializes RMSNorm layer
    /// - Parameters:
    ///   - normalizedShape: The dimension to normalize over
    ///   - eps: Small value for numerical stability (default: 1e-6)
    public init(normalizedShape: Int, eps: Float = 1e-6) {
        self.normalizedShape = normalizedShape
        self.eps = eps
        
        // Initialize scale parameter to ones
        self.weight = MLX.ones([normalizedShape])
        
        super.init()
    }
    
    // MARK: - Forward Pass
    
    /// Forward pass for RMSNorm
    /// - Parameter input: Input tensor of shape [..., normalizedShape]
    /// - Returns: Normalized tensor with same shape as input
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let squared = input * input
        let meanSquared = MLX.mean(squared, axis: -1, keepDims: true)
        let rms = MLX.sqrt(meanSquared + eps)
        
        // Normalize and scale: (x / RMS(x)) * weight
        let normalized = input / rms
        return normalized * weight
    }
}