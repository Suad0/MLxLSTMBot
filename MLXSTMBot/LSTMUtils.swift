//
//  LSTMUtils.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Shared utilities and constants for xLSTM implementations
public struct LSTMUtils {
    
    // MARK: - Constants
    
    /// Numerical stability epsilon for division operations
    public static let epsilon: Float = 1e-8
    
    /// Maximum allowed tensor dimension to prevent memory issues
    public static let maxTensorDim: Int = 10000
    
    /// Minimum batch size
    public static let minBatchSize: Int = 1
    
    // MARK: - Error Types
    
    public enum LSTMError: Error, LocalizedError {
        case invalidTensorShape(expected: String, actual: String)
        case deviceMismatch(String)
        case numericalInstability(String)
        case invalidDimension(String)
        case memoryAllocationFailed(String)
        
        public var errorDescription: String? {
            switch self {
            case .invalidTensorShape(let expected, let actual):
                return "Invalid tensor shape. Expected: \(expected), Actual: \(actual)"
            case .deviceMismatch(let message):
                return "Device mismatch: \(message)"
            case .numericalInstability(let message):
                return "Numerical instability detected: \(message)"
            case .invalidDimension(let message):
                return "Invalid dimension: \(message)"
            case .memoryAllocationFailed(let message):
                return "Memory allocation failed: \(message)"
            }
        }
    }
    
    // MARK: - Tensor Shape Validation
    
    /// Validates that a tensor has exactly 3 dimensions for sequence processing
    /// - Parameter tensor: The input tensor to validate
    /// - Throws: LSTMError.invalidTensorShape if tensor doesn't have 3 dimensions
    public static func validateSequenceTensor(_ tensor: MLXArray) throws {
        guard tensor.ndim == 3 else {
            throw LSTMError.invalidTensorShape(
                expected: "3D tensor [batch_size, sequence_length, input_dim]",
                actual: "\(tensor.ndim)D tensor with shape \(tensor.shape)"
            )
        }
        
        // Validate positive dimensions
        for (index, dim) in tensor.shape.enumerated() {
            guard dim > 0 else {
                let dimNames = ["batch_size", "sequence_length", "input_dim"]
                throw LSTMError.invalidDimension("\(dimNames[index]) must be positive, got \(dim)")
            }
        }
    }
    
    /// Validates tensor dimensions for LSTM state tensors
    /// - Parameters:
    ///   - tensor: The state tensor to validate
    ///   - expectedShape: The expected shape as [batch_size, hidden_dim]
    /// - Throws: LSTMError.invalidTensorShape if shapes don't match
    public static func validateStateTensor(_ tensor: MLXArray, expectedShape: [Int]) throws {
        guard tensor.shape == expectedShape else {
            throw LSTMError.invalidTensorShape(
                expected: "\(expectedShape)",
                actual: "\(tensor.shape)"
            )
        }
    }
    
    /// Validates matrix memory tensor for mLSTM
    /// - Parameters:
    ///   - tensor: The matrix memory tensor to validate
    ///   - batchSize: Expected batch size
    ///   - hiddenDim: Expected hidden dimension
    /// - Throws: LSTMError.invalidTensorShape if shape is incorrect
    public static func validateMatrixMemory(_ tensor: MLXArray, batchSize: Int, hiddenDim: Int) throws {
        let expectedShape = [batchSize, hiddenDim, hiddenDim]
        guard tensor.shape == expectedShape else {
            throw LSTMError.invalidTensorShape(
                expected: "\(expectedShape)",
                actual: "\(tensor.shape)"
            )
        }
    }
    
    /// Validates that all tensors are on the same device
    /// - Parameter tensors: Array of tensors to check
    /// - Throws: LSTMError.deviceMismatch if tensors are on different devices
    public static func validateSameDevice(_ tensors: [MLXArray]) throws {
        guard !tensors.isEmpty else { return }
        
        // Note: MLX handles device management automatically for Apple Silicon
        // However, we still validate that tensors are properly allocated
        for (index, tensor) in tensors.enumerated() {
            // Check if tensor is properly allocated and accessible
            if tensor.size == 0 && tensor.shape.contains(where: { $0 > 0 }) {
                throw LSTMError.deviceMismatch("Tensor at index \(index) appears to be improperly allocated")
            }
        }
    }
    
    /// Validates input tensor for empty sequences and edge cases
    /// - Parameter input: Input tensor to validate
    /// - Throws: LSTMError for various edge cases
    public static func validateInputTensor(_ input: MLXArray) throws {
        // Check for empty tensor
        guard input.size > 0 else {
            throw LSTMError.invalidTensorShape(
                expected: "Non-empty tensor",
                actual: "Empty tensor with size 0"
            )
        }
        
        // Check for NaN or infinite values
        if hasNaNOrInf(input) {
            throw LSTMError.numericalInstability("Input tensor contains NaN or infinite values")
        }
        
        // Check for extremely large values that could cause numerical issues
        let maxAbsValue = MLX.max(MLX.abs(input)).item(Float.self)
        if maxAbsValue > 1e6 {
            print("Warning: Input tensor contains very large values (max: \(maxAbsValue)). This may cause numerical instability.")
        }
    }
    
    /// Validates state tensors for numerical stability
    /// - Parameter states: Array of state tensors to validate
    /// - Throws: LSTMError.numericalInstability if issues are detected
    public static func validateStateStability(_ states: [MLXArray]) throws {
        for (index, state) in states.enumerated() {
            if hasNaNOrInf(state) {
                throw LSTMError.numericalInstability("State tensor at index \(index) contains NaN or infinite values")
            }
            
            // Check for extremely small values in normalizer state (for sLSTM)
            if states.count == 3 && index == 2 { // normalizer state
                let minValue = MLX.min(state).item(Float.self)
                if minValue < epsilon * 10 {
                    print("Warning: Normalizer state contains very small values (min: \(minValue)). Adding stability epsilon.")
                }
            }
        }
    }
    
    /// Checks if tensor contains NaN or infinite values
    /// - Parameter tensor: Tensor to check
    /// - Returns: True if tensor contains NaN or infinite values
    private static func hasNaNOrInf(_ tensor: MLXArray) -> Bool {
        // Check for NaN: NaN != NaN
        let hasNaN = MLX.any(tensor .!= tensor).item(Bool.self)
        
        // Check for infinity by comparing with very large values
        let absValues = MLX.abs(tensor)
        let hasInf = MLX.any(absValues .> Float.greatestFiniteMagnitude).item(Bool.self)
        
        return hasNaN || hasInf
    }
    
    /// Validates sequence length for processing
    /// - Parameter sequenceLength: Length to validate
    /// - Throws: LSTMError.invalidDimension if sequence length is invalid
    public static func validateSequenceLength(_ sequenceLength: Int) throws {
        guard sequenceLength > 0 else {
            throw LSTMError.invalidDimension("Sequence length must be positive, got \(sequenceLength)")
        }
        
        if sequenceLength > 10000 {
            print("Warning: Very long sequence (\(sequenceLength) timesteps). This may cause memory issues.")
        }
    }
    
    // MARK: - Dimension Validation
    
    /// Validates that dimensions are within reasonable bounds
    /// - Parameters:
    ///   - inputDim: Input dimension
    ///   - hiddenDim: Hidden dimension
    /// - Throws: LSTMError.invalidDimension if dimensions are invalid
    public static func validateDimensions(inputDim: Int, hiddenDim: Int) throws {
        guard inputDim > 0 && inputDim <= maxTensorDim else {
            throw LSTMError.invalidDimension("Input dimension must be between 1 and \(maxTensorDim), got \(inputDim)")
        }
        
        guard hiddenDim > 0 && hiddenDim <= maxTensorDim else {
            throw LSTMError.invalidDimension("Hidden dimension must be between 1 and \(maxTensorDim), got \(hiddenDim)")
        }
    }
    
    /// Validates batch size
    /// - Parameter batchSize: The batch size to validate
    /// - Throws: LSTMError.invalidDimension if batch size is invalid
    public static func validateBatchSize(_ batchSize: Int) throws {
        guard batchSize >= minBatchSize else {
            throw LSTMError.invalidDimension("Batch size must be at least \(minBatchSize), got \(batchSize)")
        }
    }
    
    // MARK: - Numerical Stability Helpers
    
    /// Adds epsilon to prevent division by zero
    /// - Parameter tensor: Input tensor
    /// - Returns: Tensor with epsilon added
    public static func addEpsilon(_ tensor: MLXArray) -> MLXArray {
        return tensor + epsilon
    }
    
    /// Clamps exponential values to prevent overflow
    /// - Parameter tensor: Input tensor
    /// - Returns: Clamped tensor
    public static func clampExponential(_ tensor: MLXArray) -> MLXArray {
        // Clamp input to exp to prevent overflow (exp(50) is already very large)
        return MLX.clip(tensor, min: -50.0, max: 50.0)
    }
    
    // MARK: - Device Helpers
    
    /// Creates a tensor on the appropriate MLX device
    /// - Parameters:
    ///   - shape: Shape of the tensor
    ///   - value: Fill value (default: 0.0)
    /// - Returns: MLXArray filled with the specified value
    public static func createTensor(shape: [Int], value: Float = 0.0) -> MLXArray {
        return MLX.full(shape, values: value)
    }
    
    /// Creates an identity matrix tensor
    /// - Parameters:
    ///   - batchSize: Batch dimension
    ///   - dim: Matrix dimension
    /// - Returns: Batched identity matrices
    public static func createIdentityMatrix(batchSize: Int, dim: Int) -> MLXArray {
        let identity = MLX.eye(dim)
        return MLX.broadcast(identity.expandedDimensions(axis: 0), to: [batchSize, dim, dim])
    }
}
