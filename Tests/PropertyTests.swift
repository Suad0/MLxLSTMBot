//
//  PropertyTests.swift
//  MLXSTMBot Tests
//
//  Created by Kiro on 22.12.25.
//

import XCTest
import MLX
import MLXNN
@testable import MLXSTMBot

/// Property-based tests for xLSTM implementation
final class PropertyTests: XCTestCase {
    
    // MARK: - Property Test Helpers
    
    /// Generates random valid tensor shapes for testing
    private func generateValidTensorShape() -> [Int] {
        let batchSize = Int.random(in: 1...32)
        let sequenceLength = Int.random(in: 1...100)
        let inputDim = Int.random(in: 1...512)
        return [batchSize, sequenceLength, inputDim]
    }
    
    /// Generates random valid dimensions
    private func generateValidDimensions() -> (inputDim: Int, hiddenDim: Int) {
        let inputDim = Int.random(in: 1...512)
        let hiddenDim = Int.random(in: 1...512)
        return (inputDim, hiddenDim)
    }
    
    // MARK: - Property 5: Tensor shape consistency
    
    /// **Feature: xlstm-implementation, Property 5: Tensor shape consistency**
    /// For any 3D input tensor with shape (batch_size, sequence_length, input_dim),
    /// the tensor validation should process it correctly and maintain consistent shape handling
    /// **Validates: Requirements 1.4, 2.4**
    func testProperty5_TensorShapeConsistency() {
        let iterations = 100
        
        for iteration in 1...iterations {
            // Generate random valid tensor shape
            let shape = generateValidTensorShape()
            let batchSize = shape[0]
            let sequenceLength = shape[1] 
            let inputDim = shape[2]
            
            // Create tensor with this shape
            let tensor = MLX.full(shape, values: MLXArray(0.0))
            
            // Property: Valid 3D tensors should pass validation
            XCTAssertNoThrow(
                try LSTMUtils.validateSequenceTensor(tensor),
                "Iteration \(iteration): Valid 3D tensor with shape \(shape) should pass validation"
            )
            
            // Property: Extracted dimensions should match original shape
            XCTAssertEqual(tensor.shape[0], batchSize, "Iteration \(iteration): Batch size should be preserved")
            XCTAssertEqual(tensor.shape[1], sequenceLength, "Iteration \(iteration): Sequence length should be preserved")
            XCTAssertEqual(tensor.shape[2], inputDim, "Iteration \(iteration): Input dimension should be preserved")
            
            // Property: State tensors with matching batch size and hidden dim should be valid
            let hiddenDim = Int.random(in: 1...512)
            let stateTensor = MLX.full([batchSize, hiddenDim], values: MLXArray(0.0))
            let expectedStateShape = [batchSize, hiddenDim]
            
            XCTAssertNoThrow(
                try LSTMUtils.validateStateTensor(stateTensor, expectedShape: expectedStateShape),
                "Iteration \(iteration): State tensor with correct shape should pass validation"
            )
            
            // Property: Matrix memory with correct dimensions should be valid
            let matrixMemory = MLX.full([batchSize, hiddenDim, hiddenDim], values: MLXArray(0.0))
            
            XCTAssertNoThrow(
                try LSTMUtils.validateMatrixMemory(matrixMemory, batchSize: batchSize, hiddenDim: hiddenDim),
                "Iteration \(iteration): Matrix memory with correct shape should pass validation"
            )
        }
    }
    
    /// Test that invalid tensor shapes are consistently rejected
    func testProperty5_InvalidTensorShapeRejection() {
        let iterations = 50
        
        for iteration in 1...iterations {
            // Generate invalid shapes (not 3D)
            let invalidShapes = [
                [Int.random(in: 1...32)], // 1D
                [Int.random(in: 1...32), Int.random(in: 1...100)], // 2D
                [Int.random(in: 1...32), Int.random(in: 1...100), Int.random(in: 1...512), Int.random(in: 1...10)] // 4D
            ]
            
            for (shapeIndex, shape) in invalidShapes.enumerated() {
                let tensor = MLX.full(shape, values: MLXArray(0.0))
                
                // Property: Invalid dimensional tensors should be rejected
                XCTAssertThrowsError(
                    try LSTMUtils.validateSequenceTensor(tensor),
                    "Iteration \(iteration), Shape \(shapeIndex): Invalid \(shape.count)D tensor should be rejected"
                ) { error in
                    XCTAssertTrue(error is LSTMUtils.LSTMError, "Should throw LSTMError")
                }
            }
        }
    }
    
    /// Test that zero or negative dimensions are consistently rejected
    func testProperty5_InvalidDimensionRejection() {
        let iterations = 50
        
        for iteration in 1...iterations {
            // Generate shapes with zero or negative dimensions
            let validDim = Int.random(in: 1...100)
            let invalidShapes = [
                [0, validDim, validDim], // Zero batch size
                [validDim, 0, validDim], // Zero sequence length
                [validDim, validDim, 0], // Zero input dim
                [-1, validDim, validDim], // Negative batch size
                [validDim, -1, validDim], // Negative sequence length
                [validDim, validDim, -1]  // Negative input dim
            ]
            
            for (shapeIndex, shape) in invalidShapes.enumerated() {
                let tensor = MLX.full(shape.map { max(0, $0) }, values: MLXArray(0.0)) // MLX doesn't allow negative shapes, so clamp for creation
                
                if shape.contains(where: { $0 <= 0 }) {
                    // Property: Tensors with non-positive dimensions should be rejected
                    XCTAssertThrowsError(
                        try LSTMUtils.validateSequenceTensor(tensor),
                        "Iteration \(iteration), Shape \(shapeIndex): Tensor with non-positive dimensions should be rejected"
                    ) { error in
                        XCTAssertTrue(error is LSTMUtils.LSTMError, "Should throw LSTMError")
                    }
                }
            }
        }
    }
    
    /// Test batch size validation consistency
    func testProperty5_BatchSizeValidation() {
        let iterations = 100
        
        for iteration in 1...iterations {
            let batchSize = Int.random(in: 1...128)
            
            // Property: Valid batch sizes should pass validation
            XCTAssertNoThrow(
                try LSTMUtils.validateBatchSize(batchSize),
                "Iteration \(iteration): Valid batch size \(batchSize) should pass validation"
            )
            
            // Property: Zero and negative batch sizes should be rejected
            let invalidBatchSizes = [0, -1, -Int.random(in: 1...100)]
            
            for invalidBatchSize in invalidBatchSizes {
                XCTAssertThrowsError(
                    try LSTMUtils.validateBatchSize(invalidBatchSize),
                    "Iteration \(iteration): Invalid batch size \(invalidBatchSize) should be rejected"
                ) { error in
                    XCTAssertTrue(error is LSTMUtils.LSTMError, "Should throw LSTMError")
                }
            }
        }
    }
    
    /// Test dimension validation consistency
    func testProperty5_DimensionValidation() {
        let iterations = 100
        
        for iteration in 1...iterations {
            let (inputDim, hiddenDim) = generateValidDimensions()
            
            // Property: Valid dimensions should pass validation
            XCTAssertNoThrow(
                try LSTMUtils.validateDimensions(inputDim: inputDim, hiddenDim: hiddenDim),
                "Iteration \(iteration): Valid dimensions (input: \(inputDim), hidden: \(hiddenDim)) should pass validation"
            )
            
            // Property: Invalid dimensions should be rejected
            let invalidCases = [
                (0, hiddenDim), // Zero input dim
                (inputDim, 0), // Zero hidden dim
                (-1, hiddenDim), // Negative input dim
                (inputDim, -1), // Negative hidden dim
                (LSTMUtils.maxTensorDim + 1, hiddenDim), // Oversized input dim
                (inputDim, LSTMUtils.maxTensorDim + 1)  // Oversized hidden dim
            ]
            
            for (invalidInput, invalidHidden) in invalidCases {
                XCTAssertThrowsError(
                    try LSTMUtils.validateDimensions(inputDim: invalidInput, hiddenDim: invalidHidden),
                    "Iteration \(iteration): Invalid dimensions (input: \(invalidInput), hidden: \(invalidHidden)) should be rejected"
                ) { error in
                    XCTAssertTrue(error is LSTMUtils.LSTMError, "Should throw LSTMError")
                }
            }
        }
    }
}