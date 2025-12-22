//
//  LSTMUtilsTests.swift
//  MLXSTMBot Tests
//
//  Created by Kiro on 22.12.25.
//

import XCTest
import MLX
import MLXNN
@testable import MLXSTMBot

final class LSTMUtilsTests: XCTestCase {
    
    // MARK: - Tensor Shape Validation Tests
    
    func testValidateSequenceTensor_ValidInput() throws {
        // Test with valid 3D tensor
        let validTensor = MLX.full([2, 10, 5], values: MLXArray(0.0)) // [batch_size, sequence_length, input_dim]
        
        // Should not throw
        XCTAssertNoThrow(try LSTMUtils.validateSequenceTensor(validTensor))
    }
    
    func testValidateSequenceTensor_InvalidDimensions() {
        // Test with 2D tensor (invalid)
        let invalid2D = MLX.full([2, 10], values: MLXArray(0.0))
        
        XCTAssertThrowsError(try LSTMUtils.validateSequenceTensor(invalid2D)) { error in
            guard case LSTMUtils.LSTMError.invalidTensorShape(let expected, let actual) = error else {
                XCTFail("Expected invalidTensorShape error")
                return
            }
            XCTAssertTrue(expected.contains("3D tensor"))
            XCTAssertTrue(actual.contains("2D tensor"))
        }
        
        // Test with 4D tensor (invalid)
        let invalid4D = MLX.full([2, 10, 5, 3], values: MLXArray(0.0))
        
        XCTAssertThrowsError(try LSTMUtils.validateSequenceTensor(invalid4D)) { error in
            guard case LSTMUtils.LSTMError.invalidTensorShape = error else {
                XCTFail("Expected invalidTensorShape error")
                return
            }
        }
    }
    
    func testValidateSequenceTensor_ZeroDimensions() {
        // Test with zero dimensions
        let zeroTensor = MLX.full([0, 10, 5], values: MLXArray(0.0))
        
        XCTAssertThrowsError(try LSTMUtils.validateSequenceTensor(zeroTensor)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension(let message) = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
            XCTAssertTrue(message.contains("batch_size"))
        }
    }
    
    func testValidateStateTensor_ValidInput() throws {
        let stateTensor = MLX.full([2, 64], values: MLXArray(0.0)) // [batch_size, hidden_dim]
        let expectedShape = [2, 64]
        
        XCTAssertNoThrow(try LSTMUtils.validateStateTensor(stateTensor, expectedShape: expectedShape))
    }
    
    func testValidateStateTensor_InvalidShape() {
        let stateTensor = MLXArray(0.0, shape: [2, 32])
        let expectedShape = [2, 64]
        
        XCTAssertThrowsError(try LSTMUtils.validateStateTensor(stateTensor, expectedShape: expectedShape)) { error in
            guard case LSTMUtils.LSTMError.invalidTensorShape = error else {
                XCTFail("Expected invalidTensorShape error")
                return
            }
        }
    }
    
    func testValidateMatrixMemory_ValidInput() throws {
        let matrixMemory = MLXArray(0.0, shape: [2, 64, 64]) // [batch_size, hidden_dim, hidden_dim]
        
        XCTAssertNoThrow(try LSTMUtils.validateMatrixMemory(matrixMemory, batchSize: 2, hiddenDim: 64))
    }
    
    func testValidateMatrixMemory_InvalidShape() {
        let matrixMemory = MLXArray(0.0, shape: [2, 32, 64])
        
        XCTAssertThrowsError(try LSTMUtils.validateMatrixMemory(matrixMemory, batchSize: 2, hiddenDim: 64)) { error in
            guard case LSTMUtils.LSTMError.invalidTensorShape = error else {
                XCTFail("Expected invalidTensorShape error")
                return
            }
        }
    }
    
    // MARK: - Dimension Validation Tests
    
    func testValidateDimensions_ValidInput() throws {
        XCTAssertNoThrow(try LSTMUtils.validateDimensions(inputDim: 128, hiddenDim: 256))
        XCTAssertNoThrow(try LSTMUtils.validateDimensions(inputDim: 1, hiddenDim: 1))
    }
    
    func testValidateDimensions_InvalidInput() {
        // Test zero dimensions
        XCTAssertThrowsError(try LSTMUtils.validateDimensions(inputDim: 0, hiddenDim: 64)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
        
        // Test negative dimensions
        XCTAssertThrowsError(try LSTMUtils.validateDimensions(inputDim: 64, hiddenDim: -1)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
        
        // Test oversized dimensions
        XCTAssertThrowsError(try LSTMUtils.validateDimensions(inputDim: 20000, hiddenDim: 64)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
    }
    
    func testValidateBatchSize_ValidInput() throws {
        XCTAssertNoThrow(try LSTMUtils.validateBatchSize(1))
        XCTAssertNoThrow(try LSTMUtils.validateBatchSize(32))
        XCTAssertNoThrow(try LSTMUtils.validateBatchSize(1024))
    }
    
    func testValidateBatchSize_InvalidInput() {
        XCTAssertThrowsError(try LSTMUtils.validateBatchSize(0)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
        
        XCTAssertThrowsError(try LSTMUtils.validateBatchSize(-5)) { error in
            guard case LSTMUtils.LSTMError.invalidDimension = error else {
                XCTFail("Expected invalidDimension error")
                return
            }
        }
    }
    
    // MARK: - Numerical Stability Tests
    
    func testAddEpsilon() {
        let input = MLXArray([0.0, 1.0, -1.0])
        let result = LSTMUtils.addEpsilon(input)
        
        // Check that epsilon was added
        let expected = MLXArray([LSTMUtils.epsilon, 1.0 + LSTMUtils.epsilon, -1.0 + LSTMUtils.epsilon])
        
        // Compare with small tolerance
        let diff = MLX.abs(result - expected)
        let maxDiff = MLX.max(diff).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-10)
    }
    
    func testClampExponential() {
        let input = MLXArray([-100.0, 0.0, 100.0])
        let result = LSTMUtils.clampExponential(input)
        
        // Check that values are clamped to [-50, 50]
        let minVal = MLX.min(result).item(Float.self)
        let maxVal = MLX.max(result).item(Float.self)
        
        XCTAssertGreaterThanOrEqual(minVal, -50.0)
        XCTAssertLessThanOrEqual(maxVal, 50.0)
    }
    
    // MARK: - Device Helper Tests
    
    func testCreateTensor() {
        let tensor = LSTMUtils.createTensor(shape: [2, 3], value: 1.5)
        
        XCTAssertEqual(tensor.shape, [2, 3])
        
        // Check that all values are 1.5
        let allValues = MLX.allClose(tensor, MLXArray(1.5))
        XCTAssertTrue(allValues.item(Bool.self))
    }
    
    func testCreateIdentityMatrix() {
        let identityBatch = LSTMUtils.createIdentityMatrix(batchSize: 2, dim: 3)
        
        XCTAssertEqual(identityBatch.shape, [2, 3, 3])
        
        // Check that each matrix in the batch is an identity matrix
        for i in 0..<2 {
            let matrix = identityBatch[i]
            let expectedIdentity = MLX.eye(3)
            let isIdentity = MLX.allClose(matrix, expectedIdentity)
            XCTAssertTrue(isIdentity.item(Bool.self), "Matrix at batch index \(i) should be identity")
        }
    }
    
    // MARK: - Error Message Tests
    
    func testErrorDescriptions() {
        let shapeError = LSTMUtils.LSTMError.invalidTensorShape(expected: "[2, 3]", actual: "[2, 4]")
        XCTAssertTrue(shapeError.localizedDescription.contains("Invalid tensor shape"))
        
        let deviceError = LSTMUtils.LSTMError.deviceMismatch("CPU vs GPU")
        XCTAssertTrue(deviceError.localizedDescription.contains("Device mismatch"))
        
        let numericalError = LSTMUtils.LSTMError.numericalInstability("Division by zero")
        XCTAssertTrue(numericalError.localizedDescription.contains("Numerical instability"))
        
        let dimensionError = LSTMUtils.LSTMError.invalidDimension("Negative dimension")
        XCTAssertTrue(dimensionError.localizedDescription.contains("Invalid dimension"))
        
        let memoryError = LSTMUtils.LSTMError.memoryAllocationFailed("Out of memory")
        XCTAssertTrue(memoryError.localizedDescription.contains("Memory allocation failed"))
    }
}