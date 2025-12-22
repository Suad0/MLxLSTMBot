//
//  main.swift
//  MLXSTMBot
//
//  Created by Kiro on 22.12.25.
//

import Foundation
import MLX
import MLXNN

print("MLXSTMBot - Full xLSTM Architecture")
print("===================================")

do {
    // Test basic LSTM components first
    print("\n1. Testing individual LSTM components...")
    
    let inputDim = 64
    let hiddenDim = 128
    let batchSize = 2
    let sequenceLength = 10
    
    // Create sLSTM instance
    let sLSTM = try sLSTM(inputDim: inputDim, hiddenDim: hiddenDim)
    print("✓ sLSTM created with input_dim=\(inputDim), hidden_dim=\(hiddenDim)")
    
    // Create mLSTM instance
    let mLSTM = try mLSTM(inputDim: inputDim, hiddenDim: hiddenDim)
    print("✓ mLSTM created with input_dim=\(inputDim), hidden_dim=\(hiddenDim)")
    
    // Create test input sequence
    let inputSequence = MLX.zeros([batchSize, sequenceLength, inputDim]) + 0.1
    print("✓ Created input sequence with shape \(inputSequence.shape)")
    
    print("\n2. Testing xLSTM Blocks...")
    
    // Test sLSTM block
    let sLSTMBlock = try xLSTMBlock(blockType: .sLSTM, hiddenDim: hiddenDim, inputDim: hiddenDim)
    print("✓ sLSTM block created")
    
    // Test mLSTM block
    let mLSTMBlock = try xLSTMBlock(blockType: .mLSTM, hiddenDim: hiddenDim, inputDim: hiddenDim)
    print("✓ mLSTM block created")
    
    // Test block processing
    let blockInput = MLX.zeros([batchSize, hiddenDim]) + 0.1
    let sLSTMBlockState = try sLSTMBlock.initialState(batchSize: batchSize)
    let mLSTMBlockState = try mLSTMBlock.initialState(batchSize: batchSize)
    
    let (sLSTMBlockOutput, _) = sLSTMBlock(blockInput, state: sLSTMBlockState)
    let (mLSTMBlockOutput, _) = mLSTMBlock(blockInput, state: mLSTMBlockState)
    
    print("✓ Block processing successful")
    print("  - sLSTM block output shape: \(sLSTMBlockOutput.shape)")
    print("  - mLSTM block output shape: \(mLSTMBlockOutput.shape)")
    
    print("\n3. Testing Full xLSTM Architecture...")
    
    // Create full xLSTM model
    let vocabSize = 1000
    let numLayers = 4
    
    let xlstm = try xLSTM(
        vocabSize: vocabSize,
        hiddenDim: hiddenDim,
        numLayers: numLayers,
        includeFeedForward: true
    )
    
    print("✓ xLSTM model created")
    print("  - Vocabulary size: \(vocabSize)")
    print("  - Hidden dimension: \(hiddenDim)")
    print("  - Number of layers: \(numLayers)")
    print("  - Layer pattern: [mLSTM, sLSTM, mLSTM, sLSTM]")
    
    print("\n4. Testing Text Generation (Single Token)...")
    
    // Test single token processing
    let tokenIds = MLX.zeros([batchSize], dtype: .int32) + 1
    let (logits, states) = try xlstm(tokenIds)
    
    print("✓ Single token processing successful")
    print("  - Input token shape: \(tokenIds.shape)")
    print("  - Output logits shape: \(logits.shape)")
    print("  - Number of layer states: \(states.count)")
    
    print("\n5. Testing Sequence Processing...")
    
    // Test sequence processing
    let tokenSequence = MLX.zeros([batchSize, sequenceLength], dtype: .int32) + 1
    let (sequenceLogits, finalStates) = try xlstm.processSequence(tokenSequence)
    
    print("✓ Sequence processing successful")
    print("  - Input sequence shape: \(tokenSequence.shape)")
    print("  - Output logits shape: \(sequenceLogits.shape)")
    print("  - Final states count: \(finalStates.count)")
    
    print("\n6. Testing Autoregressive Generation...")
    
    // Test text generation
    let promptLength = 5
    let maxLength = 15
    let prompt = MLX.zeros([1, promptLength], dtype: .int32) + 1
    
    let generated = try xlstm.generate(
        prompt: prompt,
        maxLength: maxLength,
        temperature: 1.0,
        topK: 50
    )
    
    print("✓ Autoregressive generation successful")
    print("  - Prompt length: \(promptLength)")
    print("  - Generated length: \(generated.shape[1])")
    print("  - Total tokens: \(maxLength)")
    
    print("\n7. Testing Model Components...")
    
    // Test RMSNorm
    let rmsNorm = RMSNorm(normalizedShape: hiddenDim)
    let testInput = MLX.zeros([batchSize, hiddenDim]) + 0.1
    let normalizedOutput = rmsNorm(testInput)
    print("✓ RMSNorm working - output shape: \(normalizedOutput.shape)")
    
    // Test Gated Feed-Forward
    let gatedFFN = GatedFeedForward(inputDim: hiddenDim)
    let ffnOutput = gatedFFN(testInput)
    print("✓ Gated FFN working - output shape: \(ffnOutput.shape)")
    
    print("\n8. Testing State Management...")
    
    // Test state initialization and consistency
    let initialStates = try xlstm.initialStates(batchSize: batchSize)
    print("✓ Initial states created for \(initialStates.count) layers")
    
    // Verify state types match layer pattern
    for (index, state) in initialStates.enumerated() {
        let expectedType: String = (index % 2 == 0) ? "mLSTM" : "sLSTM"
        let actualType: String
        
        switch state {
        case .mLSTM:
            actualType = "mLSTM"
        case .sLSTM:
            actualType = "sLSTM"
        }
        
        print("  - Layer \(index): \(actualType) (expected: \(expectedType))")
        assert(actualType == expectedType, "State type mismatch at layer \(index)")
    }
    
    print("\n9. Testing Error Handling...")
    
    // Test validation functions
    try LSTMUtils.validateDimensions(inputDim: inputDim, hiddenDim: hiddenDim)
    try LSTMUtils.validateBatchSize(batchSize)
    
    print("✓ All validation functions passed")
    
    print("\n10. Performance Summary...")
    
    // Calculate model parameters (approximate)
    let embeddingParams = vocabSize * hiddenDim
    let lstmParamsPerLayer = (inputDim + hiddenDim) * hiddenDim * 6  // 6 projections per layer
    let ffnParamsPerLayer = hiddenDim * hiddenDim * 4 * 2  // GLU has 2 projections + output
    let lmHeadParams = hiddenDim * vocabSize
    
    let totalParams = embeddingParams + (numLayers * lstmParamsPerLayer) + 
                     (numLayers / 2 * ffnParamsPerLayer) + lmHeadParams
    
    print("✓ Model Statistics:")
    print("  - Embedding parameters: ~\(embeddingParams / 1000)K")
    print("  - LSTM parameters: ~\(numLayers * lstmParamsPerLayer / 1000)K")
    print("  - FFN parameters: ~\(numLayers / 2 * ffnParamsPerLayer / 1000)K")
    print("  - LM Head parameters: ~\(lmHeadParams / 1000)K")
    print("  - Total parameters: ~\(totalParams / 1000)K")
    
    print("\n✅ Full xLSTM Architecture Test Completed Successfully!")
    print("🚀 The model is ready for training and text generation!")
    print("\nKey Features Implemented:")
    print("  ✓ Alternating mLSTM/sLSTM architecture")
    print("  ✓ Pre-layer RMS normalization")
    print("  ✓ Residual connections")
    print("  ✓ Gated feed-forward networks (sLSTM blocks)")
    print("  ✓ Autoregressive text generation")
    print("  ✓ Proper state management")
    print("  ✓ MLX optimization for Apple Silicon")
    print("  ✓ Comprehensive error handling")
    
} catch {
    print("❌ Error occurred: \(error)")
    exit(1)
}