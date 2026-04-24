//
//  xLSTM.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Main xLSTM Architecture
/// 
/// This class implements the complete xLSTM architecture with:
/// - Token embedding layer
/// - Configurable sequence of mLSTM and sLSTM blocks
/// - Language modeling head for text generation
/// - Proper state management for autoregressive generation
public class xLSTM: Module {
    
    // MARK: - Properties
    
    public let vocabSize: Int
    public let hiddenDim: Int
    public let numLayers: Int
    
    public let embedding: Embedding
    public let blocks: [xLSTMBlock]
    public let finalNorm: LayerNorm
    public let lmHead: Linear
    
    // MARK: - Initialization
    
    /// Initializes the xLSTM model
    /// - Parameters:
    ///   - vocabSize: Size of the vocabulary
    ///   - hiddenDim: Hidden dimension for all layers
    ///   - blockSpec: Array specifying the type of each block in order
    /// - Throws: LSTMError for invalid configurations
    public init(vocabSize: Int, hiddenDim: Int, blockSpec: [xLSTMBlockType]) throws {
        guard vocabSize > 0 else {
            throw LSTMUtils.LSTMError.invalidDimension("Vocabulary size must be positive, got \(vocabSize)")
        }
        guard !blockSpec.isEmpty else {
            throw LSTMUtils.LSTMError.invalidDimension("Block specification array cannot be empty")
        }
        try LSTMUtils.validateDimensions(inputDim: hiddenDim, hiddenDim: hiddenDim)
        
        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim
        self.numLayers = blockSpec.count
        
        // Initialize embedding layer
        self.embedding = Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
        
        // Initialize xLSTM blocks using blockSpec
        var blocks: [xLSTMBlock] = []
        blocks.reserveCapacity(numLayers)
        
        for (_, blockType) in blockSpec.enumerated() {
            let inputDim = hiddenDim
            let block = try xLSTMBlock(
                blockType: blockType,
                hiddenDim: hiddenDim,
                inputDim: inputDim,
                includeFeedForward: true // true for sLSTM usually, false for mLSTM internally handled
            )
            blocks.append(block)
        }
        self.blocks = blocks
        
        // Initialize final normalization and language modeling head
        self.finalNorm = LayerNorm(dimensions: hiddenDim, eps: 1e-5)
        self.lmHead = Linear(hiddenDim, vocabSize)
        
        super.init()
        initializeWeights()
    }
    
    private func initializeWeights() {
        _ = 1.0 / sqrt(Float(hiddenDim))
    }
    
    // MARK: - State Management
    
    public func initialStates(batchSize: Int) throws -> [LayerState] {
        try LSTMUtils.validateBatchSize(batchSize)
        var states: [LayerState] = []
        states.reserveCapacity(numLayers)
        for block in blocks {
            let state = try block.initialState(batchSize: batchSize)
            states.append(state)
        }
        return states
    }
    
    // MARK: - Forward Pass
    
    public func callAsFunction(_ tokenIds: MLXArray, states: [LayerState]? = nil) throws -> (MLXArray, [LayerState]) {
        let input: MLXArray
        if tokenIds.ndim == 1 {
            input = tokenIds.expandedDimensions(axis: 1)
        } else if tokenIds.ndim == 2 && tokenIds.shape[1] == 1 {
            input = tokenIds
        } else {
            throw LSTMUtils.LSTMError.invalidTensorShape(expected: "[batch_size] or [batch_size, 1]", actual: "\(tokenIds.shape)")
        }
        
        let batchSize = input.shape[0]
        
        var currentStates: [LayerState]
        if let states = states {
            guard states.count == numLayers else {
                throw LSTMUtils.LSTMError.invalidDimension("Expected \(numLayers) states, got \(states.count)")
            }
            currentStates = states
        } else {
            currentStates = try initialStates(batchSize: batchSize)
        }
        
        let embedded = embedding(input)
        var x = embedded.squeezed(axis: 1)
        
        var newStates: [LayerState] = []
        newStates.reserveCapacity(numLayers)
        
        for (blockIndex, block) in blocks.enumerated() {
            let currentState = currentStates[blockIndex]
            let (output, newState) = try block(x, state: currentState)
            x = output
            newStates.append(newState)
        }
        
        x = finalNorm(x)
        let logits = lmHead(x)
        
        return (logits, newStates)
    }
    
    public func processSequence(_ tokenIds: MLXArray, states: [LayerState]? = nil) throws -> (MLXArray, [LayerState]) {
        guard tokenIds.ndim == 2 else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "2D tensor [batch_size, sequence_length]",
                actual: "\(tokenIds.ndim)D tensor with shape \(tokenIds.shape)"
            )
        }
        
        let batchSize = tokenIds.shape[0]
        let sequenceLength = tokenIds.shape[1]
        try LSTMUtils.validateSequenceLength(sequenceLength)
        
        var currentStates: [LayerState]
        if let states = states {
            guard states.count == numLayers else {
                throw LSTMUtils.LSTMError.invalidDimension("Expected \(numLayers) states, got \(states.count)")
            }
            currentStates = states
        } else {
            currentStates = try initialStates(batchSize: batchSize)
        }
        
        let embedded = embedding(tokenIds)
        var x = embedded
        var newStates: [LayerState] = []
        newStates.reserveCapacity(numLayers)
        
        for (blockIndex, block) in blocks.enumerated() {
            let currentState = currentStates[blockIndex]
            let (output, newState) = try block.processSequence(x, initialState: currentState)
            x = output
            newStates.append(newState)
        }
        
        x = finalNorm(x)
        let logits = lmHead(x)
        
        return (logits, newStates)
    }
    
    // MARK: - Text Generation
    
    public func generate(
        prompt: MLXArray,
        maxLength: Int,
        temperature: Float = 1.0,
        topK: Int? = nil
    ) throws -> MLXArray {
        guard prompt.ndim == 2 else {
            throw LSTMUtils.LSTMError.invalidTensorShape(expected: "2D tensor [batch_size, prompt_length]", actual: "\(prompt.ndim)D")
        }
        
        let promptLength = prompt.shape[1]
        guard maxLength > promptLength else {
            throw LSTMUtils.LSTMError.invalidDimension("maxLength (\(maxLength)) must be greater than prompt length (\(promptLength))")
        }
        
        let (_, initialStates) = try processSequence(prompt)
        
        var generatedTokens = prompt
        var currentStates = initialStates
        var lastToken = prompt[0..., -1]
        
        for _ in promptLength..<maxLength {
            let (logits, newStates) = try callAsFunction(lastToken, states: currentStates)
            
            let scaledLogits = logits / temperature
            let finalLogits: MLXArray
            if let k = topK {
                finalLogits = applyTopK(scaledLogits, k: k)
            } else {
                finalLogits = scaledLogits
            }
            
            // Bug B15 Fix: categorical sampling instead of argmax
            let nextToken = sampleFromLogits(finalLogits)
            generatedTokens = MLX.concatenated([generatedTokens, nextToken.expandedDimensions(axis: 1)], axis: 1)
            
            lastToken = nextToken
            currentStates = newStates
        }
        
        return generatedTokens
    }
    
    private func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
        let vocabSize = logits.shape[1]
        let actualK = min(k, vocabSize)
        
        let sortedTokens = MLX.sorted(logits, axis: -1)
        let kthIndex = vocabSize - actualK
        let kth = sortedTokens[0..., kthIndex...kthIndex]
        
        return MLX.where(logits .>= kth, logits, MLXArray(-1e9))
    }
    
    private func sampleFromLogits(_ logits: MLXArray) -> MLXArray {
        // Evaluate categorical sampling correctly for multinomial sampling
        return MLXRandom.categorical(logits)
    }
}

// MARK: - Convenience Extensions

extension xLSTM {
    public func initialStates() -> [LayerState] {
        do {
            return try initialStates(batchSize: 1)
        } catch {
            fatalError("Failed to create initial states: \(error)")
        }
    }
    
    public func generateSingle(
        promptTokens: MLXArray,
        maxLength: Int,
        temperature: Float = 1.0,
        topK: Int? = nil
    ) -> MLXArray {
        do {
            let batchedPrompt = promptTokens.expandedDimensions(axis: 0)
            let generated = try generate(prompt: batchedPrompt, maxLength: maxLength, temperature: temperature, topK: topK)
            return generated.squeezed(axis: 0)
        } catch {
            fatalError("Generation failed: \(error)")
        }
    }
}
