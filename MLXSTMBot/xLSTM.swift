//
//  xLSTM.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Main xLSTM Architecture
/// 
/// This class implements the complete xLSTM architecture with:
/// - Token embedding layer
/// - Alternating mLSTM and sLSTM blocks
/// - Language modeling head for text generation
/// - Proper state management for autoregressive generation
/// 
/// The architecture follows the pattern: [mLSTM, sLSTM, mLSTM, sLSTM, ...]
/// where layers alternate between mLSTM and sLSTM blocks.
public class xLSTM: Module {
    
    // MARK: - Properties
    
    /// Vocabulary size
    public let vocabSize: Int
    
    /// Hidden dimension
    public let hiddenDim: Int
    
    /// Number of layers
    public let numLayers: Int
    
    /// Token embedding layer
    public let embedding: Embedding
    
    /// xLSTM blocks (alternating mLSTM and sLSTM)
    public let blocks: [xLSTMBlock]
    
    /// Final layer normalization
    public let finalNorm: RMSNorm
    
    /// Language modeling head (projects to vocabulary)
    public let lmHead: Linear
    
    // MARK: - Initialization
    
    /// Initializes the xLSTM model
    /// - Parameters:
    ///   - vocabSize: Size of the vocabulary
    ///   - hiddenDim: Hidden dimension for all layers
    ///   - numLayers: Number of xLSTM layers
    ///   - includeFeedForward: Whether to include FFN in sLSTM blocks (default: true)
    /// - Throws: LSTMError for invalid configurations
    public init(vocabSize: Int, hiddenDim: Int, numLayers: Int, includeFeedForward: Bool = true) throws {
        // Validate parameters
        guard vocabSize > 0 else {
            throw LSTMUtils.LSTMError.invalidDimension("Vocabulary size must be positive, got \(vocabSize)")
        }
        
        guard numLayers > 0 else {
            throw LSTMUtils.LSTMError.invalidDimension("Number of layers must be positive, got \(numLayers)")
        }
        
        try LSTMUtils.validateDimensions(inputDim: hiddenDim, hiddenDim: hiddenDim)
        
        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        
        // Initialize embedding layer
        self.embedding = Embedding(embeddingCount: vocabSize, dimensions: hiddenDim)
        
        // Initialize xLSTM blocks with alternating pattern
        var blocks: [xLSTMBlock] = []
        blocks.reserveCapacity(numLayers)
        
        for layerIndex in 0..<numLayers {
            // Alternate between mLSTM and sLSTM
            // Pattern: [mLSTM, sLSTM, mLSTM, sLSTM, ...]
            let blockType: xLSTMBlockType = (layerIndex % 2 == 0) ? .mLSTM : .sLSTM
            
            // First layer takes embedding dimension as input, others take hiddenDim
            let inputDim = (layerIndex == 0) ? hiddenDim : hiddenDim
            
            let block = try xLSTMBlock(
                blockType: blockType,
                hiddenDim: hiddenDim,
                inputDim: inputDim,
                includeFeedForward: includeFeedForward
            )
            
            blocks.append(block)
        }
        
        self.blocks = blocks
        
        // Initialize final normalization and language modeling head
        self.finalNorm = RMSNorm(normalizedShape: hiddenDim)
        self.lmHead = Linear(hiddenDim, vocabSize)
        
        super.init()
        
        // Apply scaled initialization for stability
        initializeWeights()
    }
    
    /// Initialize weights with scaled variance for training stability
    private func initializeWeights() {
        // Scale factor based on hidden dimension (similar to scaled_dot_product_attention)
        _ = 1.0 / sqrt(Float(hiddenDim))
        
        // Note: MLX handles weight initialization automatically for most layers
        // This method can be extended for custom initialization if needed
        
        // The embedding and linear layers in MLX are already initialized with appropriate scales
        // Additional custom initialization can be added here if required
    }
    
    // MARK: - State Management
    
    /// Creates initial states for all layers
    /// - Parameter batchSize: Batch size
    /// - Returns: Array of initial layer states
    /// - Throws: LSTMError for invalid batch size
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
    
    /// Forward pass for text generation (single timestep)
    /// 
    /// This method processes a single token and updates all layer states.
    /// It's designed for autoregressive text generation where tokens are processed one at a time.
    /// 
    /// - Parameters:
    ///   - tokenIds: Input token IDs [batch_size] or [batch_size, 1]
    ///   - states: Current states for all layers (optional, creates initial states if nil)
    /// - Returns: Tuple of (logits, new_states) where logits are [batch_size, vocab_size]
    /// - Throws: LSTMError for invalid inputs
    public func callAsFunction(_ tokenIds: MLXArray, states: [LayerState]? = nil) throws -> (MLXArray, [LayerState]) {
        // Validate and reshape input
        let input: MLXArray
        if tokenIds.ndim == 1 {
            // Single token per batch: [batch_size] -> [batch_size, 1]
            input = tokenIds.expandedDimensions(axis: 1)
        } else if tokenIds.ndim == 2 && tokenIds.shape[1] == 1 {
            // Already correct shape: [batch_size, 1]
            input = tokenIds
        } else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "[batch_size] or [batch_size, 1]",
                actual: "\(tokenIds.shape)"
            )
        }
        
        let batchSize = input.shape[0]
        
        // Use provided states or create initial states
        var currentStates: [LayerState]
        if let states = states {
            guard states.count == numLayers else {
                throw LSTMUtils.LSTMError.invalidDimension(
                    "Expected \(numLayers) states, got \(states.count)"
                )
            }
            currentStates = states
        } else {
            currentStates = try initialStates(batchSize: batchSize)
        }
        
        // Token embedding: [batch_size, 1] -> [batch_size, 1, hidden_dim]
        let embedded = embedding(input)
        
        // Remove sequence dimension for single timestep processing: [batch_size, hidden_dim]
        var x = embedded.squeezed(axis: 1)
        
        // Process through all xLSTM blocks
        var newStates: [LayerState] = []
        newStates.reserveCapacity(numLayers)
        
        for (blockIndex, block) in blocks.enumerated() {
            let currentState = currentStates[blockIndex]
            let (output, newState) = block(x, state: currentState)
            x = output
            newStates.append(newState)
        }
        
        // Final normalization
        x = finalNorm(x)
        
        // Language modeling head: [batch_size, hidden_dim] -> [batch_size, vocab_size]
        let logits = lmHead(x)
        
        return (logits, newStates)
    }
    
    /// Forward pass for sequence processing (training mode)
    /// 
    /// This method processes a full sequence of tokens, typically used during training.
    /// 
    /// - Parameters:
    ///   - tokenIds: Input token sequence [batch_size, sequence_length]
    ///   - states: Optional initial states for all layers
    /// - Returns: Tuple of (logits, final_states) where logits are [batch_size, sequence_length, vocab_size]
    /// - Throws: LSTMError for invalid inputs
    public func processSequence(_ tokenIds: MLXArray, states: [LayerState]? = nil) throws -> (MLXArray, [LayerState]) {
        // Validate input
        guard tokenIds.ndim == 2 else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "2D tensor [batch_size, sequence_length]",
                actual: "\(tokenIds.ndim)D tensor with shape \(tokenIds.shape)"
            )
        }
        
        let batchSize = tokenIds.shape[0]
        let sequenceLength = tokenIds.shape[1]
        
        // Validate sequence length
        try LSTMUtils.validateSequenceLength(sequenceLength)
        
        // Use provided states or create initial states
        var currentStates: [LayerState]
        if let states = states {
            guard states.count == numLayers else {
                throw LSTMUtils.LSTMError.invalidDimension(
                    "Expected \(numLayers) states, got \(states.count)"
                )
            }
            currentStates = states
        } else {
            currentStates = try initialStates(batchSize: batchSize)
        }
        
        // Token embedding: [batch_size, sequence_length] -> [batch_size, sequence_length, hidden_dim]
        let embedded = embedding(tokenIds)
        
        // Process through all xLSTM blocks
        var x = embedded
        var newStates: [LayerState] = []
        newStates.reserveCapacity(numLayers)
        
        for (blockIndex, block) in blocks.enumerated() {
            let currentState = currentStates[blockIndex]
            let (output, newState) = try block.processSequence(x, initialState: currentState)
            x = output
            newStates.append(newState)
        }
        
        // Final normalization: [batch_size, sequence_length, hidden_dim]
        x = finalNorm(x)
        
        // Language modeling head: [batch_size, sequence_length, hidden_dim] -> [batch_size, sequence_length, vocab_size]
        let logits = lmHead(x)
        
        return (logits, newStates)
    }
    
    // MARK: - Text Generation
    
    /// Generate text autoregressively
    /// 
    /// This method generates text by sampling from the model's output distribution.
    /// It processes tokens one at a time and maintains state across timesteps.
    /// 
    /// - Parameters:
    ///   - prompt: Initial prompt tokens [batch_size, prompt_length]
    ///   - maxLength: Maximum number of tokens to generate
    ///   - temperature: Sampling temperature (default: 1.0)
    ///   - topK: Top-k sampling parameter (default: nil for no top-k)
    /// - Returns: Generated token sequence [batch_size, total_length]
    /// - Throws: LSTMError for invalid inputs
    public func generate(
        prompt: MLXArray,
        maxLength: Int,
        temperature: Float = 1.0,
        topK: Int? = nil
    ) throws -> MLXArray {
        guard prompt.ndim == 2 else {
            throw LSTMUtils.LSTMError.invalidTensorShape(
                expected: "2D tensor [batch_size, prompt_length]",
                actual: "\(prompt.ndim)D tensor with shape \(prompt.shape)"
            )
        }
        
        let _ = prompt.shape[0]
        let promptLength = prompt.shape[1]
        
        guard maxLength > promptLength else {
            throw LSTMUtils.LSTMError.invalidDimension(
                "maxLength (\(maxLength)) must be greater than prompt length (\(promptLength))"
            )
        }
        
        // Process prompt to get initial states
        let (_, initialStates) = try processSequence(prompt)
        
        // Initialize generation
        var generatedTokens = prompt
        var currentStates = initialStates
        var lastToken = prompt[0..., -1]  // Last token of prompt for each batch
        
        // Generate tokens autoregressively
        for _ in promptLength..<maxLength {
            // Forward pass for single timestep
            let (logits, newStates) = try callAsFunction(lastToken, states: currentStates)
            
            // Apply temperature scaling
            let scaledLogits = logits / temperature
            
            // Apply top-k filtering if specified
            let finalLogits: MLXArray
            if let k = topK {
                finalLogits = applyTopK(scaledLogits, k: k)
            } else {
                finalLogits = scaledLogits
            }
            
            // Sample next token
            let nextToken = sampleFromLogits(finalLogits)
            
            // Append to generated sequence
            generatedTokens = MLX.concatenated([generatedTokens, nextToken.expandedDimensions(axis: 1)], axis: 1)
            
            // Update for next iteration
            lastToken = nextToken
            currentStates = newStates
        }
        
        return generatedTokens
    }
    
    /// Apply top-k filtering to logits
    private func applyTopK(_ logits: MLXArray, k: Int) -> MLXArray {
        let vocabSize = logits.shape[1]
        let actualK = min(k, vocabSize)
        
        let sortedTokens = MLX.sorted(logits, axis: -1)
        let kthIndex = vocabSize - actualK
        let kth = sortedTokens[0..., kthIndex...kthIndex]
        
        return MLX.where(logits .>= kth, logits, MLXArray(-1e9))
    }
    
    /// Sample tokens from logits using multinomial sampling
    private func sampleFromLogits(_ logits: MLXArray) -> MLXArray {
        // Convert logits to probabilities
        let probs = MLX.softmax(logits, axis: -1)
        
        // Simple sampling: take argmax for now (can be improved with proper sampling)
        return MLX.argMax(probs, axis: -1)
    }
}

// MARK: - Convenience Extensions

extension xLSTM {
    
    /// Creates initial states with default batch size of 1
    /// - Returns: Initial states for single sample
    public func initialStates() -> [LayerState] {
        do {
            return try initialStates(batchSize: 1)
        } catch {
            fatalError("Failed to create initial states: \(error)")
        }
    }
    
    /// Generate text from a single prompt (convenience method)
    /// - Parameters:
    ///   - promptTokens: 1D array of prompt token IDs
    ///   - maxLength: Maximum length to generate
    ///   - temperature: Sampling temperature
    ///   - topK: Top-k sampling parameter
    /// - Returns: 1D array of generated tokens
    public func generateSingle(
        promptTokens: MLXArray,
        maxLength: Int,
        temperature: Float = 1.0,
        topK: Int? = nil
    ) -> MLXArray {
        do {
            // Add batch dimension
            let batchedPrompt = promptTokens.expandedDimensions(axis: 0)
            
            // Generate
            let generated = try generate(
                prompt: batchedPrompt,
                maxLength: maxLength,
                temperature: temperature,
                topK: topK
            )
            
            // Remove batch dimension
            return generated.squeezed(axis: 0)
        } catch {
            fatalError("Generation failed: \(error)")
        }
    }
}
