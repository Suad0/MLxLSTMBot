//
//  ChatDataProvider.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import Tokenizers

/// Data provider for loading and tokenizing JSON conversational data
/// 
/// This class handles loading JSON data in the format [{"content": "..."}]
/// and provides batched, tokenized data for training with proper padding/truncation.
public class ChatDataProvider {
    
    // MARK: - Properties
    
    /// The tokenizer instance from MLXLLM
    private let tokenizer: any Tokenizer
    
    /// Public access to pad token for loss masking
    public var padTokenId: Int32 {
        return Int32(tokenizer.eosTokenId ?? 2)
    }
    
    /// Raw text data loaded from JSON
    private var textData: [String] = []
    
    /// Current position in the dataset
    private var currentIndex: Int = 0
    
    /// Whether to shuffle data between epochs
    private let shuffle: Bool
    
    /// Random number generator for shuffling
    private var rng = SystemRandomNumberGenerator()
    
    // MARK: - Initialization
    
    /// Initializes the data provider
    /// - Parameters:
    ///   - jsonPath: Path to the JSON file containing conversational data
    ///   - tokenizer: Tokenizer instance from MLXLLM
    ///   - shuffle: Whether to shuffle data between epochs (default: true)
    /// - Throws: Error if file cannot be loaded or parsed
    public init(jsonPath: String, tokenizer: any Tokenizer, shuffle: Bool = true) throws {
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        
        try loadData(from: jsonPath)
        
        if shuffle {
            shuffleData()
        }
        
        print("ChatDataProvider initialized with \(textData.count) samples")
    }
    
    // MARK: - Data Loading
    
    /// Loads and parses JSON data from file
    /// - Parameter path: Path to the JSON file
    /// - Throws: Error if file cannot be loaded or parsed
    private func loadData(from path: String) throws {
        // Load JSON file
        let url = URL(fileURLWithPath: path)
        let data = try Data(contentsOf: url)
        
        // Parse JSON
        guard let jsonArray = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw DataProviderError.invalidJSONFormat("Expected array of objects")
        }
        
        // Extract content strings
        var loadedTexts: [String] = []
        for item in jsonArray {
            guard let content = item["content"] as? String else {
                throw DataProviderError.missingContentField("Missing 'content' field in JSON object")
            }
            
            // Skip empty content
            if !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                loadedTexts.append(content)
            }
        }
        
        guard !loadedTexts.isEmpty else {
            throw DataProviderError.emptyDataset("No valid content found in JSON file")
        }
        
        self.textData = loadedTexts
    }
    
    /// Shuffles the data for the next epoch
    private func shuffleData() {
        textData.shuffle(using: &rng)
        currentIndex = 0
    }
    
    // MARK: - Batch Generation
    
    /// Generates the next batch of tokenized data
    /// - Parameters:
    ///   - batchSize: Number of samples in the batch
    ///   - seqLen: Target sequence length (will pad/truncate to this length)
    /// - Returns: Tuple of (inputs, targets) where inputs are tokens 0 to N-1 and targets are tokens 1 to N
    /// - Throws: Error if tokenization fails
    public func nextBatch(batchSize: Int, seqLen: Int) throws -> (inputs: MLXArray, targets: MLXArray) {
        var inputBatch: [[Int32]] = []
        var targetBatch: [[Int32]] = []
        
        for _ in 0..<batchSize {
            // Get next text sample (with wraparound)
            if currentIndex >= textData.count {
                if shuffle {
                    shuffleData()
                } else {
                    currentIndex = 0
                }
            }
            
            let text = textData[currentIndex]
            currentIndex += 1
            
            // Tokenize the text
            let tokens = try tokenizeText(text)
            
            // Create input/target sequences with proper length handling
            let (inputSeq, targetSeq) = createSequencePair(from: tokens, targetLength: seqLen)
            
            inputBatch.append(inputSeq)
            targetBatch.append(targetSeq)
        }
        
        // Convert to MLXArrays
        let inputs = try createMLXArray(from: inputBatch)
        let targets = try createMLXArray(from: targetBatch)
        
        return (inputs: inputs, targets: targets)
    }
    
    /// Tokenizes text using the provided tokenizer
    /// - Parameter text: Input text to tokenize
    /// - Returns: Array of token IDs
    /// - Throws: Error if tokenization fails
    private func tokenizeText(_ text: String) throws -> [Int32] {
        // Use the tokenizer to encode the text
        let encoded = tokenizer.encode(text: text)
        
        // Convert to Int32 array
        return encoded.map { Int32($0) }
    }
    
    /// Creates input/target sequence pair from tokens
    /// - Parameters:
    ///   - tokens: Original token sequence
    ///   - targetLength: Desired sequence length
    /// - Returns: Tuple of (input_sequence, target_sequence)
    private func createSequencePair(from tokens: [Int32], targetLength: Int) -> ([Int32], [Int32]) {
        var processedTokens = tokens
        let requiredLength = targetLength + 1
        
        // Handle sequence length
        if processedTokens.count < requiredLength {
            // Pad using EOS token
            let padToken = self.padTokenId
            let paddingNeeded = requiredLength - processedTokens.count
            processedTokens.append(contentsOf: Array(repeating: padToken, count: paddingNeeded))
        } else if processedTokens.count > requiredLength {
            // Truncate to required length
            processedTokens = Array(processedTokens.prefix(requiredLength))
        }
        
        // Create input (tokens 0 to N-1) and target (tokens 1 to N) sequences
        let finalInputSeq = Array(processedTokens.prefix(targetLength))
        let finalTargetSeq = Array(processedTokens.dropFirst().prefix(targetLength))
        
        return (finalInputSeq, finalTargetSeq)
    }
    
    /// Converts 2D array to MLXArray
    /// - Parameter batch: 2D array of token sequences
    /// - Returns: MLXArray with shape [batch_size, seq_len]
    /// - Throws: Error if conversion fails
    private func createMLXArray(from batch: [[Int32]]) throws -> MLXArray {
        guard !batch.isEmpty else {
            throw DataProviderError.emptyBatch("Cannot create MLXArray from empty batch")
        }
        
        let batchSize = batch.count
        let seqLen = batch[0].count
        
        // Verify all sequences have the same length
        for (index, sequence) in batch.enumerated() {
            guard sequence.count == seqLen else {
                throw DataProviderError.inconsistentSequenceLength(
                    "Sequence at index \(index) has length \(sequence.count), expected \(seqLen)"
                )
            }
        }
        
        // Flatten the batch into a single array
        let flattenedData = batch.flatMap { $0 }
        
        // Create MLXArray
        let mlxArray = MLXArray(flattenedData)
        return mlxArray.reshaped([batchSize, seqLen])
    }
    
    // MARK: - Utility Methods
    
    /// Returns the total number of samples in the dataset
    public var datasetSize: Int {
        return textData.count
    }
    
    /// Resets the data provider to the beginning
    public func reset() {
        currentIndex = 0
        if shuffle {
            shuffleData()
        }
    }
    
    /// Returns the current progress through the dataset (0.0 to 1.0)
    public var progress: Double {
        return Double(currentIndex) / Double(textData.count)
    }
}

// MARK: - Error Types

public enum DataProviderError: Error, LocalizedError {
    case invalidJSONFormat(String)
    case missingContentField(String)
    case emptyDataset(String)
    case emptyBatch(String)
    case inconsistentSequenceLength(String)
    case tokenizationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidJSONFormat(let message):
            return "Invalid JSON format: \(message)"
        case .missingContentField(let message):
            return "Missing content field: \(message)"
        case .emptyDataset(let message):
            return "Empty dataset: \(message)"
        case .emptyBatch(let message):
            return "Empty batch: \(message)"
        case .inconsistentSequenceLength(let message):
            return "Inconsistent sequence length: \(message)"
        case .tokenizationFailed(let message):
            return "Tokenization failed: \(message)"
        }
    }
}

// MARK: - Extensions

extension ChatDataProvider {
    
    /// Creates a sample JSON file for testing
    /// - Parameter path: Path where to save the sample file
    /// - Throws: Error if file cannot be written
    public static func createSampleData(at path: String) throws {
        let sampleData = [
            ["content": "Hello, how are you today?"],
            ["content": "I'm doing well, thank you for asking. How can I help you?"],
            ["content": "Can you explain what machine learning is?"],
            ["content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."],
            ["content": "That's a great explanation! Can you give me an example?"],
            ["content": "Sure! A common example is email spam detection. The system learns to identify spam by analyzing thousands of emails and their classifications."],
            ["content": "What are the main types of machine learning?"],
            ["content": "The main types are supervised learning, unsupervised learning, and reinforcement learning. Each has different applications and approaches."],
            ["content": "Thank you for the explanation!"],
            ["content": "You're welcome! Feel free to ask if you have any more questions about machine learning or AI."]
        ]
        
        let jsonData = try JSONSerialization.data(withJSONObject: sampleData, options: .prettyPrinted)
        let url = URL(fileURLWithPath: path)
        try jsonData.write(to: url)
        
        print("Sample data created at: \(path)")
    }
}
