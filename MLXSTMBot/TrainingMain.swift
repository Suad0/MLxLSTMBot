//
//  TrainingMain.swift
//  MLXSTMBot
//
//  Created by Kiro on 22.12.25.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import Tokenizers

/// Main training execution for xLSTM knowledge distillation
public class TrainingMain {
    
    // MARK: - Configuration
    
    struct TrainingConfig {
        let teacherModelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        let vocabSize = 128256  // Match teacher vocabulary
        let hiddenDim = 512
        let numLayers = 6
        let batchSize = 4
        let sequenceLength = 256
        let numIterations = 1
        let learningRate: Float = 1e-8  // Even smaller learning rate
        let printEvery = 1  // Print every step for debugging
        let saveEvery = 10
        let dataPath = "training_data.json"
        let outputPath = "xLSTM_Final.safetensors"
        let statsPath = "training_stats.json"
    }
    
    // MARK: - Properties
    
    private let config = TrainingConfig()
    private var dataProvider: ChatDataProvider?
    private var trainer: DistillationTrainer?
    private var studentModel: xLSTM?
    private var teacherModel: (any LanguageModel)?
    
    // MARK: - Main Training Function
    
    /// Runs the complete training pipeline
    public func runTraining() async {
        print("🚀 Starting xLSTM Knowledge Distillation Training")
        print("=" * 60)
        
        do {
            // Step 1: Initialize models
            try await initializeModels()
            
            // Step 2: Setup data provider
            try setupDataProvider()
            
            // Step 3: Initialize trainer
            initializeTrainer()
            
            // Step 4: Run training loop
            try runTrainingLoop()
            
            // Step 5: Save final model
            try saveModel()
            
            print("\n✅ Training completed successfully!")
            
        } catch {
            print("\n❌ Training failed with error: \(error)")
            exit(1)
        }
    }
    
    // MARK: - Initialization
    
    /// Initializes teacher and student models
    private func initializeModels() async throws {
        print("\n📚 Initializing Models...")
        
        // Load teacher model (Llama-3.2-1B-Instruct-4bit)
        print("Loading teacher model: \(config.teacherModelId)")
        let modelFactory = LLMModelFactory.shared
        let configuration = ModelConfiguration(id: config.teacherModelId)
        
        let container = try await modelFactory.loadContainer(configuration: configuration) { progress in
            let percent = String(format: "%.1f", progress.fractionCompleted * 100)
            print("\rTeacher model download: \(percent)%", terminator: "")
            fflush(stdout)
        }
        
        print("\n✓ Teacher model loaded successfully")
        
        // Enable Teacher Model
        self.teacherModel = await container.perform { $0.model }
        if self.teacherModel == nil {
             print("Warning: Could not load teacher model. Distillation might fail.")
        }

        // Initialize student xLSTM model
        print("Initializing student xLSTM model...")
        self.studentModel = try xLSTM(
            vocabSize: config.vocabSize,
            hiddenDim: config.hiddenDim,
            numLayers: config.numLayers,
            includeFeedForward: true
        )
        
        print("✓ Student xLSTM model initialized")
        print("  - Vocabulary size: \(config.vocabSize)")
        print("  - Hidden dimension: \(config.hiddenDim)")
        print("  - Number of layers: \(config.numLayers)")
        
        // Calculate and display model parameters
        displayModelStats()
    }
    
    /// Sets up the data provider
    private func setupDataProvider() throws {
        print("\n📊 Setting up Data Provider...")
        
        // Create sample data if it doesn't exist
        if !FileManager.default.fileExists(atPath: config.dataPath) {
            print("Creating sample training data...")
            try ChatDataProvider.createSampleData(at: config.dataPath)
        }
        
        // We need a tokenizer - for now we'll create a simple placeholder
        // In practice, you'd get this from the teacher model
        let tokenizer = SimpleTokenizer(vocabSize: config.vocabSize)
        
        self.dataProvider = try ChatDataProvider(
            jsonPath: config.dataPath,
            tokenizer: tokenizer,
            shuffle: true
        )
        
        print("✓ Data provider initialized")
        print("  - Dataset size: \(dataProvider?.datasetSize ?? 0) samples")
        print("  - Batch size: \(config.batchSize)")
        print("  - Sequence length: \(config.sequenceLength)")
    }
    
    /// Initializes the distillation trainer
    private func initializeTrainer() {
        print("\n🎯 Initializing Distillation Trainer...")
        
        guard let student = studentModel else {
            fatalError("Student model not initialized")
        }
        
        self.trainer = DistillationTrainer(
            studentModel: student,
            teacherModel: self.teacherModel,
            learningRate: config.learningRate,
            temperature: 2.0,
            distillationWeight: 0.7,
            groundTruthWeight: 0.3
        )
        
        print("✓ Distillation trainer initialized (student-only mode)")
    }
    
    // MARK: - Training Loop
    
    /// Runs the main training loop
    private func runTrainingLoop() throws {
        print("\n🏃‍♂️ Starting Training Loop...")
        print("Training for \(config.numIterations) iterations")
        print("-" * 50)
        
        guard let trainer = trainer, let dataProvider = dataProvider else {
            throw TrainingError.initializationFailed("Trainer or data provider not initialized")
        }
        
        var totalLoss: Float = 0.0
        let startTime = Date()
        
        for iteration in 1...config.numIterations {
            // Get next batch
            let (inputs, targets) = try dataProvider.nextBatch(
                batchSize: config.batchSize,
                seqLen: config.sequenceLength
            )
            
            // Perform training step
            let loss = try trainer.trainingStep(inputs: inputs, targets: targets)
            totalLoss += loss
            
            // Print progress
            if iteration % config.printEvery == 0 {
                let avgLoss = totalLoss / Float(config.printEvery)
                let elapsed = Date().timeIntervalSince(startTime)
                let stepsPerSec = Float(iteration) / Float(elapsed)
                
                print(String(format: "Step %4d | Loss: %.4f | Avg Loss: %.4f | Steps/sec: %.2f | Progress: %.1f%%",
                            iteration, loss, avgLoss, stepsPerSec, 
                            Float(iteration) / Float(config.numIterations) * 100))
                
                totalLoss = 0.0
            }
            
            // Save checkpoint
            if iteration % config.saveEvery == 0 {
                try saveCheckpoint(iteration: iteration)
            }
        }
        
        let totalTime = Date().timeIntervalSince(startTime)
        print("\n✓ Training completed in \(String(format: "%.2f", totalTime)) seconds")
        print("  - Final loss: \(String(format: "%.4f", trainer.losses.last ?? 0.0))")
        print("  - Average loss (last 10): \(String(format: "%.4f", trainer.averageLoss()))")
    }
    
    // MARK: - Model Saving
    
    /// Saves the final trained model
    private func saveModel() throws {
        print("\n💾 Saving Final Model...")
        
        guard let student = studentModel else {
            throw TrainingError.saveFailed("Student model not initialized")
        }
        
        // Save model weights using MLX save function
        let modelDict = Dictionary(uniqueKeysWithValues: student.parameters().flattened())
        try MLX.save(arrays: modelDict, url: URL(fileURLWithPath: config.outputPath))
        
        print("✓ Model saved to: \(config.outputPath)")
        
        // Save training statistics
        try trainer?.saveTrainingStats(to: config.statsPath)
        
        // Display final model info
        displayFinalModelInfo()
    }
    
    /// Saves a training checkpoint
    private func saveCheckpoint(iteration: Int) throws {
        let checkpointPath = "checkpoint_\(iteration).safetensors"
        
        guard let student = studentModel else { return }
        
        let modelDict = Dictionary(uniqueKeysWithValues: student.parameters().flattened())
        try MLX.save(arrays: modelDict, url: URL(fileURLWithPath: checkpointPath))
        
        print("  💾 Checkpoint saved: \(checkpointPath)")
    }
    
    // MARK: - Utility Methods
    
    /// Displays model statistics
    private func displayModelStats() {
        guard let student = studentModel else { return }
        
        // Calculate approximate parameter count
        let embeddingParams = config.vocabSize * config.hiddenDim
        let lstmParamsPerLayer = (config.hiddenDim + config.hiddenDim) * config.hiddenDim * 6
        let ffnParamsPerLayer = config.hiddenDim * config.hiddenDim * 4 * 2
        let lmHeadParams = config.hiddenDim * config.vocabSize
        
        let totalParams = embeddingParams + (config.numLayers * lstmParamsPerLayer) + 
                         (config.numLayers / 2 * ffnParamsPerLayer) + lmHeadParams
        
        print("\n📊 Model Statistics:")
        print("  - Embedding parameters: ~\(embeddingParams / 1_000_000)M")
        print("  - LSTM parameters: ~\(config.numLayers * lstmParamsPerLayer / 1_000_000)M")
        print("  - FFN parameters: ~\(config.numLayers / 2 * ffnParamsPerLayer / 1_000_000)M")
        print("  - LM Head parameters: ~\(lmHeadParams / 1_000_000)M")
        print("  - Total parameters: ~\(totalParams / 1_000_000)M")
    }
    
    /// Displays final model information
    private func displayFinalModelInfo() {
        print("\n📈 Final Training Results:")
        print("  - Total training steps: \(trainer?.step ?? 0)")
        print("  - Final loss: \(String(format: "%.4f", trainer?.losses.last ?? 0.0))")
        print("  - Model saved to: \(config.outputPath)")
        print("  - Training stats saved to: \(config.statsPath)")
        
        if let losses = trainer?.losses, losses.count > 10 {
            let improvement = losses.first! - losses.last!
            print("  - Loss improvement: \(String(format: "%.4f", improvement))")
        }
    }
}

// MARK: - Simple Tokenizer Implementation

/// Simple tokenizer implementation for demonstration
/// In practice, you would use the tokenizer from the teacher model
public class SimpleTokenizer: Tokenizer {
    private let vocabSize: Int
    
    public init(vocabSize: Int) {
        self.vocabSize = vocabSize
    }
    
    // MARK: - Required Tokenizer Protocol Methods
    
    public func tokenize(text: String) -> [String] {
        // Simple character-level tokenization
        return text.map { String($0) }
    }
    
    public func encode(text: String) -> [Int] {
        // Simple character-level tokenization for demonstration
        let chars = Array(text.utf8)
        return chars.map { Int($0) % vocabSize }
    }
    
    public func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = encode(text: text)
        
        if addSpecialTokens {
            // Add BOS token at the beginning if available
            if let bosId = bosTokenId {
                tokens.insert(bosId, at: 0)
            }
            // Add EOS token at the end if available
            if let eosId = eosTokenId {
                tokens.append(eosId)
            }
        }
        
        return tokens
    }
    
    public func decode(tokens: [Int]) throws -> String {
        // Simple decoding - convert back to characters
        let chars = tokens.map { UInt8($0 % 256) }
        return String(bytes: chars, encoding: .utf8) ?? ""
    }
    
    public func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        // Simple decoding with special token handling
        var filteredTokens = tokens
        
        if skipSpecialTokens {
            // Filter out special tokens
            let specialTokenIds = [bosTokenId, eosTokenId, unknownTokenId].compactMap { $0 }
            filteredTokens = tokens.filter { !specialTokenIds.contains($0) }
        }
        
        let chars = filteredTokens.map { UInt8($0 % 256) }
        return String(bytes: chars, encoding: .utf8) ?? ""
    }
    
    public func convertTokenToId(_ token: String) -> Int? {
        // Simple conversion for single characters
        guard token.count == 1, let char = token.first else { return nil }
        return Int(char.asciiValue ?? 0) % vocabSize
    }
    
    public func convertIdToToken(_ id: Int) -> String? {
        // Convert ID back to character
        let charValue = UInt8(id % 256)
        return String(Character(UnicodeScalar(charValue) ?? UnicodeScalar(32)!))
    }
    
    // MARK: - Required Properties
    
    public var bosToken: String? { return "<bos>" }
    public var bosTokenId: Int? { return 1 }
    public var eosToken: String? { return "<eos>" }
    public var eosTokenId: Int? { return 2 }
    public var unknownToken: String? { return "<unk>" }
    public var unknownTokenId: Int? { return 0 }
    
    // MARK: - Chat Template Methods (Simplified Implementation)
    
    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        // Simple implementation - just concatenate message content
        let combinedText = messages.compactMap { $0["content"] as? String }.joined(separator: " ")
        return encode(text: combinedText, addSpecialTokens: true)
    }
    
    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
        return try applyChatTemplate(messages: messages)
    }
    
    public func applyChatTemplate(messages: [Message], tools: [ToolSpec]?, additionalContext: [String : Any]?) throws -> [Int] {
        return try applyChatTemplate(messages: messages)
    }
    
    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
        return try applyChatTemplate(messages: messages)
    }
    
    public func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
        return try applyChatTemplate(messages: messages)
    }
    
    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?) throws -> [Int] {
        var tokens = try applyChatTemplate(messages: messages)
        
        if let maxLen = maxLength, tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }
        
        return tokens
    }
    
    public func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument?, addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [ToolSpec]?, additionalContext: [String : Any]?) throws -> [Int] {
        return try applyChatTemplate(messages: messages, chatTemplate: chatTemplate, addGenerationPrompt: addGenerationPrompt, truncation: truncation, maxLength: maxLength, tools: tools)
    }
}

// MARK: - Training Errors

public enum TrainingError: Error, LocalizedError {
    case initializationFailed(String)
    case saveFailed(String)
    case dataLoadFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .initializationFailed(let message):
            return "Initialization failed: \(message)"
        case .saveFailed(let message):
            return "Save failed: \(message)"
        case .dataLoadFailed(let message):
            return "Data load failed: \(message)"
        }
    }
}

// MARK: - String Extension for Repetition

extension String {
    static func * (string: String, count: Int) -> String {
        return String(repeating: string, count: count)
    }
}
