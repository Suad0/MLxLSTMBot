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
        var teacherModelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
        var vocabSize = 128256  // Match teacher vocabulary
        var hiddenDim = 512
        var numLayers = 6
        var batchSize = 4
        var sequenceLength = 256
        var numIterations = 1000 // Bug B12 Fix
        var learningRate: Float = 1e-4
        var printEvery = 1  // Print every step for debugging
        var saveEvery = 10
        var dataPath = "training_data.json"
        var outputPath = "xLSTM_Final.safetensors"
        var statsPath = "training_stats.json"
    }
    
    // MARK: - Properties
    
    private let config = TrainingConfig()
    private var dataProvider: ChatDataProvider?
    private var trainer: DistillationTrainer?
    private var studentModel: xLSTM?
    private var teacherModel: (any LanguageModel)?
    private var tokenizer: Tokenizer?
    
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
        
        print("Initializing student xLSTM model...")
        let blockSpec: [xLSTMBlockType] = [.mLSTM, .mLSTM, .mLSTM, .mLSTM, .mLSTM, .sLSTM]
        self.studentModel = try xLSTM(
            vocabSize: config.vocabSize,
            hiddenDim: config.hiddenDim,
            blockSpec: blockSpec
        )
        
        print("✓ Student xLSTM model initialized")
        print("  - Vocabulary size: \(config.vocabSize)")
        print("  - Hidden dimension: \(config.hiddenDim)")
        print("  - Number of layers: \(config.numLayers)")
        
        // Calculate and display model parameters
        displayModelStats()

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
        self.teacherModel = await container.perform { model, _ in model }
        if self.teacherModel == nil {
             print("Warning: Could not load teacher model. Distillation might fail.")
        }
        
        // Extract teacher tokenizer
        self.tokenizer = await container.perform { _, tokenizer in tokenizer }
    }
    
    /// Sets up the data provider
    private func setupDataProvider() throws {
        print("\n📊 Setting up Data Provider...")
        
        // Create sample data if it doesn't exist
        if !FileManager.default.fileExists(atPath: config.dataPath) {
            print("Creating sample training data...")
            try ChatDataProvider.createSampleData(at: config.dataPath)
        }
        
        guard let actualTokenizer = self.tokenizer else {
            fatalError("🚨 Critical Initialization Error: Teacher Tokenizer not found! Distillation requires a valid teacher tokenizer.")
        }
        
        // Use exact teacher vocabulary size for safety if possible
        // Note: the tokenizer size might be larger than config.vocabSize, in reality we should match them
        self.dataProvider = try ChatDataProvider(
            jsonPath: config.dataPath,
            tokenizer: actualTokenizer,
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
            let loss = try trainer.trainingStep(inputs: inputs, targets: targets, padId: dataProvider.padTokenId)
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
        
        // Calculate approximate parameter count based on correct mLSTM layout
        let embeddingParams = config.vocabSize * config.hiddenDim
        
        let mLSTMParamsPerBlock = 
            (config.hiddenDim * config.hiddenDim * 2) + // upProj
            (config.hiddenDim * config.hiddenDim * 2) + // downProj
            (config.hiddenDim * config.hiddenDim * 7) + // input, forget, output, skip, key, value, query
            (config.hiddenDim * 4) + // conv1d (approx depthwise)
            (config.hiddenDim) // groupNorm
            
        let sLSTMParamsPerBlock = 
            (config.hiddenDim * config.hiddenDim * 4) + // input, forget, cell, output
            (config.hiddenDim * config.hiddenDim * 4 * 2) // FFN
            
        let lmHeadParams = config.hiddenDim * config.vocabSize
        
        // 5 mLSTM + 1 sLSTM layout for 6 layers
        let num_mLSTM = 5
        let num_sLSTM = 1
        
        let totalParams = embeddingParams + 
                          (num_mLSTM * mLSTMParamsPerBlock) +
                          (num_sLSTM * sLSTMParamsPerBlock) + 
                          lmHeadParams
        
        print("\n📊 Model Statistics:")
        print("  - Embedding parameters: ~\(embeddingParams / 1_000_000)M")
        print("  - mLSTM block parameters: ~\(num_mLSTM * mLSTMParamsPerBlock / 1_000_000)M")
        print("  - sLSTM block parameters: ~\(num_sLSTM * sLSTMParamsPerBlock / 1_000_000)M")
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

private extension String {
    static func * (string: String, count: Int) -> String {
        return String(repeating: string, count: count)
    }
}
