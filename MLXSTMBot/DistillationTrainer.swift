//
//  DistillationTrainer.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers
import MLXLMCommon
import MLXLLM

/// Knowledge Distillation Trainer for xLSTM
/// 
/// This class implements knowledge distillation from a teacher LLM to a student xLSTM model.
/// It combines cross-entropy loss (student vs ground truth) with KL divergence loss (student vs teacher).
public class DistillationTrainer {
    
    // MARK: - Properties
    
    /// Student xLSTM model to be trained
    private let studentModel: xLSTM
    
    /// Teacher LLM model (frozen, eval mode) - optional for student-only training
    private let teacherModel: (any LanguageModel)?
    
    /// AdamW optimizer for the student model
    private let optimizer: AdamW
    
    /// Temperature for KL divergence loss
    private let temperature: Float
    
    /// Weight for the distillation loss (KL divergence)
    private let distillationWeight: Float
    
    /// Weight for the ground truth loss (cross-entropy)
    private let groundTruthWeight: Float
    
    /// Current training step
    private var currentStep: Int = 0
    
    /// Loss history for monitoring
    private var lossHistory: [Float] = []
    
    // MARK: - Initialization
    
    /// Initializes the distillation trainer
    /// - Parameters:
    ///   - studentModel: The xLSTM model to train
    ///   - teacherModel: The teacher LLM model (optional, nil for student-only training)
    ///   - learningRate: Learning rate for AdamW optimizer (default: 1e-4)
    ///   - temperature: Temperature for KL divergence (default: 2.0)
    ///   - distillationWeight: Weight for distillation loss (default: 0.7)
    ///   - groundTruthWeight: Weight for ground truth loss (default: 0.3)
    public init(
        studentModel: xLSTM,
        teacherModel: (any LanguageModel)?,
        learningRate: Float = 1e-4,
        temperature: Float = 2.0,
        distillationWeight: Float = 0.7,
        groundTruthWeight: Float = 0.3
    ) {
        self.studentModel = studentModel
        self.teacherModel = teacherModel
        self.temperature = temperature
        self.distillationWeight = distillationWeight
        self.groundTruthWeight = groundTruthWeight
        
        // Initialize AdamW optimizer
        self.optimizer = AdamW(learningRate: learningRate)
        
        // Set teacher model to eval mode (no gradients)
        setTeacherEvalMode()
        
        print("DistillationTrainer initialized:")
        print("  - Learning rate: \(learningRate)")
        print("  - Temperature: \(temperature)")
        print("  - Distillation weight: \(distillationWeight)")
        print("  - Ground truth weight: \(groundTruthWeight)")
    }
    
    /// Sets the teacher model to evaluation mode
    private func setTeacherEvalMode() {
        // The teacher model should be in eval mode to prevent gradient computation
        // This is handled by not including teacher parameters in the gradient computation
        print("Teacher model set to evaluation mode")
    }
    
    // MARK: - Training Step
    
    /// Performs a single training step with knowledge distillation
    /// - Parameters:
    ///   - inputs: Input token sequences [batch_size, seq_len]
    ///   - targets: Target token sequences [batch_size, seq_len]
    /// - Returns: Combined loss value
    /// - Throws: Error if training step fails
    public func trainingStep(inputs: MLXArray, targets: MLXArray) throws -> Float {
        currentStep += 1
        
        // Validate input shapes
        guard inputs.shape.count == 2 && targets.shape.count == 2 else {
            throw DistillationError.invalidInputShape("Expected 2D tensors for inputs and targets")
        }
        
        guard inputs.shape[0] == targets.shape[0] && inputs.shape[1] == targets.shape[1] else {
            throw DistillationError.shapeMismatch("Input and target shapes must match")
        }
        
        let batchSize = inputs.shape[0]
        let seqLen = inputs.shape[1]
        
        // Get teacher logits (no gradients) - skip if no teacher model
        let teacherLogits = if let teacher = teacherModel {
            try getTeacherLogits(inputs: inputs)
        } else {
            // Create dummy teacher logits with same shape as student output
            MLX.zeros([batchSize, seqLen, studentModel.vocabSize])
        }
        
        // Define the loss function and gradient computation
        let lossAndGrad = MLXNN.valueAndGrad(model: studentModel) { [self] model, inputs, targets in
            // Initialize student states for this batch
            let initialStates = try! model.initialStates(batchSize: batchSize)
            
            // Forward pass through student model
            let (studentLogits, _) = try! model.processSequence(inputs, states: initialStates)
            
            // Reshape logits for loss computation
            // studentLogits: [batch_size, seq_len, vocab_size]
            // targets: [batch_size, seq_len]
            
            let vocabSize = studentLogits.shape[2]
            
             // Debug: Check for NaNs
            if LSTMUtils.hasNaNOrInf(studentLogits) {
                 print("  DEBUG: Student Logits contain NaN/Inf")
            }
            if LSTMUtils.hasNaNOrInf(teacherLogits) {
                 print("  DEBUG: Teacher Logits contain NaN/Inf")
            }
            
            let flatStudentLogits = studentLogits.reshaped([-1, vocabSize]) // [batch_size * seq_len, vocab_size]
            let flatTargets = targets.reshaped([-1]) // [batch_size * seq_len]
            let flatTeacherLogits = teacherLogits.reshaped([-1, vocabSize]) // [batch_size * seq_len, vocab_size]
            
            // Compute ground truth loss (cross-entropy)
            let groundTruthLoss = MLXNN.crossEntropy(
                logits: flatStudentLogits,
                targets: flatTargets
            ).mean()
            
            // Compute distillation loss (KL divergence)
            let distillationLoss = computeKLDivergence(
                studentLogits: flatStudentLogits,
                teacherLogits: flatTeacherLogits,
                temperature: self.temperature
            )
            
            print("  DEBUG: GT Loss: \(groundTruthLoss.item(Float.self)), Distill Loss: \(distillationLoss.item(Float.self))")
            
            // Combine losses
            let distillWeight = (self.teacherModel != nil) ? self.distillationWeight : 0.0
            let totalLoss = self.groundTruthWeight * groundTruthLoss +
                           distillWeight * distillationLoss
            
            return totalLoss
        }
        
        // Compute loss and gradients
        let (loss, gradients) = lossAndGrad(studentModel, inputs, targets)
        
        // Apply gradient clipping to prevent exploding gradients
        let clippedGradients = clipGradients(gradients, maxNorm: 1.0)
        
        // Update model parameters
        optimizer.update(model: studentModel, gradients: clippedGradients)
        // Record loss
        let lossValue = loss.item(Float.self)
        lossHistory.append(lossValue)
        
        // Force evaluation of the graph once per step, rather than sequentially per-timestep
        MLX.eval(studentModel, optimizer)
        
        return lossValue
    }
    
    /// Gets teacher model logits without computing gradients
    /// - Parameter inputs: Input token sequences
    /// - Returns: Teacher logits
    /// - Throws: Error if teacher inference fails
    private func getTeacherLogits(inputs: MLXArray) throws -> MLXArray {
        guard let teacher = teacherModel else {
            throw DistillationError.teacherInferenceFailed("No teacher model available")
        }
        
        // Wrap inputs in LMInput.Text for the LanguageModel protocol
        let textInput = LMInput.Text(tokens: inputs)
        
        // Perform inference using the teacher model
        // Passing cache: nil ensures the model processes the entire sequence in parallel
        // output.logits shape: [batch_size, seq_len, vocab_size]
        let output = teacher(textInput, cache: nil, state: nil)
        
        return output.logits
    }
    
    /// Computes KL divergence loss between student and teacher logits
    /// - Parameters:
    ///   - studentLogits: Student model logits
    ///   - teacherLogits: Teacher model logits
    ///   - temperature: Temperature for softmax
    /// - Returns: KL divergence loss
    private func computeKLDivergence(
        studentLogits: MLXArray,
        teacherLogits: MLXArray,
        temperature: Float
    ) -> MLXArray {
        let scaledStudent = studentLogits / temperature
        let scaledTeacher = teacherLogits / temperature

        // Use logSoftmax on both sides — numerically stable via log-sum-exp trick
        // This avoids log(~0) = -inf that occurs when computing log(softmax(...)) explicitly
        let logStudentProbs = MLXNN.logSoftmax(scaledStudent, axis: -1)
        let logTeacherProbs = MLXNN.logSoftmax(scaledTeacher, axis: -1)

        // exp(logSoftmax) is bounded [0,1] — never explodes
        let teacherProbs = MLX.exp(logTeacherProbs)

        // KL(teacher || student) = sum(teacher * (log_teacher - log_student))
        let klDiv = teacherProbs * (logTeacherProbs - logStudentProbs)

        // Mean over batch, sum over vocabulary dimension
        return MLX.mean(MLX.sum(klDiv, axis: -1)) * (temperature * temperature)
    }
    
    /// Clips gradients to prevent exploding gradients
    /// - Parameters:
    ///   - gradients: ModuleParameters (NestedDictionary) of gradients
    ///   - maxNorm: Maximum gradient norm
    /// - Returns: Clipped gradients
    private func clipGradients(_ gradients: ModuleParameters, maxNorm: Float) -> ModuleParameters {
        // Calculate total gradient norm
        var totalNorm: Float = 0.0
        for (_, grad) in gradients.flattened() {
            let gradNorm = MLX.sum(grad * grad).item(Float.self)
            totalNorm += gradNorm
        }
        totalNorm = sqrt(totalNorm)
        
        // If total norm exceeds maxNorm, scale down all gradients
        if totalNorm > maxNorm {
            let scale = maxNorm / totalNorm
            
            // Apply scaling while preserving the nested structure
            let clippedGradients = gradients.mapValues { $0 * scale }
            
            print("Gradient clipping applied: norm=\(totalNorm), scale=\(scale)")
            return clippedGradients
        }
        
        return gradients
    }
    
    // MARK: - Training Utilities
    
    /// Returns the current training step
    public var step: Int {
        return currentStep
    }
    
    /// Returns the loss history
    public var losses: [Float] {
        return lossHistory
    }
    
    /// Returns the average loss over the last N steps
    /// - Parameter steps: Number of recent steps to average (default: 10)
    /// - Returns: Average loss
    public func averageLoss(overLast steps: Int = 10) -> Float {
        guard !lossHistory.isEmpty else { return 0.0 }
        
        let recentLosses = Array(lossHistory.suffix(steps))
        return recentLosses.reduce(0, +) / Float(recentLosses.count)
    }
    
    /// Resets the training state
    public func reset() {
        currentStep = 0
        lossHistory.removeAll()
        print("Training state reset")
    }
    
    /// Saves training statistics to a file
    /// - Parameter path: Path to save the statistics
    /// - Throws: Error if file cannot be written
    public func saveTrainingStats(to path: String) throws {
        let stats = [
            "total_steps": currentStep,
            "final_loss": Double(lossHistory.last ?? 0.0),
            "average_loss_last_10": Double(averageLoss(overLast: 10)),
            "loss_history": lossHistory.map { Double($0) }
        ] as [String : Any]
        
        // Ensure directory exists
        let directoryURL = URL(fileURLWithPath: path).deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: directoryURL.path) {
            try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true)
        }
        
        let jsonData = try JSONSerialization.data(withJSONObject: stats, options: .prettyPrinted)
        let url = URL(fileURLWithPath: path)
        try jsonData.write(to: url)
        
        print("Training statistics saved to: \(path)")
    }
}

// MARK: - Error Types

public enum DistillationError: Error, LocalizedError {
    case invalidInputShape(String)
    case shapeMismatch(String)
    case teacherInferenceFailed(String)
    case optimizationFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .invalidInputShape(let message):
            return "Invalid input shape: \(message)"
        case .shapeMismatch(let message):
            return "Shape mismatch: \(message)"
        case .teacherInferenceFailed(let message):
            return "Teacher inference failed: \(message)"
        case .optimizationFailed(let message):
            return "Optimization failed: \(message)"
        }
    }
}

// MARK: - Extensions

extension DistillationTrainer {
    
    /// Performs a validation step (no parameter updates)
    /// - Parameters:
    ///   - inputs: Input token sequences
    ///   - targets: Target token sequences
    /// - Returns: Validation loss
    /// - Throws: Error if validation fails
    public func validationStep(inputs: MLXArray, targets: MLXArray) throws -> Float {
        let batchSize = inputs.shape[0]
        
        // Get teacher logits (skip if no teacher)
        let teacherLogits = if teacherModel != nil {
            try getTeacherLogits(inputs: inputs)
        } else {
            MLX.zeros([batchSize, inputs.shape[1], studentModel.vocabSize])
        }
        
        // Forward pass through student (no gradients)
        let initialStates = try studentModel.initialStates(batchSize: batchSize)
        let (studentLogits, _) = try studentModel.processSequence(inputs, states: initialStates)
        
        // Compute losses
        let vocabSize = studentLogits.shape[2]
        let flatStudentLogits = studentLogits.reshaped([-1, vocabSize])
        let flatTargets = targets.reshaped([-1])
        let flatTeacherLogits = teacherLogits.reshaped([-1, vocabSize])
        
        let groundTruthLoss = MLXNN.crossEntropy(
            logits: flatStudentLogits,
            targets: flatTargets
        ).mean()
        
        let distillationLoss = computeKLDivergence(
            studentLogits: flatStudentLogits,
            teacherLogits: flatTeacherLogits,
            temperature: temperature
        )
        
        let distillWeight = (teacherModel != nil) ? distillationWeight : 0.0
        let totalLoss = groundTruthWeight * groundTruthLoss + distillWeight * distillationLoss
        
        return totalLoss.item(Float.self)
    }
}
