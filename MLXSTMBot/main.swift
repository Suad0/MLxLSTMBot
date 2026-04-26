//
//  main.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN
import MLXLMCommon
import MLXLLM
import AppKit

// MARK: - Main Application Entry Point

print("🚀 MLXSTMBot - xLSTM Knowledge Distillation Training")
print(String(repeating: "=", count: 60))

// Check command line arguments for mode selection
let arguments = CommandLine.arguments

if arguments.count > 1 && arguments[1] == "train" {
    // Training Mode
    print("🎯 Starting Training Mode...")
    
    let trainingMain = TrainingMain()
    
    // Run training in async context
    let _ = Task {
        await trainingMain.runTraining()
        exit(0)
    }
    
    // Keep the program alive
    RunLoop.main.run()
    
} else if arguments.count > 1 && arguments[1] == "test" {
    // Test Mode - Run the original architecture tests
    print("🧪 Starting Test Mode...")
    runArchitectureTests()
    
} else {
    // Interactive Mode - Load teacher model and demonstrate inference
    print("💬 Starting Interactive Mode...")
    runInteractiveMode()
}

// MARK: - Interactive Mode (Original Llama Test)

func runInteractiveMode() {
    let modelId = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    let modelFactory = LLMModelFactory.shared
    let configuration = ModelConfiguration(id: modelId)

    print("Loading model: \(modelId)...")

    let _ = Task {
        do {
            let model = try await modelFactory.loadContainer(configuration: configuration) { progress in
                let percent = String(format: "%.1f", progress.fractionCompleted * 100)
                print("\rDownload progress: \(percent)%", terminator: "")
                fflush(stdout)
            }
            print("\nModel loaded successfully!")

            try await model.perform { context in
                let prompt = "Tell me a joke about robots."
                print("\nUser: \(prompt)")
                print("Assistant: ", terminator: "")
                
                let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
                let params = GenerateParameters(temperature: 0.7)
                let tokenStream = try generate(input: input, parameters: params, context: context)
                
                for try await token in tokenStream {
                    if let text = token.chunk {
                        print(text, terminator: "")
                        fflush(stdout)
                    }
                }
                print("\n\n--- Interactive Test Complete ---")
            }
        } catch {
            print("\nError: \(error.localizedDescription)")
        }
        
        exit(0)
    }

    RunLoop.main.run()
}

// MARK: - Architecture Test Mode

func runArchitectureTests() {
    do {
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
        
        let (sLSTMBlockOutput, _) = try sLSTMBlock(blockInput, state: sLSTMBlockState)
        let (mLSTMBlockOutput, _) = try mLSTMBlock(blockInput, state: mLSTMBlockState)
        
        print("✓ Block processing successful")
        print("  - sLSTM block output shape: \(sLSTMBlockOutput.shape)")
        print("  - mLSTM block output shape: \(mLSTMBlockOutput.shape)")
        
        print("\n3. Testing Full xLSTM Architecture...")
        
        // Create full xLSTM model
        let vocabSize = 1000
        let blockSpec: [xLSTMBlockType] = [.mLSTM, .sLSTM, .mLSTM, .sLSTM]
        
        let xlstm = try xLSTM(
            vocabSize: vocabSize,
            hiddenDim: hiddenDim,
            blockSpec: blockSpec
        )
        
        print("✓ xLSTM model created")
        print("  - Vocabulary size: \(vocabSize)")
        print("  - Hidden dimension: \(hiddenDim)")
        print("  - Number of layers: \(blockSpec.count)")
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
        
        print("\n✅ Architecture tests completed successfully!")
        
    } catch {
        print("❌ Error occurred: \(error)")
        exit(1)
    }
}

// MARK: - Usage Information

func printUsage() {
    print("""
    
    📖 Usage:
    
    ./MLXSTMBot                 - Interactive mode (Llama inference demo)
    ./MLXSTMBot train          - Training mode (Knowledge distillation)
    ./MLXSTMBot test           - Test mode (Architecture validation)
    
    🎯 Training Mode:
    - Loads Llama-3.2-1B-Instruct-4bit as teacher
    - Trains xLSTM student model via knowledge distillation
    - Saves final model to xLSTM_Final.safetensors
    
    🧪 Test Mode:
    - Validates xLSTM architecture components
    - Tests individual LSTM layers and blocks
    - Verifies end-to-end functionality
    
    💬 Interactive Mode:
    - Demonstrates teacher model inference
    - Shows text generation capabilities
    - Useful for testing model loading
    
    """)
}
