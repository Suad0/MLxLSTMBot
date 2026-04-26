//
//  xLSTMBlock.swift
//  MLXSTMBot
//
//  Created by Suad on 22.12.25.
//

import Foundation
import MLX
import MLXNN

/// Layer state type for xLSTM blocks
public enum LayerState {
    case sLSTM(h: MLXArray, c: MLXArray, n: MLXArray, m: MLXArray)
    case mLSTM(h: MLXArray, C: MLXArray, n: MLXArray, m: MLXArray)
    
    /// Extract hidden state from any layer type
    public var hiddenState: MLXArray {
        switch self {
        case .sLSTM(let h, _, _, _):
            return h
        case .mLSTM(let h, _, _, _):
            return h
        }
    }
}

/// xLSTM Block Type
public enum xLSTMBlockType {
    case sLSTM
    case mLSTM
}

/// xLSTM Block wrapper that can contain either sLSTM or mLSTM
/// 
/// This class implements the complete xLSTM block with:
/// - Pre-layer normalization using LayerNorm
/// - Residual connections around the LSTM layer
/// - Proper projection handling (up/down proj) for mLSTM
/// - Feed-forward network (for sLSTM blocks)
public class xLSTMBlock: Module {
    
    public let blockType: xLSTMBlockType
    public let hiddenDim: Int
    
    public let preNorm: LayerNorm
    
    // sLSTM specifics
    public let sLSTMLayer: sLSTM?
    public let feedForward: SwiGLUFeedForward?
    public let postNorm: LayerNorm?
    
    // mLSTM specifics
    public let mLSTMLayer: mLSTM?
    public let upProjection: Linear?
    public let downProjection: Linear?
    
    public init(blockType: xLSTMBlockType, hiddenDim: Int, inputDim: Int? = nil, includeFeedForward: Bool = true) throws {
        self.blockType = blockType
        self.hiddenDim = hiddenDim
        
        let actualInputDim = inputDim ?? hiddenDim
        try LSTMUtils.validateDimensions(inputDim: actualInputDim, hiddenDim: hiddenDim)
        
        // Bug B7 Fix: Use LayerNorm instead of RMSNorm
        self.preNorm = LayerNorm(dimensions: hiddenDim, eps: 1e-5)
        
        switch blockType {
        case .sLSTM:
            self.sLSTMLayer = try sLSTM(inputDim: actualInputDim, hiddenDim: hiddenDim)
            self.mLSTMLayer = nil
            self.upProjection = nil
            self.downProjection = nil
            
            if includeFeedForward {
                self.feedForward = SwiGLUFeedForward(inputDim: hiddenDim)
                self.postNorm = LayerNorm(dimensions: hiddenDim, eps: 1e-5)
            } else {
                self.feedForward = nil
                self.postNorm = nil
            }
            
        case .mLSTM:
            // x_up is split 50/50 into x_l and x_g, each of width hiddenDim.
            // mLSTM receives x_l which is hiddenDim wide.
            // This is explicit and matches the split on lines below.
            let mLSTMInputDim = hiddenDim  // x_l width after splitting x_up in half
            self.mLSTMLayer = try mLSTM(inputDim: mLSTMInputDim, hiddenDim: hiddenDim)
            self.sLSTMLayer = nil
            
            // Bug B5: Complete mLSTM block structure
            self.upProjection = Linear(actualInputDim, hiddenDim * 2)
            self.downProjection = Linear(hiddenDim * 2, hiddenDim)
            
            self.feedForward = nil
            self.postNorm = nil
        }
        
        super.init()
    }
    
    public func initialState(batchSize: Int) throws -> LayerState {
        try LSTMUtils.validateBatchSize(batchSize)
        
        switch blockType {
        case .sLSTM:
            guard let sLSTM = sLSTMLayer else { fatalError("sLSTM layer not initialized") }
            let (h, c, n, m) = try sLSTM.initialState(batchSize: batchSize)
            return .sLSTM(h: h, c: c, n: n, m: m)
            
        case .mLSTM:
            guard let mLSTM = mLSTMLayer else { fatalError("mLSTM layer not initialized") }
            let (h, C, n, m) = try mLSTM.initialState(batchSize: batchSize)
            return .mLSTM(h: h, C: C, n: n, m: m)
        }
    }
    
    public func callAsFunction(_ input: MLXArray, state: LayerState) throws -> (MLXArray, LayerState) {
        guard input.ndim == 2 && input.shape[1] == hiddenDim else {
            throw LSTMUtils.LSTMError.invalidTensorShape(expected: "2D tensor [batch_size, \(hiddenDim)]", actual: "\(input.ndim)D shape \(input.shape)")
        }
        
        let normalizedInput = preNorm(input)
        let finalOutput: MLXArray
        let newLSTMState: LayerState
        
        switch blockType {
        case .sLSTM:
            guard let sLSTM = sLSTMLayer else { fatalError("sLSTM layer not initialized") }
            guard case .sLSTM(let h, let c, let n, let m) = state else {
                fatalError("Invalid state type for sLSTM block")
            }
            
            let (lstmOutput, (newH, newC, newN, newM)) = try sLSTM(normalizedInput, state: (h, c, n, m))
            newLSTMState = .sLSTM(h: newH, c: newC, n: newN, m: newM)
            
            let afterLSTM = input + lstmOutput
            
            if let ffn = feedForward, let postNorm = postNorm {
                let normalizedAfterLSTM = postNorm(afterLSTM)
                let ffnOutput = ffn(normalizedAfterLSTM)
                finalOutput = afterLSTM + ffnOutput
            } else {
                finalOutput = afterLSTM
            }
            
        case .mLSTM:
            guard let mLSTM = mLSTMLayer, let upProj = upProjection, let downProj = downProjection else {
                fatalError("mLSTM layer/projections not initialized")
            }
            guard case .mLSTM(let h, let C, let n, let m) = state else {
                fatalError("Invalid state type for mLSTM block")
            }
            
            let x_up = upProj(normalizedInput)
            
            // Split into x_l and x_g
            // Shape of x_up is [batchSize, hiddenDim * 2]
            let x_l = x_up[0..., 0..<hiddenDim]
            let x_g = x_up[0..., hiddenDim...]
            
            assert(x_l.shape[1] == mLSTM.inputDim,
                   "x_l width \(x_l.shape[1]) must match mLSTM inputDim \(mLSTM.inputDim)")
            
            let (h_t, (newH, newC, newN, newM)) = try mLSTM(x_l, state: (h, C, n, m))
            newLSTMState = .mLSTM(h: newH, C: newC, n: newN, m: newM)
            
            let combined = MLX.concatenated([h_t, x_g], axis: -1)
            let y = downProj(combined)
            
            finalOutput = input + y
        }
        
        return (finalOutput, newLSTMState)
    }
    
    public func processSequence(_ sequence: MLXArray, initialState: LayerState? = nil) throws -> (MLXArray, LayerState) {
        try LSTMUtils.validateSequenceTensor(sequence)
        
        let batchSize = sequence.shape[0]
        let sequenceLength = sequence.shape[1]
        
        var currentState = try initialState ?? self.initialState(batchSize: batchSize)
        
        if sequenceLength == 0 {
            return (LSTMUtils.createTensor(shape: [batchSize, 0, hiddenDim]), currentState)
        }
        
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(sequenceLength)
        
        // As a fallback for causal convolution across correct timesteps in processSequence, 
        // we'd optimally process the whole sequence with conv then feed timestep by timestep or trace it.
        // However, the callAsFunction handles the internal slice appropriately.
        for t in 0..<sequenceLength {
            let timestepInput = sequence[0..., t, 0...]
            let (output, newState) = try callAsFunction(timestepInput, state: currentState)
            outputs.append(output)
            currentState = newState
        }
        
        let allOutputs = stacked(outputs, axis: 1)
        return (allOutputs, currentState)
    }
}
