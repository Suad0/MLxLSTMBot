# GEMINI.md — MLXSTMBot xLSTM Project

## Role

You are a senior Swift/MLX engineer working on **MLXSTMBot**, an xLSTM language model trained via knowledge distillation from a Llama teacher. Your job is to fix a specific set of reviewed bugs without changing the overall architecture unless a fix explicitly requires it.

---

## Project overview

| File | Purpose |
|---|---|
| `main.swift` | Entry point: selects interactive / test / train mode |
| `TrainingMain.swift` | Orchestrates the full distillation pipeline |
| `DistillationTrainer.swift` | AdamW training loop, KL + CE loss, gradient clipping |
| `xLSTM.swift` | Top-level model: embedding → blocks → LayerNorm → LM head |
| `xLSTMBlock.swift` | Block wrapper: preNorm, residual, dispatches to sLSTM or mLSTM |
| `sLSTM.swift` | Scalar LSTM with exponential gating and normalizer state |
| `mLSTM.swift` | Matrix LSTM with outer-product memory and causal conv |
| `FeedForward.swift` | GatedFeedForward (SwiGLU) and MLPFeedForward |
| `LSTMUtils.swift` | Shared validation helpers and numerical stability utilities |
| `ChatDataProvider.swift` | JSON loader, tokenizer wrapper, batch generator |

**Frameworks in use:** MLX, MLXNN, MLXOptimizers, MLXRandom, MLXLMCommon, MLXLLM, Tokenizers (all via Swift Package Manager).

---

## Build and verification workflow

> **Always use these scripts — never invoke `swift build` or `swift run` directly.**

### After implementing any fix, verify with:

```bash
# 1. Build the project
./build.sh

# 2. If the build succeeds, run a short training smoke-test
./train.sh
```

Both scripts must exit with code 0 before you mark a fix complete. If either fails, diagnose the error output, fix it, and re-run both scripts.

### Expected `build.sh` content (create if missing):

```bash
#!/bin/bash
set -e
echo "=== Building MLXSTMBot ==="
swift build -c release 2>&1
echo "=== Build succeeded ==="
```

### Expected `train.sh` content (create if missing):

```bash
#!/bin/bash
set -e
echo "=== Running training smoke-test ==="
swift run -c release MLXSTMBot train 2>&1 | head -60
echo "=== Smoke-test complete ==="
```

---

## Ordered fix list

Work through these fixes **in order**. Each fix is self-contained. Run `./build.sh` after every fix and `./train.sh` after fix 8 (the last one) to confirm the full pipeline.

---

### Fix 1 — `sLSTM`: normalizer state initialised to wrong value

**File:** `sLSTM.swift`  
**Severity:** 🔴 Critical — causes division by ~0 on the very first training step.

**Problem:** `n_t` is initialised to `0.0`. At step 0 the denominator becomes `max(0.0, 1e-8) = 1e-8`, so `h_t = o_t * (c_t / 1e-8)`, amplifying activations by 100 million.

**Fix:** Change the initial value to `1.0` in `initialState(batchSize:)`:

```swift
// sLSTM.swift — func initialState(batchSize: Int)
// BEFORE:
let n_t = LSTMUtils.createTensor(shape: stateShape, value: 0.0)
// AFTER:
let n_t = LSTMUtils.createTensor(shape: stateShape, value: 1.0)
```

**Verify:** `./build.sh` exits 0.

---

### Fix 2 — `mLSTM`: causal convolution leaks future context

**File:** `mLSTM.swift`  
**Severity:** 🔴 Critical — the model is not causal during training; it sees future tokens.

**Problem:** `Conv1d(padding: 3)` pads *both* sides with 3 zeros (total 6), making future tokens visible to past positions. A causal convolution of kernel size 4 must pad `kernelSize - 1 = 3` on the **left only**.

MLX's `Conv1d` does not support asymmetric padding natively. The fix is to pad manually and set `padding: 0`.

**Fix — step 1:** Change the Conv1d declaration in `init`:

```swift
// mLSTM.swift — init
// BEFORE:
self.conv1dCausal = Conv1d(inputChannels: inputDim, outputChannels: inputDim,
                            kernelSize: 4, stride: 1, padding: 3, groups: inputDim)
// AFTER:
self.conv1dCausal = Conv1d(inputChannels: inputDim, outputChannels: inputDim,
                            kernelSize: 4, stride: 1, padding: 0, groups: inputDim)
```

**Fix — step 2:** Replace the single-timestep conv call in `callAsFunction`:

```swift
// mLSTM.swift — callAsFunction, replace the conv block
// BEFORE:
let x_expanded = input.reshaped([batchSize, 1, inputDim])
var convOut = conv1dCausal(x_expanded)
convOut = convOut[0..., 0..<1, 0...]
let x_conv = convOut.squeezed(axis: 1)

// AFTER:
// For single-step autoregressive inference, left-pad with kernelSize-1 zeros
let leftPad = MLX.zeros([batchSize, 3, inputDim])
let x_padded = MLX.concatenated([leftPad, input.reshaped([batchSize, 1, inputDim])], axis: 1)
let convOut = conv1dCausal(x_padded) // shape: [batchSize, 1, inputDim]
let x_conv = convOut.squeezed(axis: 1)
```

**Fix — step 3:** In `processSequence` (the extension), replace the per-timestep loop's conv handling by processing the whole sequence at once before the loop:

```swift
// mLSTM.swift — processSequence extension
// Add this block BEFORE the timestep loop:
// Left-pad the full sequence for causal conv, then apply once
let seqLeftPad = MLX.zeros([batchSize, 3, inputDim])
let seqPadded  = MLX.concatenated([seqLeftPad, sequence], axis: 1)
let convSequence = conv1dCausal(seqPadded) // [batchSize, sequenceLength, inputDim]

// Then inside the loop, pass the pre-computed conv slice instead of recomputing:
// (you will need to thread convSequence into callAsFunction or compute x_conv directly here)
```

> **Note:** The cleanest architecture change here is to add an optional `precomputedConv: MLXArray?` parameter to `callAsFunction` so `processSequence` can pass the pre-computed conv output in. This avoids running the depthwise conv `sequenceLength` times.

**Verify:** `./build.sh` exits 0.

---

### Fix 3 — `DistillationTrainer`: `try!` inside gradient closure crashes instead of throwing

**File:** `DistillationTrainer.swift`  
**Severity:** 🔴 Critical — any shape mismatch or invalid state silently kills the process with no error message.

**Problem:** Lines 127–130 use `try!` inside the `valueAndGrad` closure. The gradient tracer cannot propagate Swift errors; a forced try that triggers kills the entire run.

**Fix:** Move state initialisation outside the closure and propagate errors through the enclosing `throws` function:

```swift
// DistillationTrainer.swift — trainingStep(inputs:targets:padId:)
// BEFORE (inside the closure):
let lossAndGrad = MLXNN.valueAndGrad(model: studentModel) { [self] model, inputs, targets in
    let initialStates = try! model.initialStates(batchSize: batchSize)
    let (studentLogits, _) = try! model.processSequence(inputs, states: initialStates)
    // ...
}

// AFTER:
// Outside and before the closure definition:
let initialStates = try studentModel.initialStates(batchSize: batchSize)

let lossAndGrad = MLXNN.valueAndGrad(model: studentModel) { [self] model, inputs, targets in
    // processSequence is still fallible inside the closure; wrap it:
    guard let (studentLogits, _) = try? model.processSequence(inputs, states: initialStates) else {
        // Return a large sentinel loss so training can continue; log the failure
        print("Warning: processSequence failed in gradient closure at step \(self.currentStep)")
        return MLXArray(Float.greatestFiniteMagnitude)
    }
    // ... rest of loss computation unchanged
}
```

**Verify:** `./build.sh` exits 0.

---

### Fix 4 — `DistillationTrainer`: `MLX.eval` and `.item()` ordering is wrong

**File:** `DistillationTrainer.swift`  
**Severity:** 🔴 Critical — the lazy graph is flushed in the wrong order; the optimizer update and loss readout race.

**Problem:** `.item(Float.self)` on line 184 implicitly triggers a graph flush *before* `MLX.eval(studentModel, optimizer)` on line 188, meaning parameter weights may be read before the update is fully materialized.

**Fix:** Reorder so the model and optimizer are evaluated first, then read the scalar:

```swift
// DistillationTrainer.swift — end of trainingStep, replace lines 179–190
let clippedGradients = clipGradients(gradients, maxNorm: 1.0)
optimizer.update(model: studentModel, gradients: clippedGradients)

// Flush the full computation graph (params + optimizer state) before reading any scalar
MLX.eval(studentModel, optimizer)

// Now it is safe to read the loss scalar
let lossValue = loss.item(Float.self)
lossHistory.append(lossValue)

return lossValue
```

**Verify:** `./build.sh` exits 0.

---

### Fix 5 — `DistillationTrainer`: teacher model API call uses wrong signature

**File:** `DistillationTrainer.swift`  
**Severity:** 🔴 Critical — will not compile or calls the wrong overload at runtime.

**Problem:** `teacher(textInput, cache: nil, state: nil)` — the `LanguageModel` protocol in `MLXLLM` does not declare a `state:` parameter. This is a dangling argument that either fails to compile or silently dispatches to an unintended overload.

**Fix:**

```swift
// DistillationTrainer.swift — getTeacherLogits
// BEFORE:
let output = teacher(textInput, cache: nil, state: nil)

// AFTER:
let output = teacher(textInput, cache: nil)
```

**Verify:** `./build.sh` exits 0.

---

### Fix 6 — `DistillationTrainer`: teacher model is never actually frozen

**File:** `DistillationTrainer.swift`  
**Severity:** 🟠 High — teacher parameters are included in gradient computation, wasting memory and corrupting distillation.

**Problem:** `setTeacherEvalMode()` only prints a message. Teacher parameters need to be excluded from the gradient graph.

**Fix:** Replace the no-op with an actual freeze call. Since `teacherModel` is typed as `any LanguageModel` (a protocol), you need to conditionally cast to `Module`:

```swift
// DistillationTrainer.swift — setTeacherEvalMode()
private func setTeacherEvalMode() {
    if let teacherModule = teacherModel as? (any Module) {
        teacherModule.freeze(recursive: true)
        print("Teacher model frozen — parameters excluded from gradient graph")
    } else {
        print("Warning: could not freeze teacher model — ensure it is not trainable")
    }
}
```

> If `LanguageModel` does not conform to `Module` directly in your version of MLXLLM, check the concrete type returned by `LLMModelFactory` and cast to that instead (e.g. `LlamaModel`).

**Verify:** `./build.sh` exits 0.

---

### Fix 7 — `DistillationTrainer`: KL divergence T² scaling dominates loss weighting

**File:** `DistillationTrainer.swift`  
**Severity:** 🟠 High — at temperature 2.0, distillation loss is automatically 4× larger in raw magnitude before `distillationWeight` is applied, making the weight parameters meaningless.

**Problem:** The `temperature * temperature` multiplier in `computeKLDivergence` is the correct Hinton correction for gradient scale, but it means the returned loss is on a completely different scale than the CE loss. The weight parameters `distillationWeight` and `groundTruthWeight` cannot compensate for this without being tuned to the temperature, which breaks whenever temperature changes.

**Fix:** Remove the T² multiplier from `computeKLDivergence` and instead document that the caller is responsible for scale balancing. Add a normalisation comment:

```swift
// DistillationTrainer.swift — computeKLDivergence
// BEFORE:
return MLX.mean(MLX.sum(klDiv, axis: -1)) * (temperature * temperature)

// AFTER:
// Return raw KL without T² correction; caller applies distillationWeight.
// If you want the Hinton gradient-scale correction, multiply the *gradient*
// by T², not the loss value itself. Multiplying the loss shifts the absolute
// magnitude and breaks the distillationWeight / groundTruthWeight ratio.
return MLX.mean(MLX.sum(klDiv, axis: -1))
```

**Verify:** `./build.sh` exits 0.

---

### Fix 8 — `xLSTMBlock`: mLSTM receives wrong input dimension

**File:** `xLSTMBlock.swift`  
**Severity:** 🟠 High — the mLSTM processes only half the expanded representation, losing the benefit of the up-projection.

**Problem:** `upProjection` expands to `hiddenDim * 2`, then `x_l = x_up[0..., 0..<hiddenDim]` is only `hiddenDim` wide. But `mLSTMLayer` was created with `inputDim: hiddenDim`, which happens to match — but only because both sides are truncated. Per the xLSTM paper, the mLSTM should process the full expanded `x_l`. The fix is to make `mLSTM`'s `inputDim` equal the split width explicitly, making the intent clear:

```swift
// xLSTMBlock.swift — init, case .mLSTM:
// BEFORE:
self.mLSTMLayer = try mLSTM(inputDim: hiddenDim, hiddenDim: hiddenDim)
self.upProjection = Linear(actualInputDim, hiddenDim * 2)
self.downProjection = Linear(hiddenDim * 2, hiddenDim)

// AFTER:
// x_up is split 50/50 into x_l and x_g, each of width hiddenDim.
// mLSTM receives x_l which is hiddenDim wide.
// This is explicit and matches the split on lines below.
let mLSTMInputDim = hiddenDim  // x_l width after splitting x_up in half
self.mLSTMLayer    = try mLSTM(inputDim: mLSTMInputDim, hiddenDim: hiddenDim)
self.upProjection  = Linear(actualInputDim, hiddenDim * 2)
self.downProjection = Linear(hiddenDim * 2, hiddenDim)
```

Then add an assertion in `callAsFunction` to make the contract explicit:

```swift
// xLSTMBlock.swift — callAsFunction, case .mLSTM, after the split:
let x_l = x_up[0..., 0..<hiddenDim]
let x_g = x_up[0..., hiddenDim...]
assert(x_l.shape[1] == mLSTMLayer!.inputDim,
       "x_l width \(x_l.shape[1]) must match mLSTM inputDim \(mLSTMLayer!.inputDim)")
```

**Verify:** Run both scripts:

```bash
./build.sh
./train.sh
```

Both must exit 0 and `train.sh` must print loss values for at least 5 steps without NaN.

---

## Secondary improvements (implement after all 8 fixes pass)

These are not blocking for a working training run but should be addressed before any real training:

### `FeedForward.swift` — rename or fix gate activation

The class is named `GatedFeedForward` and the doc comment says `gate = sigmoid(...)`, but the code uses GELU (making it SwiGLU). Pick one:

- **Option A (preferred):** Rename the class to `SwiGLUFeedForward` and update its doc comment.  
- **Option B:** Change `MLXNN.gelu(gateProjection(input))` to `MLX.sigmoid(gateProjection(input))` to match the name and docs.

### `LSTMUtils.swift` — fix `createIdentityMatrix` using wrong broadcast

```swift
// BEFORE:
return MLX.broadcast(identity.expandedDimensions(axis: 0), to: [batchSize, dim, dim])

// AFTER:
return MLX.stacked(Array(repeating: identity, count: batchSize), axis: 0)
```

### `DistillationTrainer.swift` — remove inline debug prints from the training hot path

The `print("DEBUG: ...")` calls on lines 139–144 and the gradient clipping print run every single step. Gate them behind a flag:

```swift
// Add to DistillationTrainer init params:
private let verbose: Bool

// Replace bare prints with:
if verbose { print("  DEBUG: GT Loss: ...") }
```

### `TrainingMain.swift` — scope the `String * Int` operator

```swift
// BEFORE (global extension):
extension String {
    static func * (string: String, count: Int) -> String { ... }
}

// AFTER (file-private):
private extension String {
    static func * (string: String, count: Int) -> String { ... }
}
```

### `sLSTM.swift` / `mLSTM.swift` — remove `fatalError` from convenience extensions

```swift
// Replace fatalError with a logged fallback or remove the no-throw convenience wrapper entirely.
// Callers should use the throwing initialState(batchSize:) directly.
```

---

## Constraints — do not change these

- Do not change the model's public API (`callAsFunction`, `processSequence`, `initialState`, `generate`).
- Do not change `TrainingConfig` default values.
- Do not upgrade or change any Swift Package dependencies.
- Do not add new files unless a fix strictly requires it (e.g. a conv state buffer struct).
- Do not change the `LayerState` enum cases or their associated value names.
- All fixes must compile cleanly with `swift build -c release` — no warnings promoted to errors.

---

## Reference: key mathematical formulations

### sLSTM (Beck et al. 2024)

```
m_t = max(z_f + m_{t-1}, z_i)
i'_t = exp(z_i - m_t)            # always in (0, 1]
f'_t = exp(z_f + m_{t-1} - m_t)  # always in (0, 1]
n_t  = f'_t * n_{t-1} + i'_t     # normalizer, init = 1.0
c_t  = f'_t * c_{t-1} + i'_t * tanh(W_c * [x; h])
h_t  = sigmoid(W_o * [x; h]) * (c_t / max(n_t, epsilon))
```

### mLSTM (Beck et al. 2024)

```
q_t = W_q * conv_causal(x)       # query from causal-conv feature
k_t = W_k * x / sqrt(d)          # key, length-normalised
v_t = W_v * x                    # value
m_t = max(z_f + m_{t-1}, z_i)
i_t = exp(z_i - m_t)
f_t = exp(z_f + m_{t-1} - m_t)
C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t^T)   # matrix memory
n_t = f_t * n_{t-1} + i_t * k_t
denom    = max(|n_t^T q_t|, 1)
retrieved = GroupNorm(C_t q_t / denom)
h_t = sigmoid(W_o * x) * GELU(W_skip * x) * retrieved
```

---

## Done checklist

- [ ] Fix 1: `n_t` init = 1.0 in `sLSTM.initialState`  
- [ ] Fix 2: causal conv left-pad only in `mLSTM`  
- [ ] Fix 3: no `try!` inside gradient closure  
- [ ] Fix 4: `MLX.eval` before `.item()`  
- [ ] Fix 5: teacher API call removes `state:` argument  
- [ ] Fix 6: teacher model actually frozen  
- [ ] Fix 7: KL T² removed from loss value  
- [ ] Fix 8: mLSTM input dim made explicit with assertion  
- [ ] `./build.sh` exits 0  
- [ ] `./train.sh` exits 0 and prints 5+ loss steps without NaN