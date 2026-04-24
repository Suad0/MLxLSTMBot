# GEMINI.md — MLXSTMBot Project Context
# Place this file in the root of your MLXSTMBot Xcode project.
# Gemini CLI reads it automatically at the start of every session.

---

## Project Identity

**Name:** MLXSTMBot  
**Language:** Swift 5.9+  
**Platform:** macOS (Apple Silicon only — M1/M2/M3/M4)  
**Frameworks:** MLX, MLXNN, MLXOptimizers, MLXLMCommon, MLXLLM, Tokenizers  
**Purpose:** Train an xLSTM language model on Apple Silicon using knowledge
distillation from a Llama teacher model via the MLX framework.

---

## Architecture Reference — Beck et al. (2024)

This project implements **xLSTM: Extended Long Short-Term Memory** (Beck et al. 2024).
All architectural decisions MUST match the paper unless a deviation is explicitly
marked with `// DEVIATION(paper): reason`.

### sLSTM — Correct Equations (§3.2)

```
z_i = W_i [x_t; h_{t-1}] + b_i
z_f = W_f [x_t; h_{t-1}] + b_f
m_t = max(z_f + m_{t-1}, z_i)            ← log-domain stabilizer
i_t = exp(z_i - m_t)                      ← input gate, bounded ≤ 1
f_t = exp(z_f + m_{t-1} - m_t)           ← forget gate, bounded ≤ 1
z_c = tanh(W_c [x_t; h_{t-1}] + b_c)
o_t = sigmoid(W_o [x_t; h_{t-1}] + b_o)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ z_c
n_t = f_t ⊙ n_{t-1} + i_t               ← normalizer state, init 0 NOT 1
h_t = o_t ⊙ (c_t / max(|n_t|, ε))       ← ε = 1.0 per paper
```

**sLSTM Block structure (Figure 3):**
```
x' = x + sLSTM(LayerNorm(x))
y  = x' + GeGLU_FFN(LayerNorm(x'))
```

**sLSTM state tuple:** `(h, c, n, m)` — 4 tensors, all `[batch, hiddenDim]`  
**sLSTM initial state:** all zeros (n_t init = 0.0, NOT 1.0)

### mLSTM — Correct Equations (§3.3)

```
x_up         = W_up LayerNorm(x)           ← up-project: d → 2d
[x_l, x_g]  = split(x_up, axis=-1)        ← split into LSTM and gate streams
q_t          = W_q Conv1d_causal(x_l)      ← query through depthwise causal conv
k_t          = W_k x_l / sqrt(d)           ← scaled key
v_t          = W_v x_l
z_i          = W_i x_l + b_i
z_f          = W_f x_l + b_f
m_t          = max(z_f + m_{t-1}, z_i)
i_t          = exp(z_i - m_t)
f_t          = exp(z_f + m_{t-1} - m_t)
n_t          = f_t ⊙ n_{t-1} + i_t ⊙ k_t  ← normalizer VECTOR [batch, d]
C_t          = f_t⊙C_{t-1} + i_t⊙(v_t⊗k_t^T)  ← matrix memory [batch, d, d]
denom        = max(|n_t^T q_t|, 1)         ← eq. 8, scalar per batch
retrieved    = GroupNorm(C_t q_t / denom)   ← GroupNorm before output gate
o_t          = sigmoid(W_o x_l) ⊙ GeLU(W_skip x_l)  ← skip-connection gate
h_t          = o_t ⊙ retrieved
y            = W_down([h_t ; x_g])          ← down-project: 2d → d
output       = x + y                        ← residual
```

**mLSTM state tuple:** `(h, C, n, m)` — 4 tensors  
**mLSTM initial state:** ALL zeros (C_t = zeros, NOT identity matrix)  
**C_t init note:** identity matrix init is WRONG per paper — use zeros

### Architecture Pattern

The block sequence MUST be configurable, not hardcoded:

```swift
// CORRECT — accept a blockSpec array
public init(vocabSize: Int, hiddenDim: Int, blockSpec: [xLSTMBlockType], ...) throws

// WRONG — do not do this
let blockType = (layerIndex % 2 == 0) ? .mLSTM : .sLSTM
```

Paper's recommended ratio for language models: **7 mLSTM : 1 sLSTM** per 8 layers.  
Default blockSpec for 6 layers: `[.mLSTM, .mLSTM, .mLSTM, .mLSTM, .mLSTM, .sLSTM]`

### Normalization

Use **`MLXNN.LayerNorm`** (mean + variance centering), NOT `RMSNorm`.  
RMSNorm is from the Llama teacher model — it does NOT belong in the xLSTM student.

```swift
// CORRECT
self.preNorm = LayerNorm(dimensions: hiddenDim, eps: 1e-5)

// WRONG
self.preNorm = RMSNorm(normalizedShape: hiddenDim)
```

### FFN Activation (GeGLU, not GLU)

The sLSTM block FFN uses **GeLU** on the gate branch (GeGLU variant):

```swift
// CORRECT — GeGLU
let gate  = MLXNN.gelu(gateProjection(input))
let value = valueProjection(input)
return outputProjection(gate * value)

// WRONG — sigmoid GLU
let gate = MLX.sigmoid(gateProjection(input))
```

---

## Known Bugs — Do NOT Reintroduce

The following bugs were identified in code review. Never reintroduce them:

| # | File | Bug | Status |
|---|------|-----|--------|
| B1 | `sLSTM.swift` | `n_t` initialised to `1.0` — must be `0.0` | OPEN |
| B2 | `mLSTM.swift` | Missing normalizer vector `n_t` in state | OPEN |
| B3 | `mLSTM.swift` | `tanh(C_t @ q_t)` used instead of `C_t @ q_t / denom` | OPEN |
| B4 | `mLSTM.swift` | `C_t` initialised to identity matrix — must be zeros | OPEN |
| B5 | `xLSTMBlock.swift` | mLSTM block missing: up-proj, causal conv, GroupNorm, skip gate, down-proj | OPEN |
| B6 | `FeedForward.swift` | `sigmoid` used in gate — must be `MLXNN.gelu` | OPEN |
| B7 | `xLSTMBlock.swift` | `RMSNorm` used — must be `MLXNN.LayerNorm` | OPEN |
| B8 | `xLSTM.swift` | Hardcoded alternating block pattern — must be `blockSpec` array | OPEN |
| B9 | `DistillationTrainer.swift` | `teacherLogits` not eval'd before gradient closure | OPEN |
| B10 | `DistillationTrainer.swift` | Per-parameter GPU→CPU sync in `clipGradients` loop | OPEN |
| B11 | `ChatDataProvider.swift` | Padding with token `0` = `<unk>`; no loss masking | OPEN |
| B12 | `TrainingMain.swift` | `numIterations = 1` (debug leftover) | OPEN |
| B13 | `sLSTM.swift`, `mLSTM.swift` | `fatalError` inside `callAsFunction` — must be `throws` | OPEN |
| B14 | `RMSNorm.swift` | `weight` declared `let` — may not register as trainable param | OPEN |
| B15 | `xLSTM.swift` | `sampleFromLogits` does `argmax`, not actual sampling | OPEN |

When fixing a bug from this list, change its status to FIXED and add the fix
description as a comment above the relevant code block.

---

## Coding Rules — Swift / MLX

### 1. Error Handling

`callAsFunction` on any `Module` subclass MUST be `throws`. Never use `fatalError`
inside a forward pass — it crashes with no diagnostic during training:

```swift
// CORRECT
public func callAsFunction(_ input: MLXArray, state: (...)) throws -> (...) {
    // errors propagate cleanly to the training loop
}

// WRONG — kills the process during training with no context
fatalError("sLSTM forward pass error: ...")
```

### 2. MLX Lazy Evaluation

Always call `MLX.eval()` to materialise tensors that will be captured in a
subsequent gradient closure, especially teacher logits:

```swift
let teacherLogits = try getTeacherLogits(inputs: inputs)
MLX.eval(teacherLogits)          // ← required before building student graph
let lossAndGrad = MLXNN.valueAndGrad(model: studentModel) { ... }
```

Call `MLX.eval(model, optimizer)` once per training step, not per-timestep.

### 3. Gradient Clipping — Fused Norm

Never loop and call `.item(Float.self)` per-parameter — this forces a CPU
synchronisation for every parameter tensor. Compute the global norm in a single
fused operation:

```swift
// CORRECT
let flatGrads   = gradients.flattened().map { $0.1 }
let squaredNorms = flatGrads.map { MLX.sum($0 * $0) }
let totalNorm   = MLX.sqrt(squaredNorms.reduce(MLXArray(0.0), +))
MLX.eval(totalNorm)
let normValue   = totalNorm.item(Float.self)   // single CPU sync

// WRONG — N CPU syncs for N parameter tensors
for (_, grad) in gradients.flattened() {
    let gradNorm = MLX.sum(grad * grad).item(Float.self)
    ...
}
```

### 4. Sequence Processing

Avoid per-timestep Swift `for` loops that dispatch one MLX op at a time.
Process in chunks of ~32 timesteps and call `MLX.eval` at chunk boundaries
to allow graph fusion:

```swift
let chunkSize = 32
for chunkStart in stride(from: 0, to: sequenceLength, by: chunkSize) {
    let chunkEnd = min(chunkStart + chunkSize, sequenceLength)
    for t in chunkStart..<chunkEnd {
        let x = sequence[0..., t, 0...]
        let (out, newState) = try step(x, state: currentState)
        outputs.append(out)
        currentState = newState
    }
    MLX.eval(outputs.last!, currentState.hiddenState)
}
```

### 5. Trainable Parameters

All learnable tensors on `Module` subclasses must be `var` and decorated with
`@ParameterInfo` (or at minimum declared `var`, not `let`) so MLX's reflection-
based parameter collection includes them in `model.parameters()`:

```swift
// CORRECT
@ParameterInfo var weight: MLXArray

// WRONG — may be invisible to optimizer
public let weight: MLXArray
```

### 6. No Dead-Code Print Statements

Remove all `print("Using default Linear layer initialization...")` calls from
`initializeWeights()`. These fire on every model construction and pollute logs.
Either delete the method or actually implement custom init.

### 7. Configuration

`TrainingConfig` must use `var` properties, not `let`, and accept external
overrides. Never commit `numIterations = 1`:

```swift
// CORRECT
struct TrainingConfig {
    var numIterations: Int = 1000
    var batchSize: Int = 4
    var hiddenDim: Int = 512
    var numLayers: Int = 6
    var learningRate: Float = 1e-4
}

public init(config: TrainingConfig = TrainingConfig()) { ... }
```

### 8. Padding Tokens

Never pad sequences with token ID `0`. Token `0` is `<unk>` in Llama's tokenizer.
Always use the tokenizer's EOS token and mask it out of the loss:

```swift
let padId = Int32(tokenizer.eosTokenId ?? 2)
// ...
let mask = (flatTargets .!= padId).asType(.float32)
let loss = (MLXNN.crossEntropy(...) * mask).sum() / (mask.sum() + 1e-8)
```

### 9. Sampling

`sampleFromLogits` must use `MLX.random.categorical`, not `argmax`. `argmax`
is greedy decoding and ignores temperature and top-k entirely:

```swift
// CORRECT
return MLX.random.categorical(logits)   // true multinomial sampling

// WRONG — greedy, temperature has no effect
return MLX.argMax(probs, axis: -1)
```

### 10. Parameter Counting

Never estimate parameter counts by formula in display methods. Use the actual
model:

```swift
let actualParams = studentModel.parameters()
    .flattened()
    .map { $0.1.size }
    .reduce(0, +)
print("Parameters: \(actualParams / 1_000_000)M")
```

---

## File Map

```
MLXSTMBot/
├── GEMINI.md               ← this file (project context for Gemini CLI)
├── sLSTM.swift             ← sLSTM cell (exponential gating)
├── mLSTM.swift             ← mLSTM cell (matrix memory) — MAJOR BUGS, see B2-B4
├── xLSTMBlock.swift        ← Block wrapper — MAJOR BUG B5 (missing mLSTM structure)
├── xLSTM.swift             ← Full model + generation
├── RMSNorm.swift           ← Wrong norm type — replace with LayerNorm
├── FeedForward.swift       ← GeGLU FFN — activation bug B6
├── LSTMUtils.swift         ← Shared utilities, validation, epsilon constants
├── ChatDataProvider.swift  ← JSON data loader + tokenizer — padding bug B11
├── DistillationTrainer.swift ← KL distillation training step — bugs B9, B10
├── TrainingMain.swift      ← Training pipeline entry — bug B12
└── main.swift              ← CLI entry point (train / test / interactive modes)
```

---

## Correct Parameter Counts (hiddenDim=512, vocabSize=128256, 6 layers)

Use these to verify your implementation after fixing the mLSTM block structure:

| Component | Count |
|-----------|-------|
| Token Embedding | 65,667,072 |
| 3× mLSTM blocks (paper-compliant) | 13,387,776 |
| 3× sLSTM cells | 12,595,200 |
| 3× GeGLU FFN | 9,451,008 |
| 12× LayerNorm | 12,288 |
| LM Head | 65,795,328 |
| **Total** | **~167M** |

README states ~67M — this is wrong by 2.5×. Update after fixing B5.

---

## What Gemini Should Always Do

- Check the Known Bugs table before writing any code touching those files.
- Match every gate equation, state shape, and activation to the paper formulas above.
- Use `throws` on all `callAsFunction` overrides — never `fatalError`.
- Use `MLXNN.LayerNorm`, not `RMSNorm`, for all block normalisation.
- Use `MLXNN.gelu`, not `MLX.sigmoid`, in the FFN gate branch.
- Call `MLX.eval(teacherLogits)` before any gradient closure that captures it.
- Pad with EOS token, not token 0, and apply a loss mask.
- Accept `blockSpec: [xLSTMBlockType]` instead of computing a ratio in a loop.

## What Gemini Should Never Do

- Do NOT initialise `n_t` to `1.0` — always `0.0`.
- Do NOT initialise `C_t` to an identity matrix — always zeros.
- Do NOT use `tanh(C_t @ q_t)` as the mLSTM output — use the normalised form.
- Do NOT use `fatalError` inside any `Module.callAsFunction`.
- Do NOT call `.item(Float.self)` inside a parameter loop.
- Do NOT hardcode a `%2` alternating block pattern.
- Do NOT add `print(...)` statements inside module `init` or forward passes.
- Do NOT commit `numIterations = 1`.
- Do NOT use `MLX.argMax` in `sampleFromLogits`.
- Do NOT use `RMSNorm` anywhere in the xLSTM student model.
- Do NOT import `AppKit` in `main.swift` (prevents non-GUI / Linux builds).

---

## Knowledge Distillation Setup

**Teacher:** `mlx-community/Llama-3.2-1B-Instruct-4bit`  
**Student:** xLSTM (this project)  
**Loss:** `L = 0.3 × CrossEntropy(student, targets) + 0.7 × KL(student/T, teacher/T) × T²`  
**Temperature:** `T = 2.0`  
**Optimizer:** AdamW, lr = 1e-4  
**Vocab size:** 128,256 (must match teacher exactly — assert this before training)

The KL divergence implementation in `DistillationTrainer.computeKLDivergence`
is **correct** — do not modify it.

---

## MLX-Specific Reminders

- MLX uses **lazy evaluation** — operations build a graph and execute only when
  `.item()`, `MLX.eval()`, or I/O is called.
- All tensors live in **unified memory** — there is no `.to(device)` or `.cuda()`.
- `MLXNN.valueAndGrad(model:_:)` returns `(loss, gradients)` where gradients
  have the same nested structure as `model.parameters()`.
- `ModuleParameters.mapValues` applies a transform to every leaf tensor while
  preserving the nested structure — use it for gradient scaling.
- `MLX.random.categorical(logits)` samples from a categorical distribution
  given unnormalised log-probabilities (logits directly, not softmax output).
- `MLXNN.GroupNorm` is available — use it for the mLSTM retrieval normalisation.
- `MLXNN.Conv1d` supports `groups` parameter for depthwise convolution.
