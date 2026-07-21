# ONNX Runtime GenAI — Architecture & Design

> A comprehensive guide to the architecture, design decisions, trade-offs, and improvement roadmap for the ONNX Runtime Generative AI library.

---

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Inference Pipeline](#core-inference-pipeline)
4. [Component Deep Dives](#component-deep-dives)
   - [Model Abstraction & State Pattern](#model-abstraction--state-pattern)
   - [Device / Hardware Backend Abstraction](#device--hardware-backend-abstraction)
   - [KV Cache Management](#kv-cache-management)
   - [Search & Sampling](#search--sampling)
   - [Logits Processing Pipeline](#logits-processing-pipeline)
   - [Tokenizer Integration](#tokenizer-integration)
   - [Multi-Modal Support](#multi-modal-support)
   - [LoRA / Adapter Support](#lora--adapter-support)
   - [Constrained Generation (Guidance)](#constrained-generation-guidance)
   - [Request Batching Engine](#request-batching-engine)
   - [Configuration System](#configuration-system)
   - [Performance Tracing](#performance-tracing)
5. [API Layers & Language Bindings](#api-layers--language-bindings)
6. [Error Handling](#error-handling)
7. [Memory Management](#memory-management)
8. [Build System](#build-system)
9. [Testing Strategy](#testing-strategy)
10. [Design Trade-Offs](#design-trade-offs)
11. [Improvement Opportunities & Roadmap](#improvement-opportunities--roadmap)

---

## Overview

ONNX Runtime GenAI is a high-performance inference library that implements the complete generative AI loop: **tokenization → model inference → logits processing → search/sampling → KV cache management → token decoding**. It supports decoder-only LLMs, encoder-decoder models (Whisper, Marian), and multi-modal models (vision, audio) across six hardware backends.

### Key Design Goals

| Goal | Approach |
|------|----------|
| **Cross-platform** | C core with bindings for Python, C#, Java, Objective-C |
| **Hardware flexibility** | DeviceInterface abstraction over CUDA, DML, CPU, QNN, OpenVINO, WebGPU |
| **Model agnostic** | State pattern with per-architecture implementations |
| **Production ready** | C API boundary, RAII memory management, leak detection |

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Language Bindings                          │
│   Python (pybind)  │  C# (P/Invoke)  │  Java (JNI)  │  ObjC   │
├─────────────────────────────────────────────────────────────────┤
│                     C API  (ort_genai_c.h)                      │
│              Stable ABI · OgaResult error boundary              │
├─────────────────────────────────────────────────────────────────┤
│                   C++ Core  (generators.h)                      │
│          Generator · GeneratorParams · Model · Tokenizer        │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│   Models     │   Search     │   Engine     │   Config           │
│  (State      │  (Greedy,    │  (Scheduler, │  (JSON parser,     │
│   pattern)   │   Beam,      │   Batching,  │   runtime overlay) │
│              │   Sampling)  │   Paged KV)  │                    │
├──────────────┴──────────────┴──────────────┴────────────────────┤
│               Device Interface Abstraction                      │
│  CUDA  │  DirectML  │  CPU  │  QNN  │  OpenVINO  │  WebGPU     │
├─────────────────────────────────────────────────────────────────┤
│                   ONNX Runtime (libonnxruntime)                  │
│            Session · ExecutionProvider · Allocator               │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── generators.h/cpp        # Core Generator orchestration
├── ort_genai_c.h/cpp       # Stable C API boundary
├── ort_genai.h             # Zero-cost C++ wrapper over C API
├── config.h/cpp            # genai_config.json parsing
├── search.h/cpp            # Greedy/beam search, sampling
├── beam_search_scorer.h    # Beam hypothesis management
├── constrained_logits_processor.h/cpp  # Guidance integration
├── models/                 # Model implementations
│   ├── model.h/cpp         # Base Model class, device init
│   ├── decoder_only.h/cpp  # Decoder-only LLMs (GPT, Phi, Llama)
│   ├── gpt.h/cpp           # GPT-style combined KV cache
│   ├── whisper.h/cpp       # Encoder-decoder (speech)
│   ├── marian.h/cpp        # Encoder-decoder (translation)
│   ├── kv_cache.h/cpp      # Default & combined KV cache
│   ├── windowed_kv_cache.h/cpp   # Sliding window KV cache
│   ├── logits.h/cpp        # Logits extraction & type conversion
│   ├── input_ids.h/cpp     # Token sequence management
│   ├── position_inputs.h   # Position IDs & attention masks
│   ├── adapters.h/cpp      # LoRA adapter management
│   └── processor.h         # Multi-modal processor base
├── engine/                 # Request batching
│   ├── engine.h            # AddRequest / Step loop
│   ├── scheduler.h/cpp     # Static & dynamic batch scheduling
│   ├── request.h/cpp       # Request lifecycle tracking
│   └── paged_key_value_cache.h/cpp  # Paged KV allocation
├── cuda/                   # CUDA device interface & kernels
├── dml/                    # DirectML device interface
├── cpu/                    # CPU reference implementation
├── qnn/                    # Qualcomm Neural Network backend
├── openvino/               # Intel OpenVINO backend
├── webgpu/                 # Browser GPU backend
├── python/                 # Python binding (C++ + setup.py)
├── csharp/                 # C# binding (P/Invoke + .csproj)
├── java/                   # Java binding (JNI + Gradle)
└── objectivec/             # Objective-C binding (Framework)
```

---

## Core Inference Pipeline

The generation loop is orchestrated by the `Generator` class (`src/generators.h`):

```
1. Model::Create(path)             → Load config, create ORT sessions
2. Tokenizer::Create(model)        → Initialize tokenizer from model dir
3. GeneratorParams::Create(model)  → Configure search params (top-k, top-p, etc.)
4. Generator::Create(model, params)
   │
   ├── State::Create()             → Allocate KV cache, position inputs
   │
   └── Generation Loop:
       │
       ├── ComputeLogits()         → Run ORT session (prefill or decode)
       │   └── State::Run()        → Feed input_ids → get logits
       │
       ├── ProcessLogits()         → Apply processors chain
       │   ├── RepetitionPenalty
       │   ├── MinLength enforcement
       │   └── Guidance constraints (optional)
       │
       ├── GenerateNextToken()     → Search / sampling
       │   ├── Greedy: argmax or top-k/top-p sampling
       │   └── Beam: expand, score, select top-2K hypotheses
       │
       └── State::Update()         → Move present→past KV cache, update positions
           └── Loop until IsDone() (all sequences hit EOS or max_length)
```

### Prefill vs Decode

- **Prefill**: Process full prompt in one forward pass (variable seq_len)
- **Decode**: Process one token at a time (seq_len=1), leverage KV cache
- Transition is transparent—`State::Run()` handles both cases via input shape

---

## Component Deep Dives

### Model Abstraction & State Pattern

**Pattern**: Each model architecture extends the base `State` class, composing reusable sub-components.

```
State (virtual base)
  ├── DecoderOnly_State    → GPT-style decoder-only LLMs
  ├── Gpt_State            → Models with combined KV cache layout
  ├── Whisper_State         → Encoder-decoder (audio → text)
  ├── Marian_State          → Encoder-decoder (text → text)
  └── DecoderOnlyPipeline_State → Split encoder/decoder pipeline
```

**Composition over inheritance**: Each State aggregates specialized components:

| Component | Responsibility |
|-----------|---------------|
| `InputIDs` | Manages token sequences per batch element |
| `Logits` | Extracts model output logits, handles fp16→fp32 conversion |
| `KeyValueCache` | Past/present KV tensor lifecycle |
| `PositionInputs` | Position IDs and attention mask generation |
| `ExtraInputs` | User-provided optional tensors |

**Model loading pipeline**:
1. Parse `genai_config.json` → determine model type, architecture, vocab size
2. Select execution provider (CUDA > DML > CPU) based on config
3. Create ONNX Runtime session(s) with appropriate provider options
4. Initialize device-specific State with KV cache pre-allocation

**Key files**: `src/models/model.h`, `src/models/decoder_only.h`, `src/models/gpt.h`

---

### Device / Hardware Backend Abstraction

Each backend implements `DeviceInterface` (declared in `src/smartptrs.h`):

```cpp
struct DeviceInterface {
  virtual std::unique_ptr<DeviceBuffer> Allocate(size_t size) = 0;
  virtual void CopyToDevice(DeviceSpan<T> dst, std::span<const T> src) = 0;
  virtual DeviceSpan<T> WrapMemory(std::span<T>) = 0;
  virtual Ort::Allocator& GetAllocator() = 0;
  // Device-specific ops: UpdatePositionIds, UpdateAttentionMask, Cast, etc.
};
```

**Six backends**:

| Backend | Directory | Notes |
|---------|-----------|-------|
| CUDA | `src/cuda/` | Custom kernels for topk, softmax, sampling |
| DirectML | `src/dml/` | Command queues, execution contexts, shader compilation |
| CPU | `src/cpu/` | Reference implementation, always available |
| QNN | `src/qnn/` | Qualcomm Neural Network for mobile/edge |
| OpenVINO | `src/openvino/` | Intel optimized inference |
| WebGPU | `src/webgpu/` | Browser-based GPU compute |

**Three-device architecture**: The model may use different devices for different roles:

- `p_device_` — Main computation device (runs ORT session)
- `p_device_inputs_` — Input preparation device (often CPU for DML/WebGPU)
- `p_device_kvcache_` — KV cache allocation device

This separation enables hybrid execution where inputs are prepared on CPU but computed on GPU, avoiding unnecessary device transfers for tensor construction.

**Device selection priority**: CUDA → DML → CPU (fallback). Configured via `session_options.providers` in `genai_config.json`.

---

### KV Cache Management

Four KV cache implementations serve different use cases:

| Cache Type | File | Use Case |
|-----------|------|----------|
| `DefaultKeyValueCache` | `kv_cache.h/cpp` | Separate past/present buffers, standard decoder-only |
| `CombinedKeyValueCache` | `kv_cache.h/cpp` | Single concatenated K/V tensor, GPT-style |
| `WindowedKeyValueCache` | `windowed_kv_cache.h/cpp` | Sliding window for long-context models |
| `PagedKeyValueCache` | `engine/paged_key_value_cache.h` | Block-allocated for dynamic batching |

**Update flow per decode step**:
1. `Add()` — Register past/present tensor names in ORT session I/O
2. `Update()` — After model run, move present→past (with optional beam reordering via `beam_indices`)
3. `RewindTo()` — Truncate past tensors to restore a prior state (session continuation)

**Beam search reordering**: When beams are selected, `PickPastState` reorders KV cache buffers to match the new beam assignment, maintaining hypothesis consistency.

**Memory layout**: `[batch × num_beams, num_heads, seq_len, head_size]` — standard transformer KV shape.

---

### Search & Sampling

**Strategy hierarchy** (`src/search.h`):

| Strategy | Implementation | Method |
|----------|---------------|--------|
| Greedy | `GreedySearch_Cpu` / `GreedySearch_Cuda` | Argmax |
| Top-K Sampling | `GreedySearch_Cpu::SampleTopK` | Filter to K highest logits, sample |
| Top-P (Nucleus) | `GreedySearch_Cpu::SampleTopP` | Cumulative probability threshold |
| Top-K + Top-P | `GreedySearch_Cpu::SampleTopKTopP` | Combined filtering |
| Beam Search | `BeamSearch_Cpu` | Expand → Score → Select top-2K |

**Beam search** uses `BeamSearchScorer` (`src/beam_search_scorer.h`) which tracks `BeamHypotheses` — sorted hypothesis pools with early stopping via `CanImprove()`.

**Constraint processors** are applied before token selection:
- `ApplyMinLength()` — Suppress EOS before minimum token count
- `ApplyRepetitionPenalty()` — Penalize tokens already in the sequence
- `ProcessLogits()` (Guidance) — Mask tokens that violate grammar/schema constraints

---

### Logits Processing Pipeline

```
Raw Logits (ORT session output, possibly fp16)
  │
  ├── Logits::Update()           → Shape normalization, fp16→fp32 cast
  │                                 Extract last-token logits for batched prefill
  │
  ├── Search::ApplyMinLength()   → Suppress EOS if below min_length
  ├── Search::ApplyRepetitionPenalty()  → Penalize repeated tokens
  │
  ├── GuidanceLogitsProcessor::ProcessLogits()  → Constrained masking (optional)
  │
  └── Search::SelectTop()        → Final token selection (greedy/beam/sampling)
      └── GetNextTokens()
```

The `Logits` class (`src/models/logits.h`) handles the critical transition between ORT session output and the search layer, including:
- Trimming logits to only the last position for efficient autoregressive decoding
- Type conversion when the model outputs fp16 but search operates in fp32

---

### Tokenizer Integration

Built on **ONNX Runtime Extensions** (`ortx_tokenizer.h`) for HuggingFace-compatible tokenization:

- **`Tokenizer`** — Encode text→tokens, decode tokens→text, chat template support
- **`TokenizerStream`** — Incremental decoding for streaming output (maintains internal byte buffer for partial UTF-8 sequences)

Created via `Model::CreateTokenizer()` as a shared instance. Special token IDs (BOS, EOS, PAD) are extracted from the tokenizer config.

---

### Multi-Modal Support

Multi-modal models use a `Processor` abstraction (`src/models/processor.h`):

```
Processor (abstract base)
  ├── PhiMultiModalProcessor  → Vision + Audio (Phi-3.5-vision)
  ├── GemmaImageProcessor     → Vision only (Gemma)
  └── WhisperProcessor        → Audio only (Whisper)
```

**Processing flow**:
1. `Processor::Process(tokenizer, payload)` — Accept text + images/audio
2. Image → pixel values via `OrtxProcessor`; Audio → Mel spectrogram via `OrtxFeatureExtractor`
3. Returns `NamedTensors` map of all required model inputs

Image token placeholders are inserted into the token stream at positions corresponding to image regions, with special handling for interleaved text-image sequences.

---

### LoRA / Adapter Support

**File**: `src/models/adapters.h/cpp`

- **`Adapter`** — Reference-counted wrapper around `OrtLoraAdapter`
  - `AcquireRef()` / `ReleaseRef()` for lifetime management
- **`Adapters`** (manager) — Per-model adapter registry
  - `LoadAdapter(path, name)` / `UnloadAdapter(name)`
  - Storage: `std::unordered_map<std::string, std::unique_ptr<Adapter>>`
- **State integration**: `SetActiveAdapter()` binds a LoRA adapter to the current ORT session

**Limitation**: Only one adapter active per session at a time. Multi-LoRA serving requires separate sessions.

---

### Constrained Generation (Guidance)

**File**: `src/constrained_logits_processor.h/cpp`

When `USE_GUIDANCE=ON`, the `GuidanceLogitsProcessor` integrates with the `llguidance` library to enforce JSON schema, regex, or grammar constraints:

1. **Mask computation**: Generates a token mask indicating which tokens are valid at each step
2. **Logits masking**: Sets invalid token logits to `-inf` before sampling
3. **State tracking**: `CommitTokens()` advances the constraint automaton after each token
4. **Async masking**: `mask_future_` allows parallel mask computation for the next step

---

### Request Batching Engine

**Files**: `src/engine/`

The engine layer manages concurrent inference requests:

```
Engine
  ├── AddRequest(tokens, params)  → Queue request
  ├── Step()                      → Execute one batch
  └── RemoveRequest(id)           → Cancel/complete
         │
         ▼
    Scheduler
      ├── StaticBatchScheduler    → Fixed batch, pad to longest
      └── DynamicBatchScheduler   → (In development)
         │
         ▼
    ScheduledRequests             → Grouped batch for execution
```

**Request lifecycle**: `Unassigned → Assigned → InProgress → Completed`

Each `Request` tracks input tokens, generated tokens, and seen/unseen token bookkeeping for efficient incremental decode.

**Paged KV Cache** (`engine/paged_key_value_cache.h`): Block-allocated memory for per-request KV storage, enabling dynamic memory sharing across requests without pre-allocation.

---

### Configuration System

**File**: `src/config.h/cpp`, `src/json.h/cpp`

Models are configured via `genai_config.json` in the model directory:

```json
{
  "model": {
    "type": "phi3",
    "vocab_size": 32064,
    "context_length": 4096,
    "decoder": {
      "filename": "model.onnx",
      "num_hidden_layers": 32,
      "inputs": { ... },
      "outputs": { ... }
    }
  },
  "search": {
    "max_length": 2048,
    "top_p": 0.9,
    "top_k": 50,
    "temperature": 0.7
  },
  "session_options": {
    "providers": ["CUDAExecutionProvider"]
  }
}
```

**Runtime overlay**: A JSON string can be passed at construction time to override config values without modifying the file on disk.

**Custom JSON parser**: The codebase uses a hand-rolled event-driven JSON parser (`src/json.h/cpp`) rather than an external library, minimizing dependencies at the cost of reduced validation.

---

### Performance Tracing

**File**: `src/tracing.h/cpp`

- **Compile-time gated**: Zero overhead when `ENABLE_TRACING=OFF`
- **RAII wrapper**: `DurationTrace` for scoped timing measurement
- **Output**: Perfetto-compatible JSON format (configured via `ORTGENAI_TRACE_FILE_PATH` env var)
- **Thread-local**: Uses TLS for per-thread trace sinks

---

## API Layers & Language Bindings

All language bindings wrap the stable C API (`ort_genai_c.h`):

```
C++ Application
  └── ort_genai.h          (Zero-cost C++ wrapper, OgaCheckResult)

Python Application
  └── python.cpp           (CPython C extension, PyType_Spec)
      └── ort_genai_c.h

C# Application
  └── NativeMethods.cs     (P/Invoke declarations)
      └── libonnxruntime_genai.so/dll

Java Application
  └── JNI native/*.cpp     (JNI bridge, long nativeHandle)
      └── ort_genai_c.h

Objective-C Application
  └── ObjC wrappers        (NSObject + ARC)
      └── ort_genai_c.h
```

The C API provides a stable ABI with version checking, enabling binary-compatible updates without recompiling bindings.

---

## Error Handling

**Cross-boundary pattern**: C++ exceptions are caught at the C API boundary and converted to `OgaResult*` error objects:

```cpp
// C API implementation (ort_genai_c.cpp)
OgaResult* OgaSomeFunction(...) {
  OGA_TRY {
    // C++ code that may throw
  } OGA_CATCH  // catch → OgaResult with error message
}

// C++ consumer (ort_genai.h)
inline void OgaCheckResult(OgaResult* result) {
  if (result) {
    std::unique_ptr<OgaResult> p{result};
    throw std::runtime_error(p->GetError());  // Re-throw as C++ exception
  }
}
```

This ensures exceptions never cross shared library boundaries (which is undefined behavior in C++), while giving C++ callers idiomatic exception handling.

---

## Memory Management

### Core Abstractions (`src/span.h`, `src/tensor.h`)

| Type | Purpose |
|------|---------|
| `DeviceSpan<T>` | Non-owning view of device memory (CPU or GPU) |
| `DeviceBuffer<T>` | Owning allocation with RAII cleanup |
| `Tensor` | Named tensor with shape, type, and device allocation |

**Device memory protocol**:
- `DeviceSpan::CpuSpan()` — Force copy to CPU for host-side access
- `DeviceSpan::CopyDeviceToCpu()` — Explicit transfer
- `Tensor::MakeStatic()` — Pin allocation for reuse across iterations

### Leak Detection (`src/leakcheck.h`)

Debug builds use `LeakChecked<T>` — a CRTP base that maintains a global atomic counter per type. On shutdown, non-zero counts indicate leaked objects. This covers core types (`Generator`, `Model`, `Tokenizer`, etc.) without the overhead of full memory tracking.

---

## Build System

### Three-Layer Structure

```
build.py                    Python driver (--update, --build, --test phases)
  └── CMakeLists.txt        Top-level CMake orchestration
        ├── cmake/options.cmake    Feature flags (USE_CUDA, USE_DML, etc.)
        ├── cmake/ortlib.cmake     ONNX Runtime discovery & linking
        ├── cmake/nuget.cmake      NuGet package management (Windows)
        ├── cmake/external/        External dependencies (OpenCV, etc.)
        └── src/CMakeLists.txt     Core library + per-binding sub-projects
```

### Key Build Flags (`cmake/options.cmake`)

| Flag | Default | Purpose |
|------|---------|---------|
| `USE_CUDA` | ON | NVIDIA GPU support |
| `USE_DML` | OFF | DirectML (Windows) |
| `USE_ROCM` | OFF | AMD GPU support |
| `USE_TRT_RTX` | OFF | TensorRT provider |
| `USE_GUIDANCE` | OFF | Constrained generation (llguidance) |
| `ENABLE_PYTHON` | ON | Python wheel build |
| `ENABLE_JAVA` | OFF | Java/Android JNI |
| `ENABLE_TRACING` | OFF | Performance tracing |

### ORT Dependency Resolution

Three mechanisms (in priority order):
1. **`ORT_HOME`** env var — Pre-built ORT artifacts (CI pipelines)
2. **CMake auto-download** — `cmake/ortlib.cmake` fetches from ORT-Nightly NuGet feed
3. **Python build driver** — `tools/python/util/dependency_resolver.py` downloads NuGet packages

---

## Testing Strategy

### Current Coverage

| Area | Location | Framework |
|------|----------|-----------|
| C++ core API | `test/model_tests.cpp`, `test/c_api_tests.cpp` | CTest |
| Sampling algorithms | `test/sampling_tests.cpp` | CTest |
| CUDA kernels | `test/cuda_kernel/` | CTest + benchmarks |
| Python API | `test/python/test_onnxruntime_genai_api.py` | pytest |
| Python E2E | `test/python/test_onnxruntime_genai_e2e.py` | pytest |
| Mobile (Android/iOS) | Platform-specific test projects | Emulator/Simulator |

### CI Matrix

GitHub Actions workflows cover:
- Linux CPU x64, Linux GPU (CUDA) x64
- Windows CPU x64, Windows CUDA x64, Windows DirectML x64
- macOS ARM64
- Android, iOS

---

## Design Trade-Offs

### 1. State Pattern vs Monolithic Model

**Choice**: Separate `State` class per model architecture, composed from reusable components.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Clean separation of model-specific logic | Virtual dispatch overhead on every `Run()` |
| Reusable components (InputIDs, KV cache) | New model types require full State implementation |
| Easy to add new architectures | State lifecycle complexity (multiple init phases) |

### 2. Custom JSON Parser vs External Library

**Choice**: Hand-rolled event-driven JSON parser (`src/json.h/cpp`).

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Zero external dependencies for config parsing | No Unicode escape sequence support |
| Small binary size impact | Limited error reporting |
| Sufficient for config files | No schema validation |
| | Maintenance burden vs battle-tested libraries |

### 3. Static Batching vs Continuous Batching

**Choice**: `StaticBatchScheduler` is the primary (only mature) scheduler.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Predictable memory usage | Padding waste for variable-length sequences |
| Simple implementation | Poor throughput for heterogeneous workloads |
| Deterministic timing | No iteration-level scheduling |

### 4. Three-Device Architecture

**Choice**: Separate device interfaces for compute, inputs, and KV cache.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Enables hybrid CPU+GPU execution | Three pointers to manage per model |
| Avoids unnecessary CPU→GPU copies for input construction | Complexity in deciding which device handles each tensor |
| Allows KV cache on different memory tier | Refactoring needed (TODO: "Remove in favor of just p_device_?") |

### 5. C API Boundary

**Choice**: Stable C API (`ort_genai_c.h`) between core library and all bindings.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Stable ABI across compiler versions | Manual handle management |
| Safe exception boundary (no UB from cross-library throws) | Every new feature needs C API surface |
| Single implementation serves all language bindings | Performance overhead of indirection (minimal) |

### 6. Monolithic Library vs Per-Backend Libraries

**Choice**: Single `libonnxruntime_genai` with all backends compiled in.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Simple deployment — one library to distribute | Binary includes unused backend code |
| Runtime provider selection | Longer build times for all-backend builds |
| No dynamic loading complexity | Large binary size on platforms with many backends |

### 7. CPU Reference Implementation for Search

**Choice**: CPU-side greedy/beam search as the default, with optional CUDA kernels.

| ✅ Benefit | ⚠️ Cost |
|-----------|---------|
| Always correct baseline | CPU↔GPU transfer for logits on every step |
| Easier debugging and testing | Sampling is sequential — bottleneck at scale |
| No CUDA dependency for basic functionality | Beam search reordering on CPU is slow for large batches |

---

## Improvement Opportunities & Roadmap

### 🔴 High Priority

#### 1. Continuous Batching Scheduler

**Problem**: `StaticBatchScheduler` pads all sequences to the longest length, wasting compute and memory. `DynamicBatchScheduler` exists but is incomplete.

**Plan**:
- Implement iteration-level scheduling where new requests can join mid-generation
- Integrate with `PagedKeyValueCache` for per-request block allocation
- Add preemption support — pause low-priority requests when memory is constrained
- Benchmark: Target >2× throughput improvement for heterogeneous request lengths

**Files to modify**: `src/engine/scheduler.h/cpp`, `src/engine/engine.h`

#### 2. Async / Streaming Generator API

**Problem**: The `Generator` API is synchronous. `TokenizerStream` provides incremental decoding but the generation loop itself blocks the caller. `RewindToLength()` is limited to batch_size=1.

**Plan**:
- Add `Generator::GenerateAsync()` returning a future/callback interface
- Expose streaming at the C API level (callback-based token delivery)
- Remove batch_size=1 restriction on `RewindToLength()` — extend to per-sequence rewind
- Add cancellation token support for long-running generations

**Files to modify**: `src/generators.h/cpp`, `src/ort_genai_c.h/cpp`

#### 3. Multi-RoPE and Long-Context Support

**Problem**: TRT-RTX and DML providers lack multi-RoPE factor support, blocking models like Phi-3 with extended context (see TODOs in `src/generators.cpp` lines 448–455). Current workaround only handles batch_size=1, num_beams=1, non-multimodal, CPU/CUDA.

**Plan**:
- Implement multi-RoPE in DML shader pipeline
- Extend position ID recomputation to batch_size>1 and beam search
- Add support for multimodal models with extended context
- Coordinate with ONNX Runtime team on TRT-RTX provider updates

**Files to modify**: `src/generators.cpp`, `src/dml/interface.cpp`, `src/models/position_inputs.h`

### 🟡 Medium Priority

#### 4. Engine & Scheduler Test Coverage

**Problem**: Zero test coverage for the engine layer — no tests for `StaticBatchScheduler`, `DynamicBatchScheduler`, request lifecycle, or paged KV cache allocation.

**Plan**:
- Add unit tests for `StaticBatchScheduler` with mock requests
- Add integration tests for `Engine::AddRequest()` → `Step()` → completion flow
- Add stress tests for `PagedKeyValueCache` block allocation/deallocation
- Add multi-request concurrency tests with different sequence lengths
- Target: >80% line coverage for `src/engine/`

**New files**: `test/engine_tests.cpp`, `test/scheduler_tests.cpp`

#### 5. Search Performance Optimization

**Problem**: Beam search scoring is sequential (TODO: "use thread pool to parallel"). Top-K uses full sort instead of partial sort (TODO: "Optimize this top k with partial sort").

**Plan**:
- Replace full sort with `std::partial_sort` or `std::nth_element` for top-K — O(n) average vs O(n log n)
- Parallelize beam hypothesis scoring using the existing worker thread infrastructure
- Move greedy search fully to GPU to eliminate per-step CPU↔GPU logits transfer
- Benchmark on large vocab (128K+ tokens) models where sorting cost dominates

**Files to modify**: `src/search.cpp`, `src/beam_search_scorer.cpp`

#### 6. DML Device-Side Operations

**Problem**: DML tensor zeroing goes through CPU (TODO in `src/dml/interface.cpp`: "Implement a zeroing that runs directly on DML vs going through CPU"), causing unnecessary round-trips.

**Plan**:
- Implement DirectML `FillValueOfShape` or custom clear shader for zero-initialization
- Audit all DML↔CPU transfers and eliminate unnecessary ones
- Add DML-side beam index reordering for KV cache

**Files to modify**: `src/dml/interface.cpp`

#### 7. JSON Parser Hardening

**Problem**: Custom JSON parser lacks Unicode escape support (TODO: "Currently we ignore these"), has no schema validation, and limited error messages.

**Plan**:
- **Option A** (minimal): Add Unicode escape sequence parsing, improve error messages with line/column info
- **Option B** (recommended): Replace with a lightweight single-header library (e.g., simdjson or nlohmann/json) — reduces maintenance burden and adds schema validation capability
- Add config validation layer that checks required fields and value ranges before model loading

**Files to modify**: `src/json.h/cpp` or `cmake/external/` (if adopting library)

### 🟢 Lower Priority

#### 8. Backend Test Coverage Expansion

**Problem**: No dedicated tests for DML, QNN, OpenVINO, or WebGPU backends. Issues in these backends are caught only in CI with full model tests.

**Plan**:
- Create mock-based unit tests for each `DeviceInterface` implementation
- Test device memory operations (allocate, copy, zero) independently
- Add backend-specific kernel correctness tests (DML shaders, QNN delegates)

#### 9. Multi-LoRA Serving

**Problem**: Only one LoRA adapter active per session. Multi-LoRA serving requires duplicate sessions, wasting base model memory.

**Plan**:
- Investigate ORT session-level adapter switching without full session recreation
- Implement adapter pooling — pre-load N adapters, switch at request level
- Integrate with engine scheduler for per-request adapter assignment

**Files to modify**: `src/models/adapters.h/cpp`, `src/engine/request.h`

#### 10. Examples & Documentation Expansion

**Problem**: While core examples exist (streaming chat, batch generation, multi-modal, continuous batching with engine, whisper, guidance), some advanced patterns remain undocumented.

**Existing examples**:
- `model-chat.py` — Streaming token-by-token output with `TokenizerStream` + session continuation via `rewind_to()`
- `model-generate.py` — Multi-prompt batched generation
- `model-mm.py` — Multi-modal image + audio + text inference
- `engine/continuous-batching.py` — Engine-based concurrent request handling with threaded producer/consumer
- `whisper.py` — Speech-to-text with encoder-decoder
- `guidance-example.py` — Constrained generation with JSON schema / grammar

**Plan**:
- Add `examples/python/lora_switching.py` — LoRA adapter loading and switching
- Add architecture documentation cross-referencing examples to API concepts
- Improve inline documentation in engine examples

#### 11. Padding Optimization

**Problem**: Input padding is done separately then copied to device (TODO in `src/models/model.cpp`: "Pad directly into tensor vs copying?").

**Plan**:
- Allocate padded tensor directly on device
- Write padding values during initial tensor construction
- Eliminates one memory copy per batch element

**Files to modify**: `src/models/input_ids.cpp`, `src/models/model.cpp`

---

## Summary

ONNX Runtime GenAI achieves its goal of cross-platform, multi-backend generative AI inference through a well-layered architecture. The **State pattern** provides clean model extensibility, the **DeviceInterface** abstraction enables hardware flexibility, and the **C API boundary** ensures stable language bindings.

The primary architectural limitation is the **static batching model** — transitioning to continuous batching with the existing paged KV cache infrastructure is the highest-impact improvement. Secondary priorities focus on **async APIs**, **search performance**, and **test coverage** for the engine and non-CUDA backends.
