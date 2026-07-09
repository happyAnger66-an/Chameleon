# Qwen3-ASR：模型结构与部署要点

> 对比 **Qwen3-ASR**（Whisper 风格音频编码器 + Qwen3 LLM）与 **通用 LLM**、**SigLIP ViT**（π₀.₅ 视觉）在结构、算子与 TRT 部署上的异同。  
> 相关文档：[design/qwen3_asr_edgellm.md](../design/qwen3_asr_edgellm.md)、[cosmos3_vs_pi05.md](./cosmos3_vs_pi05.md)、[research/trt-edge-llm.md](../research/trt-edge-llm.md)。

**源码：**

| 系统 | 路径 |
|------|------|
| Qwen3-ASR | `Qwen3-ASR/qwen_asr/core/transformers_backend/`（`modeling_qwen3_asr.py`、`configuration_qwen3_asr.py`） |
| Edge-LLM ASR 导出 | `TensorRT-Edge-LLM/tensorrt_edgellm/audio_models/qwen3_asr_model.py` |
| π₀.₅ SigLIP 导出 | `Chamleon/chameleon/deploy/pi05/vit.py` |

---

## 一、一句话对照

| | Qwen3-ASR | 通用 LLM（Qwen3） | SigLIP ViT（π₀.₅） |
|---|-----------|-------------------|---------------------|
| **任务** | mel → 文本（LID + 转写） | 文本自回归生成 | 图像 → patch 语义 token |
| **主干** | Audio Encoder + Qwen3 Decoder | 纯 Decoder-only Transformer | ViT（patch + 全局 self-attn） |
| **与 LLM 关系** | LLM **就是**解码器，audio 是条件 | 本体 | **前置编码器**，输出进 LLM prefix |
| **推理形态** | encode 1 次 + decode N 步 | prefill 1 次 + decode N 步 | **单次前向**，无 decode 环 |
| **Edge 部署拆分** | **双 engine**：audio_encoder + llm | 单 llm engine（+ KV cache runtime） | 单 vit engine（与 llm 分离） |

ASR 不是传统 CTC/RNNT 声学模型，而是 **「听音频 → 生成文本」** 的多模态条件生成；与 SigLIP 同属「模态编码器 + LLM」，但前半段是 1D mel + 块对角 attn，而非 2D ViT。

---

## 二、Qwen3-ASR 模型结构

### 2.1 整体层级

```text
Qwen3ASRForConditionalGeneration
└── thinker: Qwen3ASRThinkerForConditionalGeneration
    ├── audio_tower : Qwen3ASRAudioEncoder     # mel → 连续 embedding
    ├── model       : Qwen3ASRThinkerTextModel # Qwen3 风格 Decoder-only LLM
    └── lm_head     : Linear
        ├── ASR 模型      → vocab_size（自回归生成转写）
        └── ForcedAligner → classify_num（NAR 时间戳分类，独立 checkpoint）
```

配置三层嵌套：`Qwen3ASRConfig` → `Qwen3ASRThinkerConfig` → `{audio_config, text_config}`。

### 2.2 端到端数据流

```text
16kHz PCM → WhisperFeatureExtractor (128 mel)
  → AudioEncoder: Conv×3 → 32层块对角 Transformer → proj(output_dim)
  → masked_scatter 替换 prompt 中 <|audio_pad|> 占位
  → Qwen3 LLM 因果自注意力 + MRoPE
  → lm_head → "language X<asr_text>转写文本"
```

融合方式：**不是 cross-attention**，而是 prompt 里放 `<|audio_start|> + N×<|audio_pad|> + <|audio_end|>`，用 `inputs_embeds.masked_scatter(audio_mask, audio_features)` 把 encoder 输出写进 embedding 序列（Whisper+LLM / Qwen2-Audio 范式）。

#### 2.2.1 WhisperFeatureExtractor（host 前端）

**作用**：在 **CPU/host** 把原始 PCM 波形转成 **log-mel 频谱图**，供 `Qwen3ASRAudioEncoder` 使用；**不进 TRT engine**（Edge-LLM 也要求 mel 在 Python 侧先算好再喂 runtime）。

**信号处理链路**（与 OpenAI Whisper 前端一致）：

```text
PCM (16kHz 单声道, float32, 约 [-1,1])
  → [可选] 重采样到 16kHz
  → STFT：Hann 窗, n_fft=400, hop=160（10ms/帧 → 100 帧/秒）
  → 功率谱 |STFT|²
  → 128 维 Mel 滤波器组（Slaney, 0–8kHz）
  → log10
  → 动态范围压缩：clip 到 [max-8, max]，再 (x+4)/4
  → log-mel 特征
```

**默认参数**（Qwen3-ASR / Edge-LLM `preprocess_audio.py`）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `feature_size` | 128 | mel 频带数 |
| `sampling_rate` | 16000 | 采样率 Hz |
| `hop_length` | 160 | 帧移；16000÷160 = **100 mel 帧/秒** |
| `n_fft` | 400 | FFT 窗长 25ms |

**输入**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `audio` / `raw_speech` | `np.ndarray` 1D | 浮点 PCM，通常 `[-1, 1]`；可 batch |
| `sampling_rate` | `int` | 须为 **16000**（否则先重采样） |

**输出**（`Qwen3ASRProcessor` 调用后）：

| 字段 | shape | 说明 |
|------|--------|------|
| `input_features` | `[B, 128, T]` | log-mel；T ≈ `floor(样本数 / 160) + 1` |
| `feature_attention_mask` | `[B, T]` | 有效 mel 帧 vs padding（由 `attention_mask` 重命名） |

示例：10s 音频 → 160000 样本 → mel shape `[1, 128, ~1001]`。

Processor 还会用 mel 有效长度经 `_get_feat_extract_output_lengths` 推算 **`<|audio_pad|>` 占位 token 个数**，与 encoder 输出 token 数对齐。

#### 2.2.2 音频分帧：何时算「一帧」、何时触发推理

「帧」在 Qwen3-ASR 里有 **三层含义**，开始/结束判定各不相同：

| 层级 | 谁决定边界 | 开始 | 结束 | 用途 |
|------|------------|------|------|------|
| **A. mel STFT 帧** | `WhisperFeatureExtractor` 固定参数 | 第 0 个样本对应 t=0 | 每 **160 样本**（10ms）一帧，最后一帧由音频总长决定 | 时频特征，**无 VAD** |
| **B. Encoder mel chunk** | 模型 `n_window=100` | mel 时间轴上每 **200 帧**（2s）一块；首块从 0 开始 | 整除 200 为满块；**尾块**为 `feature_lens % 200`（为 0 则当作 200） | 块对角 attn，chunk 内双向、chunk 间不互看 |
| **C. 应用推理 chunk** | 调用方 / 流式 API | 见下表三种模式 | 见下表 | **何时跑一次 ASR forward** |

**A. mel STFT 帧（最底层，纯数学）**

- 不检测语音/静音；只要给定 PCM 区间，边界由样本索引 **确定性** 推出。
- 帧 `i` 覆盖样本约 `[i×160, i×160+400)`（与 STFT 实现一致）。
- `feature_attention_mask` 标记哪些 mel 列是真实音频、哪些是 batch padding（值为 0 的 pad 帧不参与有效长度）。

**B. Encoder 内部分块（模型 forward，非 VAD）**

```text
feature_lens = 有效 mel 帧数 T
chunk_num = ceil(T / 200)
每块长度 = 200, 200, …, tail；tail = T % 200（为 0 则 tail=200）
```

- 目的：长音频 OOM 控制 + 块对角注意力；**不会在静音处切分**。
- Conv×3 后再按 `n_window_infer` 划 `cu_seqlens`，仍只在 **固定窗口** 内做 self-attn。

**C. 何时触发一次完整 ASR 推理（应用层）**

| 模式 | 开始 | 结束 | 源码 |
|------|------|------|------|
| **离线整段** | 文件/数组第 0 样本 | 最后一个样本 | `transcribe()` |
| **离线长音频切分** | 上一切分点 | 目标 `max_chunk_sec` 附近；在 ±`search_expand_sec` 内找 **滑动窗能量最小** 点作切分边界（无 overlap/gap） | `split_audio_into_chunks()` |
| **流式（vLLM）** | 流开始 `t=0`；之后每凑满 `chunk_size_sec`（默认 **2s**） | 缓冲 `buffer >= chunk_size_samples` 触发；**尾段**由 `finish_streaming_transcribe()` 刷出（可短于 2s，不 padding） | `streaming_transcribe()` |

长音频切分（C 第二种）逻辑摘要：

```text
若总长 ≤ max_chunk_sec → 整段一次推理
否则在 start + max_chunk_sec 附近 ±5s 内，用 100ms 滑窗找 |wav| 能量最低样本 → boundary
下一段从 boundary 开始；拼接所有 chunk 样本数 = 原音频（无重叠、无缺口）
过短 chunk（<0.5s）尾部 **零填充** 到 0.5s（MIN_ASR_INPUT_SECONDS）
```

流式（C 第三种）逻辑摘要：

```text
incoming PCM → state.buffer
while len(buffer) >= chunk_size_samples:   # 默认 2s = 32000 样本
    取出一块 → 追加到 audio_accum
    对 audio_accum 全量重喂模型（非增量 mel）
    用 unfixed_chunk_num / unfixed_token_num 做前缀 token 回退，减轻 chunk 边界抖动
finish: 剩余 buffer 作为 tail 再推理一次
```

**要点**：

1. **没有内置 VAD**：模型不会自动判断「这句话从哪说到哪」；整段 / 切分 / 流式 chunk 边界都由 **应用层规则** 或 **固定时长** 决定。
2. **流式边界是等间隔样本数**（默认 2s），与 mel STFT 帧、encoder 200 帧 chunk **不对齐**；每次推理仍对 **从流开始到当前的整段 audio_accum** 做 encode + decode。
3. **静音**仅在有意的切分策略里参与（长音频低能量边界）；WhisperFeatureExtractor 与 encoder 内部 **不** 因静音截断。

### 2.3 音频编码器 `Qwen3ASRAudioEncoder`

Whisper 变体，默认超参（1.7B 文档默认；0.6B checkpoint 会覆盖 text 部分）：

| 参数 | 值 | 含义 |
|------|-----|------|
| `num_mel_bins` | 128 | mel 频带 |
| `d_model` | 1280 | encoder 隐层 |
| `encoder_layers` | 32 | Transformer 层数 |
| `encoder_attention_heads` | 20 | 注意力头 |
| `output_dim` | 3584 | 投影到 LLM 空间 |
| `n_window` / `n_window_infer` | 100 / 400 | mel 分块与 attn 窗口 |

处理链路：

1. mel 按 `n_window×2` 帧分 chunk，支持长音频；
2. **3 层 Conv2d（stride=2）** 时频下采样；
3. 正弦位置编码 + **32 层 Transformer Encoder**；
4. **块对角注意力**：`cu_seqlens` 限制 attn 仅在 chunk 内，chunk 间不互看；
5. `ln_post → proj1 → GELU → proj2` → `[num_audio_tokens, output_dim]`。

下采样约 **13 audio token / 100 mel 帧**（~50 token/秒）。特点：**无 KV cache、无自回归**；官方为保精度 **encoder 不做 batch**（逐条 forward）。

### 2.4 文本解码器 `Qwen3ASRThinkerTextModel`

标准 Qwen3 Decoder-only LLM（1.7B 默认：`hidden_size=4096`，32 层，GQA，SwiGLU，`vocab_size=151936`，`max_position_embeddings=128000`）。

每层：`input_layernorm → Self-Attn(GQA + q/k_norm on head_dim + RoPE, 因果) → SwiGLU MLP`。

**MRoPE**：继承 Qwen3-Omni 的 3D 位置编码；纯 ASR 场景下 `get_rope_index` 退化为文本序列位置，但代码路径保留 MRoPE。

### 2.5 推理与输出

- **Processor**：WhisperFeatureExtractor（见 [2.2.1](#221-whisperfeatureextractorhost-前端)）+ Qwen2Tokenizer + chat template。
- **Prefill**：整段 prompt（含 audio embedding）一次过 LLM，建 KV cache。
- **Decode**：自回归；`cache_position!=0` 时不再喂 `input_features`。
- **输出格式**：`language {语种}<asr_text>{文本}`；指定 `language` 时在 prompt 追加后缀，强制只生成转写。

### 2.6 流式（应用层）

模型无专用 streaming 层；`qwen_asr` 包在 host 实现（**仅 vLLM backend**）。**分帧与触发条件见 [2.2.2](#222-音频分帧何时算一帧何时触发推理)**。

- 按 `chunk_size_sec`（默认 2s）缓冲 PCM → 凑满即触发一次推理；
- 每次对 **从流开始到当前的整段 `audio_accum`** 重喂（非增量 mel）；
- **前缀 token 回退**（`unfixed_chunk_num` / `unfixed_token_num`）减轻 chunk 边界抖动；
- `finish_streaming_transcribe()` 刷出不足一个 chunk 的尾段。

与 Edge-LLM「只接受整段 mel」兼容。

---

## 三、与 LLM、SigLIP 的结构差异

### 3.1 总览

| 维度 | Qwen3-ASR | LLM（Qwen3） | SigLIP ViT（π₀.₅） |
|------|-----------|--------------|---------------------|
| 典型瓶颈 | encoder 线性于音频长；decode 随输出 token | weight-bound GEMM + KV 带宽 | compute-bound ViT（固定 224×224） |
| 注意力类型 | encoder **双向**（块内）+ LLM **因果** | 全程因果 | patch **全局** self-attn |
| 特殊层 | Conv2d×3、块对角 mask、MRoPE、embedding scatter | GQA、q/k_norm、KV cache | patch embed、multi_modal_projector |

### 3.2 Qwen3-ASR 独有（相对 LLM / SigLIP）

| 模块 | 作用 |
|------|------|
| WhisperFeatureExtractor | 16kHz → mel，**必须在 host**，不进 TRT |
| 3× Conv2d 下采样 | SigLIP 用 patch embed；纯 LLM 无 |
| 块对角 Transformer Encoder | 长音频；非 SigLIP 全局 2D attn |
| `masked_scatter` / `embeddingLookupMultimodal` | 多模态注入，非 cross-attn |
| MRoPE | 多模态 RoPE（ASR 场景多为 1D 退化） |

### 3.3 SigLIP ViT（π₀.₅ 参考）

```text
pixel_values [B,3,224,224]
  → SigLIP vision_tower (ViT)
  → multi_modal_projector → / sqrt(hidden_size)
  → image_features [B, 256, hidden] → PaliGemma prefix → LLM KV cache
```

固定 224×224；**单次前向**；与 LLM 解耦为 `vit.engine` + `llm.engine`。和 ASR encoder 相似点：都是「模态 → token 序列给 LLM」；不同点：RGB grid vs 1D mel + Conv + 块对角 attn。

### 3.4 通用 LLM 部署相关特性

| 特性 | 说明 |
|------|------|
| GQA | decode 省 KV 带宽 |
| Q/K Head RMSNorm | norm 在 head_dim |
| `inputs_embeds` 入口 | 多模态 prefill 由 runtime 拼 embedding |
| KV cache | prefill 写 cache，decode 每步 1 token |
| 采样 | top-k / top-p / temperature（runtime，不在 ONNX） |

ASR 用的 LLM 部分即上述标准 Qwen3 causal LM + 多模态注入，**无** Cosmos3 MoT 双塔、**无** MoE。

---

## 四、部署注意点（算子、TRT、runtime）

### 4.1 Qwen3-ASR —— 特殊点最多

**Export / ONNX 改造**（Edge-LLM `qwen3_asr_model.py` 已做）：

| 原实现 | TRT 不友好原因 | 导出改法 |
|--------|----------------|----------|
| mel chunk/pad 在 model 内 | 动态 shape + 控制流 | **挪到 runtime**（GPU audioUtils 或 host Python） |
| `padded_mask_after_cnn` 布尔索引 | ONNX **NonZero** | 改为 **`indices [N,2]`** 高级索引 |
| `cu_seqlens` + FlashAttention | FA2 varlen 难导出 | **eager attn + block-diagonal attention_mask** |
| `masked_scatter` 融合 | 不在单一 ONNX 图 | LLM runtime：**`embeddingLookupMultimodal` CUDA kernel** |

**Runtime 必注意：**

1. **mel 在 host**：WhisperFeatureExtractor → FP16 safetensors；TRT 只吃 mel。
2. **双 engine**：`engines/llm/` + `engines/audio/`；`PyLLMRuntime(engine_dir, multimodal_engine_dir, ...)`。
3. **自定义 plugin / kernel**：`embeddingLookupMultimodal`；Edge-LLM `NvInfer_edgellm_plugin`（`loadEdgellmPluginLib()`）；LLM engine 可能含 **ScatterElements** 等，需 `trt.init_libnvinfer_plugins()`。
4. **MRoPE**：`initializeSequentialMRopeCache`，非简单 1D position_ids。
5. **LLM 用 `inputs_embeds`**：prefill 由 runtime 组装。
6. **KV / 动态长度**：build 设 `maxInputLen`、`maxKVCacheCapacity`；长音频 → 长 prefill。
7. **音频流式**：Edge-LLM **不支持**增量 audio；应用层整段重喂 + 前缀回退。
8. **Encoder 量化**：需 LibriSpeech 等单独校准；encoder batch 关闭。

官方 pipeline（Edge-LLM）：

```bash
tensorrt-edgellm-export-audio --model_dir Qwen/Qwen3-ASR-0.6B --output_dir onnx/audio
tensorrt-edgellm-export-llm     --model_dir Qwen/Qwen3-ASR-0.6B --output_dir onnx/llm
./audio_build --onnxDir onnx/audio --engineDir engines/audio --minTimeSteps 1000 --maxTimeSteps 3000
./llm_build   --onnxDir onnx/llm   --engineDir engines/llm   --maxBatchSize 1 --maxInputLen 1024 --maxKVCacheCapacity 4096
```

### 4.2 通用 LLM 单独部署

| 注意点 | 说明 |
|--------|------|
| KV cache | runtime 分配 per-layer past KV，decode 步进更新 |
| Prefill vs Decode | profile 上界在 build 时定死；decode GEMM 小、带宽敏感 |
| Edge-LLM plugin | FMHA、RoPE、embedding 等多为 C++ plugin |
| CUDA Graph | decode 环可 capture（`captureDecodingCudaGraph`） |
| System prompt KV cache | 固定 prompt 可 `saveSystemPromptKVCache` |
| 文本流式 | Edge-LLM `StreamChannel`（token 级）；≠ 音频流式 |
| 量化 | FP8/NVFP4 + KV FP8 需校准 |

### 4.3 SigLIP ViT 部署（π₀.₅）

| 注意点 | 说明 |
|--------|------|
| 标准 ViT ONNX | Conv/GEMM/LayerNorm/Softmax 为主 |
| Attention 导出 | `force_vision_eager_attention` + `sdp_math_backend_only`（`deploy/pi05/vit.py`） |
| 固定 224×224 | 多相机靠 batch 或多次调用 |
| 与 LLM 衔接 | ViT 输出作 prefix embedding / prefill 片段，非单一 end-to-end graph |
| 预处理 | 仅 resize/normalize，比 ASR mel 简单 |

---

## 五、算子 / 机制对照表

| 算子/机制 | ASR | LLM | SigLIP |
|-----------|-----|-----|--------|
| Conv2d 下采样 | ✅ 3 层 | ❌ | ✅ patch embed |
| 块对角 / varlen attn | ✅ audio encoder | ❌ | ❌ |
| Causal self-attn | ✅ LLM 部分 | ✅ | ❌ |
| RoPE / MRoPE | ✅ LLM | ✅ | ❌（可学习 pos embed） |
| `embeddingLookupMultimodal` | ✅ | ✅（VLM） | ❌ |
| KV cache | ❌ encoder | ✅ | ❌（结果进 LLM cache） |
| 自回归 decode 环 | ✅ | ✅ | ❌ |
| ScatterElements 等 TRT plugin | 可能（LLM engine） | 可能 | 较少 |
| Host 重预处理 | ✅ mel | ❌ | 轻量 resize/norm |
| FlashAttention / cu_seqlens | 训练用；**导出禁用** | 训练用；Edge-LLM 用 plugin | 导出禁用 |

---

## 六、Chameleon 落地建议

| 模型 | 推荐策略 | 主要风险 |
|------|----------|----------|
| **ASR** | 封装 Edge-LLM 双 engine（见 [qwen3_asr_edgellm.md](../design/qwen3_asr_edgellm.md)）；mel/prompt 在 host | mel IO、plugin 注册、MRoPE、长音频 prefill OOM |
| **LLM** | Edge-LLM `llm_build` + pybind | profile 上界、KV 容量、plugin 版本 |
| **SigLIP** | 单 stage `vit.onnx` → TRT（pi05 已验证） | SDPA 导出、projector 与 LLM hidden 对齐 |

**一句话：**

- **SigLIP** ≈ 标准 ViT，部署最简单。
- **LLM** ≈ Qwen3 + KV cache / plugin / 采样 runtime，难度中等。
- **ASR** = 特殊 audio encoder（mel + Conv + 块对角 attn）+ LLM decode + 多模态 embedding 注入 + host mel；**不宜**当作单一 TRT graph 或普通 ViT 处理。

---

## 七、参数量分布（1.7B 量级）

| 模块 | 规模感 | 说明 |
|------|--------|------|
| Audio Encoder | ~0.3–0.4B | 32×1280 维，相对 LLM 较小 |
| Text LLM | ~1.3–1.4B | 32×4096 + 15 万词表 lm_head |
| **合计** | **~1.7B** | 0.6B 版主要缩小 text backbone |

Audio `output_dim` 与 LLM `hidden_size` 在 checkpoint 的 `config.json` 中应对齐（以实际权重为准）。
