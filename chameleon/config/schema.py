"""统一任务配置 — pydantic + YAML 驱动的 quantize / compile / infer 描述。

作用：
    定义 TaskConfig 及子模型（QuantizeStep、CompileStep、InferConfig），
    描述 architecture / platform / actions / stage_runtimes / model_overrides
    等。TaskConfig.load() 从 YAML 加载并校验。

架构位置：
    入口/编排层 — 全框架配置的单一来源，被 cli.py、api.py、
    workflows/runner.py、profile/latency.py 消费。configs/*.yaml 为本
    schema 的实例。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class QuantizeStep(BaseModel):
    stage: str
    method: str = "fp8"
    weight_dtype: str = "int8"
    activation_dtype: str | None = None
    kv_cache_dtype: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class ExportStep(BaseModel):
    """单阶段 ONNX 导出（真实 pi05 / openpi 路径）。"""

    stage: str
    """子图名：``vit`` | ``llm`` | ``expert`` | ``denoise`` | ``embed_prefix``。"""

    options: dict[str, Any] = Field(default_factory=dict)
    """透传给 Chameleon pi05 stage exporter 的额外参数。"""


class DeployConfig(BaseModel):
    """TRT 部署流水线 — 真实 pi05 / cosmos3 的 ONNX 导出与 engine 构建。

    ``backend=pi05`` 时使用 ``chameleon.deploy`` 内置 pi05 exporter 与 TRT build；
    ``backend=cosmos3`` 走 cosmos3 的 vae_encode / dit / vae_decode 分阶段导出；
    ``reference`` 仍走 Chameleon reference 图捕获 + 简化编译器。
    """

    backend: str = "reference"
    """``reference`` | ``pi05``（``pi05_openpi`` 别名）| ``cosmos3``。"""

    export_dir: str | None = None
    """ONNX 输出目录；缺省 ``{output_dir}/onnx``。"""

    engine_dir: str | None = None
    """TRT engine 输出目录；缺省 ``{output_dir}/engines``。"""

    checkpoint_dir: str | None = None
    """含 ``model.safetensors`` 的 openpi PyTorch checkpoint 目录。"""

    train_config: str | None = None
    """OpenPI TrainConfig 注册名或 ``.py`` 路径；缺省取 ``data.openpi_config``。"""

    build_cfg_dir: str | None = None
    """build_cfg ``.py`` 文件目录；缺省 ``configs/build_configs``。"""

    build_cfgs: dict[str, str] = Field(default_factory=dict)
    """按 stage 覆盖默认 build_cfg 文件名（相对 ``build_cfg_dir``）。"""

    use_cudagraph: bool = False
    """TRT build 是否启用 ``BuilderFlag.CUDA_GRAPH``。"""


class CompileStep(BaseModel):
    stage: str
    options: dict[str, Any] = Field(default_factory=dict)


class TrtProfileStep(BaseModel):
    """单 stage TRT engine layer profiling（trtexec --loadEngine）。"""

    stage: str
    options: dict[str, Any] = Field(default_factory=dict)
    """Per-stage overrides: iterations, skip, trtexec_extra_args, fail_fast."""


class TrtProfileConfig(BaseModel):
    """trtexec layer profile 全局配置（``actions`` 含 ``trt_profile`` 时使用）。"""

    profile_dir: str | None = None
    """Profile JSON 输出目录；缺省 ``{output_dir}/profiles``。"""

    iterations: int = 20
    warmup: int = 200
    separate_profile_run: bool = True
    profiling_verbosity: str = "detailed"
    export_layer_info: bool = False
    export_times: bool = False
    viewer: str = "static"
    """``static`` | ``webui`` | ``both`` — 静态 HTML / 阻塞 HTTP 服务 / 两者。"""
    webui_host: str = "127.0.0.1"
    webui_port: int = 8770
    open_browser: bool = True
    plugin_lib_paths: list[str] = Field(default_factory=list)
    trtexec_path: str | None = None
    """``trtexec`` 可执行文件；须与 compile 时 TensorRT 版本一致（缺省 PATH 中 ``trtexec``）。"""
    fail_fast: bool = False
    timeout_sec: int = 1800


class InferConfig(BaseModel):
    batch_size: int = 1
    num_steps: int | None = None
    torch_device: str | None = None
    use_compiled_engines: bool = False
    """When true, infer consumes the engines produced by the compile step
    (per stage) via the platform runtime, instead of the PyTorch reference path."""
    cuda_graph: bool = False
    """Capture/replay a CUDA graph per engine (TensorRT runtime; static shapes)."""


class ActionGenConfig(BaseModel):
    """Cosmos3 动作条件生成参数（对齐 diffusers ``CosmosActionCondition``）。"""

    mode: str = "policy"
    """``policy`` | ``forward_dynamics`` | ``inverse_dynamics``。"""

    chunk_size: int = 16
    """动作 chunk 长度；视频帧数取 ``chunk_size + 1``。"""

    domain_name: str = "bridge_orig_lerobot"
    """具身域名（选择 DomainAwareLinear 权重）。"""

    resolution_tier: int = 480
    """条件画布分辨率档位：256 | 480 | 704 | 720。"""

    view_point: str = "ego_view"
    """相机视角：``ego_view`` | ``third_person_view`` | ``wrist_view`` | ``concat_view``。"""

    video: str | None = None
    """条件视频路径（policy/inverse_dynamics 需要）。"""

    image: str | None = None
    """条件图像路径（policy/forward_dynamics 可用）。"""


class GenerateConfig(BaseModel):
    """Cosmos3 生成配置 — 由 cosmos3 orchestrator 消费驱动 ``Cosmos3OmniPipeline``。

    ``mode=video`` 时走 text/image/video-to-video 扩散，输出视频（+可选 sound）；
    ``mode=action`` 时走 action 策略生成，输出 action chunk（对齐 pi05 VLA）。
    reference 路径只消费 ``mode`` / ``num_inference_steps`` / ``guidance_scale``。
    """

    mode: str = "video"
    """``video`` | ``action``。"""

    prompt: str | None = None
    """文本 prompt；video 模式建议传入 JSON-upsampled prompt 字符串。"""

    negative_prompt: str | None = None
    """CFG 负向 prompt；缺省走 pipeline 内置默认。"""

    image: str | None = None
    """image-to-video 的条件图路径。"""

    video: str | None = None
    """video-to-video 的条件视频路径。"""

    num_frames: int | None = None
    """生成帧数；缺省取架构 metadata（189）。``num_frames=1`` 为文生图。"""

    height: int | None = None
    width: int | None = None
    fps: float = 24.0
    num_inference_steps: int = 35
    guidance_scale: float = 6.0
    enable_sound: bool = False
    """是否联合生成音频（需 checkpoint 含 sound_tokenizer）。"""

    output_type: str = "pt"
    """``pt`` | ``np`` | ``pil`` | ``latent``；reference 与 infer smoke 用 ``pt``。"""

    flow_shift: float = 10.0
    """UniPCMultistepScheduler 的 flow_shift（video 模式推荐 10.0）。"""

    action: ActionGenConfig = Field(default_factory=ActionGenConfig)
    """action 模式的细化参数。"""


class DataConfig(BaseModel):
    """真实数据集配置 — 驱动 dataloader 加载 LeRobot 数据。

    ``dataset`` 为 ``chameleon.dataloader`` 注册表中的数据集名（如
    ``pi05_libero``）；其余字段覆盖该数据集的默认值。仅 evaluate / 真实数据
    推理路径消费；compile / synthetic-smoke 路径不依赖本段。
    """

    dataset: str | None = None
    """Registered dataset name (see ``chameleon.dataloader.list_datasets()``)."""

    repo_id: str | None = None
    """Override the LeRobot ``repo_id`` (else taken from the openpi DataConfig)."""

    dataset_root: str | None = None
    """Local LeRobot dataset root; falls back to the HuggingFace cache when unset."""

    openpi_config: str | None = None
    """Override the openpi TrainConfig name used to resolve repack / norm stats."""

    action_horizon: int | None = None
    """Override the sampled action-chunk length (else from the openpi model config)."""

    start_index: int = 0
    """First frame index to expose (useful for quick smoke checks)."""

    num_samples: int | None = None
    """If set, limit the dataset length to ``num_samples`` frames from ``start_index``."""


class TrtEngineNames(BaseModel):
    """pi05 TRT engine 文件名（相对 ``engine_dir``）。"""

    vit: str = "vit.engine"
    llm: str = "llm.engine"
    expert: str = "expert.engine"
    denoise: str = "denoise.engine"
    embed_prefix: str = ""


class EvaluateConfig(BaseModel):
    """评测配置 — 真实数据集上对比预测动作与 ground-truth。

    复用 openpi ``create_trained_policy``（含 norm / tokenize / 输入输出
    transform），逐帧推理后用 ``evaluate.compare_actions`` 计算误差并汇总。

    ``compare_mode=true`` + ``policy_runner=pt_trt_compare`` 时启用 PyTorch 浮点
    vs TensorRT engine 双路对比（WebUI 展示 ``pred_action`` / ``pred_action_trt``）。
    """

    checkpoint_dir: str | None = None
    """含 ``model.safetensors`` 的目录；缺省时取 model_overrides.checkpoint 的父目录。"""

    norm_stats_dir: str | None = None
    """norm_stats 的 assets 目录（含 ``{asset_id}/norm_stats.json``）；
    缺省时回退到 ``checkpoint_dir/assets``。"""

    asset_id: str | None = None
    """norm_stats 的 asset_id（如 ``physical-intelligence/libero``）；
    缺省时取 openpi DataConfig.asset_id。"""

    num_samples: int = 50
    """评测的帧数（从 data.start_index 起）。"""

    stride: int = 1
    """帧采样步长（>1 时跳帧，加快评测）。"""

    compare_horizon: int | None = None
    """仅对比前 N 个动作步；None = 全 action_horizon。"""

    default_prompt: str | None = None
    """当样本缺少 prompt 时注入的默认指令。"""

    device: str | None = None
    """TRT / PyTorch 推理设备；缺省时取 ``infer.torch_device`` 或自动选择。"""

    pytorch_load_device: str | None = "cpu"
    """openpi Policy 构建时 PyTorch 权重加载设备（``trt_only`` / ``pt_trt_compare`` TRT 路、
    ``chameleon`` + ``use_compiled_engines`` 时生效）。默认 ``cpu`` 以在挂载 TRT engine 前
    避免 GPU OOM；推理仍使用 ``device``。"""

    policy_runner: str = "openpi"
    """策略运行器：``openpi`` | ``chameleon`` | ``pt_trt_compare`` | ``trt_only``。"""

    compare_mode: bool = False
    """PyTorch 浮点 vs TensorRT engine 双路对比（须 ``policy_runner=pt_trt_compare``）。"""

    engine_dir: str | None = None
    """TRT engine 目录；缺省 ``deploy.engine_dir`` 或 ``{output_dir}/engines``。"""

    trt_engines: TrtEngineNames | None = None
    """各 stage engine 文件名；缺省 vit/llm/expert/denoise.engine。"""

    noise: str = "random"
    """flow-matching 初值噪声：``random`` | ``fixed``（双路对比建议 ``fixed``）。"""

    noise_seed: int = 0
    """``noise=fixed`` 时的基础 seed（与 sample index 组合）。"""

    precision: str | None = None
    """TRT 挂载精度；缺省取 ``model_overrides.precision`` 或 ``bf16``。"""

    trt_cuda_graph: bool = False
    """TRT 路 CUDA Graph；compare_mode 下会自动关闭。"""

    viewer: str = "console"
    """评测结果展示：``console`` | ``webui`` | ``both``。"""

    webui_host: str = "127.0.0.1"
    """WebUI WebSocket 监听地址。"""

    webui_port: int = 8765
    """WebUI WebSocket 端口。"""

    webui_path: str = "/ws"
    """WebSocket 握手路径。"""

    webui_history_size: int = 200
    """新连接回放的历史 step 条数上限。"""

    webui_queue_maxsize: int = 256
    """推理线程 → asyncio 泵的有界队列；满时丢弃 UI 事件，不阻塞推理。"""

    webui_show_wrist: bool = True
    """是否在 step 首帧推送 wrist 相机 JPEG。"""

    webui_jpeg_quality: int = 85
    """观测图像 JPEG 质量（1–100）。"""


class TaskConfig(BaseModel):
    architecture: str = "pi05"
    model: str = "pi05"
    platform: str = "generic_cpu"
    output_dir: str = "output/chameleon_run"

    actions: list[str] = Field(default_factory=lambda: ["infer"])
    """Ordered subset of ``quantize | export | compile | trt_profile | infer``."""

    model_overrides: dict[str, Any] = Field(default_factory=dict)
    """Overrides applied to the model adapter config (e.g. action_dim)."""

    stage_runtimes: dict[str, str] = Field(default_factory=dict)
    """Per-stage runtime backend, enabling stage-level backend mixing."""

    quantize: list[QuantizeStep] = Field(default_factory=list)
    export: list[ExportStep] = Field(default_factory=list)
    """ONNX 导出步骤；``actions`` 含 ``export`` 时执行。"""
    deploy: DeployConfig = Field(default_factory=DeployConfig)
    """TRT 部署（真实 pi05 ONNX 导出 + engine build）。"""
    compile: list[CompileStep] = Field(default_factory=list)
    trt_profile: list[TrtProfileStep] = Field(default_factory=list)
    """TRT engine layer profile 步骤；``actions`` 含 ``trt_profile`` 时执行。"""
    profile: TrtProfileConfig = Field(default_factory=TrtProfileConfig)
    """trtexec --dumpProfile / WebUI 配置。"""
    infer: InferConfig = Field(default_factory=InferConfig)
    generate: GenerateConfig = Field(default_factory=GenerateConfig)
    """Cosmos3 生成配置（mode=video/action）；仅 cosmos3 模型消费。"""
    data: DataConfig = Field(default_factory=DataConfig)
    """Real-dataset config consumed by the dataloader / evaluate paths."""
    evaluate: EvaluateConfig = Field(default_factory=EvaluateConfig)
    """Evaluation config consumed by the ``eval`` action / CLI subcommand."""

    @classmethod
    def load(cls, path: str | Path) -> "TaskConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)
