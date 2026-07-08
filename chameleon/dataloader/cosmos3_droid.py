"""Cosmos3 DROID 数据源 — 读取 DROID RLDS（如 droid_100），产出 cosmos3 样本。

作用：
    Cosmos3-Nano-Policy-DROID 的 action 表示（10D = 3D 平移 + 6D 旋转[Zhou 2019]
    + 1D 夹爪）与 openpi/pi05 的 DROID repack（7D 关节 + 1D 夹爪）不同，因此不能
    复用 :mod:`chameleon.dataloader.lerobot` 的 openpi repack 通路。本模块提供：

      1. **纯桥接函数**（无 TF/torch 依赖，可单测）：
         - :func:`euler_to_rotation_6d`：欧拉角 → Zhou 6D 旋转表示；
         - :func:`cartesian_gripper_to_cosmos10`：DROID 笛卡尔位姿+夹爪 → 10D；
         - :func:`build_cosmos3_gt_chunk`：整段动作 → ``[horizon, raw_action_dim]``；
         - :func:`build_cosmos3_observation`：单帧 RGB → runner 可消费的 observation。
      2. **:class:`DroidRldsDataSource`**：懒加载 tensorflow_datasets 读取 DROID RLDS，
         沿 episode 切 action chunk，产出 :class:`ChameleonSample`，接口与
         ``LeRobotDataSource`` 对齐（build / __len__ / __getitem__ / action_* 属性）。

**GT 动作空间说明（重要）：**
    这里的 10D GT 依据 Cosmos3 论文 Fig.3 的 *结构*（平移3 + 6D旋转 + 夹爪1）由 DROID
    原生笛卡尔动作构造，用于 WebUI 的逐维 pred-vs-gt 可视化对照。它 **不包含** Cosmos
    训练用的数据集归一化统计（NVIDIA 未随权重公开），故绝对数值尺度可能与模型预测不
    完全一致；曲线的 **形状 / 相对趋势** 有参考价值，绝对误差需谨慎解读。

架构位置：
    数据层 — 由 ``build_dataset`` 依 ``DatasetSpec(loader="droid_rlds")`` 构建。
    上游：evaluate / WebUI；下游：``cosmos3_trt_only`` runner。tensorflow / tfds 延迟
    导入，无依赖环境下本模块仍可 import（仅 build() 打开数据集时才需要）。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from chameleon.dataloader.base import (
    ChameleonSample,
    DatasetSpec,
    register_loader,
)

logger = logging.getLogger(__name__)

# Cosmos3 droid_lerobot 域的固定动作规格（见 diffusers pipeline
# _EMBODIMENT_TO_RAW_ACTION_DIM["droid_lerobot"] == 10；chunk_size 见 shapes.POLICY_DROID）。
DROID_RAW_ACTION_DIM = 10
DROID_ACTION_HORIZON = 16


# ---------------------------------------------------------------------------
# 纯桥接函数（无 TF/torch 依赖，可单测）
# ---------------------------------------------------------------------------
def euler_to_rotation_6d(euler_rpy: Any) -> np.ndarray:
    """欧拉角(roll, pitch, yaw) → Zhou et al.(2019) 6D 旋转表示。

    6D 表示取旋转矩阵前两列展平（列优先），是 Cosmos3 统一动作表示中 9D 位姿的
    旋转分量。采用外旋 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` 约定（DROID 笛卡尔
    动作常用 xyz 欧拉）。

    Args:
        euler_rpy: 形如 ``[roll, pitch, yaw]``（弧度）。

    Returns:
        ``np.ndarray`` 形状 ``[6]``：``[R[:,0], R[:,1]]``。
    """
    r, p, y = (float(v) for v in np.asarray(euler_rpy, dtype=np.float64).reshape(-1)[:3])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    rot = rz @ ry @ rx
    return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)


def cartesian_gripper_to_cosmos10(cartesian_6d: Any, gripper: Any) -> np.ndarray:
    """DROID 笛卡尔位姿(6D: xyz+rpy) + 夹爪(1D) → Cosmos3 10D 动作向量。

    10D = ``[x, y, z, rot6d(6), gripper]``（3 平移 + 6D 旋转 + 1 夹爪）。
    """
    cart = np.asarray(cartesian_6d, dtype=np.float32).reshape(-1)
    if cart.shape[0] < 6:
        cart = np.concatenate([cart, np.zeros(6 - cart.shape[0], dtype=np.float32)])
    xyz = cart[:3]
    rot6d = euler_to_rotation_6d(cart[3:6])
    grip = np.asarray(gripper, dtype=np.float32).reshape(-1)[:1]
    if grip.shape[0] == 0:
        grip = np.zeros(1, dtype=np.float32)
    return np.concatenate([xyz, rot6d, grip]).astype(np.float32)


def build_cosmos3_gt_chunk(
    cartesian_seq: Any,
    gripper_seq: Any,
    start: int,
    horizon: int = DROID_ACTION_HORIZON,
    raw_action_dim: int = DROID_RAW_ACTION_DIM,
) -> np.ndarray:
    """从整段笛卡尔/夹爪动作序列切出 ``[horizon, raw_action_dim]`` 的 GT chunk。

    末尾不足 ``horizon`` 时重复最后一帧（与 openpi DROID chunk 语义一致：绝对位姿
    动作，重复末帧即“保持”）。
    """
    cart = np.asarray(cartesian_seq, dtype=np.float32)
    grip = np.asarray(gripper_seq, dtype=np.float32)
    n = cart.shape[0]
    if n == 0:
        return np.zeros((horizon, raw_action_dim), dtype=np.float32)
    out = np.zeros((horizon, raw_action_dim), dtype=np.float32)
    for i in range(horizon):
        j = min(start + i, n - 1)
        vec = cartesian_gripper_to_cosmos10(cart[j], grip[j])
        out[i, : min(raw_action_dim, vec.shape[0])] = vec[:raw_action_dim]
    return out


def build_cosmos3_observation(
    exterior_rgb: Any,
    wrist_rgb: Any | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """把 DROID 单帧图像整理成 runner / WebUI 都能消费的 observation。

    - ``image``：外部相机帧 ``[H, W, 3] uint8``（runner 侧
      :func:`build_conditioning_canvas` 会 resize + 归一化到 [-1,1]；WebUI 侧
      :func:`encode_observation_images` 直接 JPEG 编码）。
    - ``wrist_image``：腕部相机帧（仅 WebUI 预览，可选）。
    - ``prompt``：语言指令（policy 下 caption 静态，仅作展示）。
    """
    obs: dict[str, Any] = {"image": np.asarray(exterior_rgb)}
    if wrist_rgb is not None:
        obs["wrist_image"] = np.asarray(wrist_rgb)
    if prompt is not None:
        obs["prompt"] = prompt
    return obs


def _decode_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").strip() or None
        except Exception:  # noqa: BLE001
            return None
    try:
        return str(value).strip() or None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# DROID RLDS 数据源
# ---------------------------------------------------------------------------
class DroidRldsDataSource:
    """按 DatasetSpec 读取 DROID RLDS（tfds），产出 cosmos3 ChameleonSample。

    spec 约定：
        - ``dataset_root``：tfds ``data_dir``（含 ``<builder>/<version>/`` 的父目录，
          如用 ``gsutil cp gs://gresearch/robotics/droid_100 ~/robot/datasets/`` 后为
          ``~/robot/datasets``）。
        - ``action_horizon``：动作 chunk 长度（缺省 16）。
        - ``extra``：``builder``（默认 ``droid_100``）、``version``（默认最新）、
          ``camera``（默认 ``exterior_image_1_left``）、``max_frames_per_episode``。

    懒加载：构造仅记录 spec，tfds/tensorflow 导入与数据读取发生在 build()。
    评测为顺序访问，build() 内按 ``start_index``+``num_samples`` 窗口一次性物化为
    ChameleonSample 列表。
    """

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self._built = False
        self._samples: list[ChameleonSample] = []
        self._action_horizon = int(spec.action_horizon or DROID_ACTION_HORIZON)
        self._action_dim = int(spec.extra.get("raw_action_dim", DROID_RAW_ACTION_DIM))
        self._repo_id = spec.repo_id or str(spec.extra.get("builder", "droid_100"))
        self._start = max(0, int(spec.start_index))
        self._episode_ids: np.ndarray | None = None

    # ------------------------------------------------------------------
    def build(self) -> "DroidRldsDataSource":
        if self._built:
            return self

        if not self.spec.dataset_root:
            raise ValueError(
                f"数据集 {self.spec.name!r} 需要 dataset_root（tfds data_dir，指向含 "
                f"{self._repo_id}/<version>/ 的父目录）。请在 TaskConfig.data.dataset_root "
                "或 build_dataset overrides 中设置。"
            )
        import os

        data_dir = os.path.expanduser(str(self.spec.dataset_root))

        try:
            import tensorflow as tf
            import tensorflow_datasets as tfds
        except ImportError as exc:  # pragma: no cover - 取决于运行环境
            raise ImportError(
                "读取 DROID RLDS 需要 tensorflow + tensorflow_datasets。"
                "请安装：pip install tensorflow tensorflow_datasets"
            ) from exc

        # 避免 TF 抢占 GPU（与 torch/TRT 推理共存）。
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:  # noqa: BLE001  pragma: no cover
            pass

        builder_name = str(self.spec.extra.get("builder", "droid_100"))
        version = self.spec.extra.get("version")
        camera = str(self.spec.extra.get("camera", "exterior_image_1_left"))
        wrist_key = str(self.spec.extra.get("wrist_camera", "wrist_image_left"))
        max_per_ep = int(self.spec.extra.get("max_frames_per_episode", 0) or 0)

        builder_kwargs: dict[str, Any] = {"data_dir": data_dir}
        if version:
            builder_kwargs["version"] = str(version)
        builder = tfds.builder(builder_name, **builder_kwargs)
        dataset = builder.as_dataset(split=self.spec.extra.get("split", "train"))

        want = None
        if self.spec.num_samples is not None:
            want = self._start + int(self.spec.num_samples)

        samples: list[ChameleonSample] = []
        episode_ids: list[int] = []
        global_index = 0
        for ep_id, episode in enumerate(dataset):
            steps = list(episode["steps"])
            if not steps:
                continue
            cart_seq, grip_seq, ext_imgs, wrist_imgs, prompt = self._episode_arrays(
                steps, camera=camera, wrist_key=wrist_key
            )
            n = len(steps)
            n_frames = min(n, max_per_ep) if max_per_ep > 0 else n
            for f in range(n_frames):
                if want is not None and global_index >= want:
                    break
                if global_index >= self._start:
                    gt = build_cosmos3_gt_chunk(
                        cart_seq, grip_seq, f, self._action_horizon, self._action_dim
                    )
                    obs = build_cosmos3_observation(
                        ext_imgs[f],
                        wrist_imgs[f] if wrist_imgs is not None else None,
                        prompt,
                    )
                    samples.append(
                        ChameleonSample(
                            observation=obs,
                            actions_gt=gt,
                            prompt=prompt,
                            index=global_index,
                            episode_id=ep_id,
                        )
                    )
                    episode_ids.append(ep_id)
                global_index += 1
            if want is not None and global_index >= want:
                break

        self._samples = samples
        self._episode_ids = np.asarray(episode_ids, dtype=np.int64)
        self._n_total = global_index
        self._built = True
        logger.info(
            "DROID RLDS ready: builder=%s data_dir=%s frames_scanned=%d samples=%d "
            "action_horizon=%d action_dim=%d",
            builder_name,
            data_dir,
            global_index,
            len(samples),
            self._action_horizon,
            self._action_dim,
        )
        return self

    @staticmethod
    def _episode_arrays(steps: list, *, camera: str, wrist_key: str):
        """把一个 episode 的 step 列表抽成 numpy 序列（笛卡尔/夹爪/图像/prompt）。"""

        def _np(x):
            return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

        def _get(d: dict, *keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        cart_list, grip_list, ext_list, wrist_list = [], [], [], []
        prompt: str | None = None
        has_wrist = True
        for st in steps:
            obs = st["observation"]
            act = st.get("action_dict", {}) if hasattr(st, "get") else st["action_dict"]
            cart = _get(act, "cartesian_position") if act is not None else None
            if cart is None:
                cart = _get(obs, "cartesian_position")
            grip = _get(act, "gripper_position") if act is not None else None
            if grip is None:
                grip = _get(obs, "gripper_position")
            cart_list.append(_np(cart) if cart is not None else np.zeros(6, np.float32))
            grip_list.append(_np(grip) if grip is not None else np.zeros(1, np.float32))
            ext = _get(obs, camera, "exterior_image_1_left", "exterior_image_2_left")
            ext_list.append(_np(ext))
            wrist = _get(obs, wrist_key, "wrist_image_left")
            if wrist is None:
                has_wrist = False
            else:
                wrist_list.append(_np(wrist))
            if prompt is None:
                prompt = _decode_prompt(
                    _np(_get(st, "language_instruction"))
                    if _get(st, "language_instruction") is not None
                    else None
                )

        cart_seq = np.asarray(cart_list, dtype=np.float32)
        grip_seq = np.asarray(grip_list, dtype=np.float32)
        wrist_imgs = wrist_list if has_wrist and len(wrist_list) == len(ext_list) else None
        return cart_seq, grip_seq, ext_list, wrist_imgs, prompt

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if not self._built:
            self.build()
        return len(self._samples)

    def __getitem__(self, index: int) -> ChameleonSample:
        if not self._built:
            self.build()
        if index < 0:
            index += len(self._samples)
        if not (0 <= index < len(self._samples)):
            raise IndexError(f"index {index} out of range [0, {len(self._samples)})")
        return self._samples[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    # --- metadata --------------------------------------------------------
    @property
    def repo_id(self) -> str | None:
        return self._repo_id

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def start_index(self) -> int:
        return self._start

    @property
    def frame_count(self) -> int:
        if not self._built:
            self.build()
        return self._n_total

    @property
    def eval_end_exclusive(self) -> int:
        if not self._built:
            self.build()
        return self._start + len(self._samples)

    @property
    def episode_ids_per_frame(self) -> np.ndarray:
        if not self._built:
            self.build()
        return self._episode_ids if self._episode_ids is not None else np.zeros(0, np.int64)


register_loader("droid_rlds", DroidRldsDataSource, override=True)
