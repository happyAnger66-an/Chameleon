"""cosmos3 真实权重 TRT 部署 E2E — profile / build_cfg / plan 校验 + 可选 export smoke。

不需 GPU/权重的部分（plan dry-run、profile 派生量、build_cfg 一致性）默认运行；
真实 export（需 CUDA + diffusers 权重）标记 e2e_slow，缺环境时自动 skip。
"""

from __future__ import annotations

import importlib.util

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.deploy.build_cfg import load_build_cfg
from chameleon.deploy.cosmos3.shapes import NANO_ACTION, POLICY_DROID, get_profile
from chameleon.workflows.runner import WorkflowRunner


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _diffusers_cosmos3_available() -> bool:
    return importlib.util.find_spec("diffusers") is not None


@pytest.mark.e2e
class TestCosmos3TrtProfiles:
    def test_profile_derived_shapes(self) -> None:
        p = POLICY_DROID
        assert p.latent_t == (p.num_frames - 1) // p.scale_factor_temporal + 1
        assert p.latent_h == p.canvas_h // p.scale_factor_spatial
        assert p.latent_w == p.canvas_w // p.scale_factor_spatial
        assert p.num_vision_tokens == p.latent_t * p.patch_h * p.patch_w
        assert p.sequence_length == p.text_prefix_len + p.num_vision_tokens + p.chunk_size

    def test_get_profile(self) -> None:
        assert get_profile("policy_droid") is POLICY_DROID
        assert get_profile("nano_action") is NANO_ACTION
        with pytest.raises(KeyError):
            get_profile("does_not_exist")

    @pytest.mark.parametrize("prefix", ["cosmos3_policy_droid", "cosmos3_nano_action"])
    def test_build_cfgs_match_profile(self, configs_dir, prefix) -> None:
        profile = POLICY_DROID if "policy_droid" in prefix else NANO_ACTION
        bc = configs_dir / "build_configs"
        dit = load_build_cfg(bc / f"{prefix}_dit_step_build_cfg.py")
        num_noisy_vision = (profile.latent_t - 1) * profile.patch_h * profile.patch_w
        expected = {
            "vision_tokens": (
                1,
                profile.latent_channels,
                profile.latent_t,
                profile.latent_h,
                profile.latent_w,
            ),
            "vision_timesteps": (num_noisy_vision,),
            "action_tokens": (profile.chunk_size, profile.action_dim),
            "action_timesteps": (profile.chunk_size,),
        }
        assert dit["opt_shapes"] == expected
        assert dit["precision"] == "bf16"

        enc = load_build_cfg(bc / f"{prefix}_vae_encode_build_cfg.py")
        assert enc["opt_shapes"]["video"] == (1, 3, profile.num_frames, profile.canvas_h, profile.canvas_w)


@pytest.mark.e2e
class TestCosmos3TrtDeployPlan:
    @pytest.mark.parametrize(
        "cfg", ["cosmos3_policy_droid_trt_deploy.yaml", "cosmos3_nano_action_trt_deploy.yaml"]
    )
    def test_deploy_plan_lists_all_stages(self, configs_dir, cfg) -> None:
        task = TaskConfig.load(configs_dir / cfg)
        assert task.model_overrides.get("use_reference") is False
        joined = "\n".join(WorkflowRunner(task).plan())
        for stage in ("vae_encode", "text_embed", "dit", "vae_decode"):
            assert stage in joined

    @pytest.mark.parametrize(
        "cfg,runner",
        [
            ("cosmos3_policy_droid_trt_eval.yaml", "cosmos3_trt_only"),
            ("cosmos3_policy_droid_pt_trt_compare.yaml", "cosmos3_pt_trt_compare"),
        ],
    )
    def test_eval_yaml_selects_runner(self, configs_dir, cfg, runner) -> None:
        task = TaskConfig.load(configs_dir / cfg)
        assert task.evaluate.policy_runner == runner


@pytest.mark.e2e_slow
@pytest.mark.skipif(
    not (_cuda_available() and _diffusers_cosmos3_available()),
    reason="cosmos3 real export needs CUDA + diffusers cosmos3 weights",
)
class TestCosmos3TrtRealExport:
    def test_real_export_produces_four_onnx(self, configs_dir, tmp_path) -> None:
        task = TaskConfig.load(configs_dir / "cosmos3_policy_droid_trt_deploy.yaml")
        task.actions = ["export"]
        task.output_dir = str(tmp_path / "cosmos3_policy_droid_trt")
        task.deploy.export_dir = str(tmp_path / "cosmos3_policy_droid_trt" / "onnx")
        manifest = WorkflowRunner(task).run(dry_run=False)
        onnx_arts = {a.stage: a for a in manifest.artifacts if a.kind == "onnx"}
        assert set(onnx_arts) == {"vae_encode", "text_embed", "dit", "vae_decode"}
        for art in onnx_arts.values():
            assert not art.metadata.get("reference", True)
