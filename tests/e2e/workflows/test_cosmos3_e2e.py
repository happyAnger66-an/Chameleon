"""cosmos3 E2E — reference infer 与 ONNX export workflow（CPU，无外部权重）。"""

from __future__ import annotations

import pytest

from chameleon.config.schema import TaskConfig
from chameleon.workflows.runner import WorkflowRunner


@pytest.mark.e2e
class TestCosmos3WorkflowE2E:
    def test_cpu_infer_via_workflow(self, configs_dir) -> None:
        task = TaskConfig.load(configs_dir / "cosmos3_cpu.yaml")
        manifest = WorkflowRunner(task).run(dry_run=False)
        kinds = [a.kind for a in manifest.artifacts]
        assert "inference" in kinds

    def test_trt_deploy_dry_run(self, configs_dir) -> None:
        task = TaskConfig.load(configs_dir / "cosmos3_trt_deploy.yaml")
        lines = WorkflowRunner(task).plan()
        joined = "\n".join(lines)
        assert "vae_encode" in joined
        assert "dit" in joined

    @pytest.mark.e2e_slow
    def test_onnx_export_workflow(self, configs_dir, tmp_path) -> None:
        task = TaskConfig.load(configs_dir / "cosmos3_trt_deploy.yaml")
        task.actions = ["export"]
        task.output_dir = str(tmp_path / "cosmos3_trt")
        task.deploy.export_dir = str(tmp_path / "cosmos3_trt" / "onnx")
        manifest = WorkflowRunner(task).run(dry_run=False)
        onnx_arts = {a.stage: a for a in manifest.artifacts if a.kind == "onnx"}
        assert set(onnx_arts) == {"vae_encode", "text_embed", "dit", "vae_decode"}
        for art in onnx_arts.values():
            assert art.path and art.path.endswith(".onnx")
