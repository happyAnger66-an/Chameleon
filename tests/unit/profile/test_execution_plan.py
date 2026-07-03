"""execution_plan 单元测试。"""

from __future__ import annotations

from chameleon.config.schema import CompileStep, ExportStep, TaskConfig
from chameleon.profile.execution_plan import PlanMode, build_execution_plan


def test_deploy_plan_skips_expert_when_denoise_present() -> None:
    task = TaskConfig(
        deploy={"backend": "pi05"},
        export=[
            ExportStep(stage="vit"),
            ExportStep(stage="llm"),
            ExportStep(stage="expert"),
            ExportStep(stage="denoise"),
        ],
        infer={"num_steps": 10, "batch_size": 1},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.DEPLOY
    stages = [(s.stage, s.repeat) for s in plan.stages]
    assert stages == [("vit", 1), ("llm", 1), ("denoise", 10)]
    assert "expert" not in [s.stage for s in plan.stages]


def test_deploy_plan_uses_expert_without_denoise() -> None:
    task = TaskConfig(
        deploy={"backend": "pi05"},
        export=[ExportStep(stage="vit"), ExportStep(stage="llm"), ExportStep(stage="expert")],
        infer={"num_steps": 5},
    )
    plan = build_execution_plan(task)
    stages = [(s.stage, s.repeat) for s in plan.stages]
    assert stages == [("vit", 1), ("llm", 1), ("expert", 5)]


def test_reference_plan_three_stages() -> None:
    task = TaskConfig(
        model_overrides={"use_reference": True},
        infer={"num_steps": 7, "batch_size": 2},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.REFERENCE
    assert plan.batch_size == 2
    assert [(s.stage, s.repeat) for s in plan.stages] == [
        ("vit", 1),
        ("llm_prefix", 1),
        ("action_expert", 7),
    ]


def test_cosmos3_reference_plan_four_stages() -> None:
    task = TaskConfig(
        architecture="cosmos3",
        model_overrides={"use_reference": True},
        infer={"num_steps": 6, "batch_size": 1},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.REFERENCE
    assert [(s.stage, s.repeat) for s in plan.stages] == [
        ("vae_encode", 1),
        ("text_embed", 1),
        ("dit", 6),
        ("vae_decode", 1),
    ]


def test_cosmos3_real_plan_uses_generate_steps() -> None:
    task = TaskConfig(
        architecture="cosmos3",
        model_overrides={"use_reference": False, "model_id": "nvidia/Cosmos3-Nano"},
        generate={"num_inference_steps": 35},
        infer={"batch_size": 1},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.REAL
    assert plan.num_steps == 35
    assert [s.stage for s in plan.stages] == ["vae_encode", "text_embed", "dit", "vae_decode"]
    assert plan.stages[2].repeat == 35


def test_cosmos3_deploy_plan() -> None:
    task = TaskConfig(
        architecture="cosmos3",
        deploy={"backend": "cosmos3"},
        export=[
            ExportStep(stage="vae_encode"),
            ExportStep(stage="text_embed"),
            ExportStep(stage="dit"),
            ExportStep(stage="vae_decode"),
        ],
        infer={"num_steps": 6},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.DEPLOY
    assert [s.stage for s in plan.stages] == ["vae_encode", "text_embed", "dit", "vae_decode"]


def test_real_plan_matches_deploy_stages() -> None:
    task = TaskConfig(
        model_overrides={"use_reference": False, "checkpoint": "models/x/model.safetensors"},
        compile=[
            CompileStep(stage="vit"),
            CompileStep(stage="llm"),
            CompileStep(stage="denoise"),
        ],
        infer={"num_steps": 10},
    )
    plan = build_execution_plan(task)
    assert plan.mode == PlanMode.REAL
    assert [s.stage for s in plan.stages] == ["vit", "llm", "denoise"]
