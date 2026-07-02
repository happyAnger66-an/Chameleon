"""Chameleon WebUI — 浏览器端配置编排前端（M1：选择 / 查看 / 编辑 / 校验配置）。

作用：
    在本地起一个轻量 Web 服务，把 ``configs/*.yaml`` 以下拉列表呈现，选中后
    可在浏览器里查看并编辑 YAML，并调用 pydantic schema 做即时校验。后续阶段
    (M2/M3) 会在此基础上加入「子进程运行 + 流式日志 + 进度/产物展示」。

架构位置：
    入口/编排层 — 与 CLI 平行的另一种入口。后端仅做配置 IO 与（后续）子进程
    编排，不直接持有 torch/tensorrt 等重依赖；真正的任务执行仍走
    ``python -m chameleon.cli <command> --config <yaml>``。
"""

from __future__ import annotations

__all__ = ["create_app", "run_webui"]


def create_app(*args, **kwargs):  # noqa: ANN002, ANN003
    """Lazy re-export，避免在未安装 fastapi 时导入即报错。"""
    from chameleon.webui.server import create_app as _create_app

    return _create_app(*args, **kwargs)


def run_webui(*args, **kwargs):  # noqa: ANN002, ANN003
    from chameleon.webui.server import run_webui as _run_webui

    return _run_webui(*args, **kwargs)
