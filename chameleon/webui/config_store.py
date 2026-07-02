"""配置仓库 — 列举 / 读取 / 校验 ``configs/*.yaml``。

作用：
    为 WebUI 后端提供纯粹的配置 IO 与校验能力，与 Web 框架解耦，便于单测。
    校验复用 ``TaskConfig``（pydantic v2）的 schema，不触发任何模型加载或运行。

架构位置：
    编排层的支撑模块 — 被 ``chameleon.webui.server`` 调用；只依赖标准库、yaml
    与 config schema。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigStoreError(Exception):
    """配置路径非法或越权访问时抛出。"""


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: str | None
    actions: list[str]


class ConfigStore:
    """限定在 ``root`` 目录内的 YAML 配置读取器。

    仅暴露顶层 ``*.yaml`` / ``*.yml`` 文件；所有按名访问都会做 root 归属校验，
    防止 ``../`` 之类的路径穿越。
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def list_configs(self) -> list[str]:
        if not self.root.is_dir():
            return []
        names = [
            p.name
            for p in sorted(self.root.iterdir())
            if p.is_file() and p.suffix in (".yaml", ".yml")
        ]
        return names

    def _resolve(self, name: str) -> Path:
        # 只接受纯文件名，杜绝子目录与路径穿越。
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            raise ConfigStoreError(f"非法配置名: {name!r}")
        path = (self.root / name).resolve()
        if path.parent != self.root:
            raise ConfigStoreError(f"配置越权访问: {name!r}")
        if not path.is_file():
            raise ConfigStoreError(f"配置不存在: {name!r}")
        return path

    def read_text(self, name: str) -> str:
        return self._resolve(name).read_text(encoding="utf-8")

    @staticmethod
    def _extract_actions(data: object) -> list[str]:
        if isinstance(data, dict):
            actions = data.get("actions")
            if isinstance(actions, list):
                return [str(a) for a in actions]
        return []

    def validate_text(self, text: str) -> ValidationResult:
        """解析并用 ``TaskConfig`` schema 校验 YAML 文本，不运行任何任务。"""
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            return ValidationResult(ok=False, error=f"YAML 解析错误: {exc}", actions=[])

        if not isinstance(data, dict):
            return ValidationResult(ok=False, error="配置顶层必须是映射(dict)。", actions=[])

        actions = self._extract_actions(data)
        try:
            from chameleon.config.schema import TaskConfig

            TaskConfig.model_validate(data)
        except Exception as exc:  # noqa: BLE001 — 面向用户回显任意校验错误
            return ValidationResult(ok=False, error=str(exc), actions=actions)
        return ValidationResult(ok=True, error=None, actions=actions)
