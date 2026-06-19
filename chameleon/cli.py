"""命令行入口 — 薄包装，触发插件注册并启动 launcher。

作用：
    用户-facing 入口（``chameleon`` / ``python -m chameleon.cli``）。
    具体子命令实现在 ``chameleon/commands/``，由 ``launcher.launch()`` 分发。

架构位置：
    入口/编排层 — 不含业务逻辑。
"""

from __future__ import annotations

import chameleon  # noqa: F401  (triggers import-time registration)
from chameleon.launcher import launch


def main(argv: list[str] | None = None) -> int:
    return launch(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
