"""Build artifacts and a lightweight manifest for provenance tracking.

Every pipeline step (quantize -> compile -> infer) produces an :class:`Artifact`.
A :class:`Manifest` records the chain of artifacts so that a later step can
consume the output of an earlier one, mirroring ``model_optimizer``'s
``artifact_manifest.json`` but unified across platforms.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    """A single build product (a graph, a quantized module, a compiled engine...)."""

    kind: str
    """``reference`` | ``onnx`` | ``quantized`` | ``engine`` | ``checkpoint``."""

    stage: str | None = None
    platform: str | None = None
    path: str | None = None
    """On-disk location, when the artifact is serialized."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Serializable side-information (shapes, dtype, quant config, build flags)."""

    payload: Any = field(default=None, repr=False, compare=False)
    """In-memory object (e.g. an ``nn.Module`` or a TRT engine handle). Never serialized."""

    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("payload", None)
        return data


class Manifest:
    """Ordered record of artifacts produced by a workflow."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.artifacts: list[Artifact] = []

    @property
    def path(self) -> Path:
        return self.output_dir / "chameleon_manifest.json"

    def add(self, artifact: Artifact) -> Artifact:
        self.artifacts.append(artifact)
        return artifact

    def latest(self, *, kind: str | None = None, stage: str | None = None) -> Artifact | None:
        for artifact in reversed(self.artifacts):
            if kind is not None and artifact.kind != kind:
                continue
            if stage is not None and artifact.stage != stage:
                continue
            return artifact
        return None

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "chameleon_manifest_v1",
            "artifacts": [a.to_dict() for a in self.artifacts],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        return self.path

    @classmethod
    def load(cls, output_dir: str | Path) -> "Manifest":
        manifest = cls(output_dir)
        if manifest.path.exists():
            data = json.loads(manifest.path.read_text())
            for entry in data.get("artifacts", []):
                manifest.artifacts.append(Artifact(**entry))
        return manifest
