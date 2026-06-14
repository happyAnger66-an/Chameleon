"""Evaluation utilities.

Compares actions produced by two configurations (e.g. PyTorch reference vs a
quantized/compiled path) for accuracy regression checks.
"""

from chameleon.evaluate.compare import ActionDiff, compare_actions

__all__ = ["ActionDiff", "compare_actions"]
