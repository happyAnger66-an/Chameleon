"""cosmos3 DROID 桥接纯函数单测（无 TF/torch 依赖）。"""

from __future__ import annotations

import numpy as np

from chameleon.dataloader.cosmos3_droid import (
    DROID_ACTION_HORIZON,
    DROID_RAW_ACTION_DIM,
    build_cosmos3_gt_chunk,
    build_cosmos3_observation,
    cartesian_gripper_to_cosmos10,
    euler_to_rotation_6d,
)


def test_euler_zero_is_identity_first_two_cols() -> None:
    rot6d = euler_to_rotation_6d([0.0, 0.0, 0.0])
    # 单位阵前两列 = [1,0,0, 0,1,0]
    assert rot6d.shape == (6,)
    np.testing.assert_allclose(rot6d, [1, 0, 0, 0, 1, 0], atol=1e-6)


def test_euler_6d_columns_are_orthonormal() -> None:
    rot6d = euler_to_rotation_6d([0.3, -0.7, 1.1])
    c0, c1 = rot6d[:3], rot6d[3:]
    np.testing.assert_allclose(np.linalg.norm(c0), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(c1), 1.0, atol=1e-5)
    np.testing.assert_allclose(float(c0 @ c1), 0.0, atol=1e-5)


def test_cartesian_gripper_to_cosmos10_layout() -> None:
    vec = cartesian_gripper_to_cosmos10([0.1, 0.2, 0.3, 0.0, 0.0, 0.0], [0.9])
    assert vec.shape == (10,)
    np.testing.assert_allclose(vec[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(vec[3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(vec[9], 0.9, atol=1e-6)


def test_cartesian_handles_scalar_gripper_and_short_input() -> None:
    vec = cartesian_gripper_to_cosmos10([0.5, 0.5, 0.5], 0.0)
    assert vec.shape == (10,)
    np.testing.assert_allclose(vec[:3], [0.5, 0.5, 0.5], atol=1e-6)


def test_build_gt_chunk_shape_and_tail_repeat() -> None:
    cart = np.tile(np.array([0.1, 0.2, 0.3, 0, 0, 0], np.float32), (3, 1))
    cart[2, 0] = 9.0  # last frame distinct
    grip = np.zeros((3, 1), np.float32)
    chunk = build_cosmos3_gt_chunk(cart, grip, start=1)
    assert chunk.shape == (DROID_ACTION_HORIZON, DROID_RAW_ACTION_DIM)
    # start=1 -> frame1 then frame2 then repeat frame2 (tail)
    np.testing.assert_allclose(chunk[0, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(chunk[1, 0], 9.0, atol=1e-6)
    np.testing.assert_allclose(chunk[-1, 0], 9.0, atol=1e-6)  # repeats last


def test_build_gt_chunk_empty_is_zeros() -> None:
    chunk = build_cosmos3_gt_chunk(np.zeros((0, 6)), np.zeros((0, 1)), start=0)
    assert chunk.shape == (DROID_ACTION_HORIZON, DROID_RAW_ACTION_DIM)
    assert not np.any(chunk)


def test_build_observation_keys() -> None:
    ext = np.zeros((8, 8, 3), np.uint8)
    wrist = np.ones((8, 8, 3), np.uint8)
    obs = build_cosmos3_observation(ext, wrist, "pick up the cube")
    assert set(obs) == {"image", "wrist_image", "prompt"}
    assert obs["image"].shape == (8, 8, 3)
    assert obs["prompt"] == "pick up the cube"


def test_build_observation_optional_wrist_prompt() -> None:
    obs = build_cosmos3_observation(np.zeros((4, 4, 3), np.uint8))
    assert set(obs) == {"image"}
