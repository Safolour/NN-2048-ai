from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from game2048.m2_models import ResidualMLP2048
from game2048.m5_pretrain import (
    SCALE_GAMES,
    SCALE_ROWS,
    TRAINING_SEEDS,
    load_label_split,
    make_optimizer,
    paired_bootstrap,
    parameter_update_checks,
    policy_only_step,
    scale_gate_131k,
    scale_shard_ranges,
    shard_game_ids,
    snapshot_parameters,
    source_attempt_seed,
    split_game_seed,
    stateless_epoch_plan,
    teacher_best_action,
    validate_completed_shard,
)
from game2048.reference_env import move_without_spawn
from game2048.symmetry import transform_action, transform_board


def test_stateless_epoch_plan_is_deterministic():
    order_a, tids_a = stateless_epoch_plan("32k", TRAINING_SEEDS[0], 7, rows=257)
    order_b, tids_b = stateless_epoch_plan("32k", TRAINING_SEEDS[0], 7, rows=257)
    order_c, tids_c = stateless_epoch_plan("32k", TRAINING_SEEDS[0], 8, rows=257)
    assert np.array_equal(order_a, order_b)
    assert np.array_equal(tids_a, tids_b)
    assert sorted(order_a.tolist()) == list(range(257))
    assert tids_a.dtype == np.uint8
    assert np.all(tids_a < 8)
    assert not (
        np.array_equal(order_a, order_c)
        and np.array_equal(tids_a, tids_c)
    )


def test_scale_row_counts_and_game_boundaries():
    assert SCALE_ROWS == {"32k": 32768, "131k": 131072, "524k": 524288}
    assert SCALE_GAMES == {"32k": 256, "131k": 1024, "524k": 4096}
    assert len(scale_shard_ranges("32k")) == 16
    assert len(scale_shard_ranges("131k")) == 64
    assert len(scale_shard_ranges("524k")) == 256
    for scale in ("32k", "131k", "524k"):
        ranges = scale_shard_ranges(scale)
        assert ranges[0] == (0, 2048)
        assert ranges[-1][1] == SCALE_ROWS[scale]
        assert all(end - start == 2048 for start, end in ranges)


def test_teacher_target_ignores_illegal_actions():
    values = np.array(
        [[1.0, 999999.0, 3.0, np.nan]], dtype=np.float64
    )
    legal = np.array([[True, False, True, False]], dtype=np.bool_)
    target = teacher_best_action(values, legal)
    assert target.tolist() == [2]
    values[0, 1] = -999999.0
    assert teacher_best_action(values, legal).tolist() == [2]


def test_d4_board_action_alignment():
    board = np.array([
        1, 1, 0, 0,
        2, 0, 2, 0,
        3, 0, 0, 3,
        0, 4, 4, 0,
    ], dtype=np.uint8)
    for tid in range(8):
        transformed_board = transform_board(board, tid)
        for action in range(4):
            original = move_without_spawn(board, action)
            transformed_action = transform_action(action, tid)
            observed = move_without_spawn(
                transformed_board, transformed_action
            )
            assert observed.moved == original.moved
            assert observed.reward == original.reward
            if original.moved:
                assert np.array_equal(
                    observed.afterstate,
                    transform_board(original.afterstate, tid),
                )


def test_training_path_never_consumes_test_split(tmp_path: Path):
    manifest = {"splits": {"test": {"shards": []}}}
    with pytest.raises(
        RuntimeError, match="M5_BLOCKED_TEST_ISOLATION"
    ):
        load_label_split(tmp_path, manifest, "test")


def test_policy_only_loss_leaves_value_heads_untouched():
    torch.manual_seed(20262101)
    model = ResidualMLP2048()
    before = snapshot_parameters(model)
    optimizer = make_optimizer(model)
    boards = torch.randint(0, 12, (32, 16), dtype=torch.uint8)
    target = torch.randint(0, 4, (32,), dtype=torch.long)
    loss = policy_only_step(model, boards, target, optimizer)
    checks = parameter_update_checks(model, before)
    assert np.isfinite(loss)
    assert checks["q_head_updated"]
    assert checks["backbone_updated"]
    assert checks["value_heads_unchanged"]


def test_scale_gate_is_deterministic():
    scores32 = np.arange(2000, dtype=np.float64)
    scores131 = scores32 + 10.0
    m32 = {
        "ce": 1.0,
        "legal_best_action_accuracy": 0.40,
        "legal_pairwise_ranking_accuracy": 0.60,
    }
    m131 = {
        "ce": 0.98,
        "legal_best_action_accuracy": 0.41,
        "legal_pairwise_ranking_accuracy": 0.61,
    }
    first = scale_gate_131k(
        scores32, scores131, m32, m131
    )
    second = scale_gate_131k(
        scores32, scores131, m32, m131
    )
    assert first == second
    assert first["trigger_524k"] is True
    assert first["pre_524_winner"] == "131k"


def test_paired_bootstrap_is_deterministic():
    a = np.linspace(0.0, 1000.0, 2000)
    b = a - np.sin(np.arange(2000))
    first = paired_bootstrap(a, b, seed=12345)
    second = paired_bootstrap(a, b, seed=12345)
    assert first == second
    assert first["pairs"] == 2000
    assert len(first["ci95"]) == 2


def test_game_shard_resume_is_seed_stable():
    ids = shard_game_ids("train", 17)
    assert ids.tolist() == list(range(272, 288))
    seeds = [
        split_game_seed("train", int(gid))[1]
        for gid in ids
    ]
    assert seeds == list(range(20265273, 20265289))
    gid, canonical, effective0 = source_attempt_seed("train", 2682, 0)
    assert (gid, canonical, effective0) == (2682, 20267683, 20267683)
    _, canonical1, effective1 = source_attempt_seed("train", 2682, 1)
    assert canonical1 == canonical
    assert effective1 == 1020267683
    _, _, effective2 = source_attempt_seed("train", 2682, 2)
    assert effective2 == 2020267683
    val_ids = shard_game_ids("validation", 2)
    assert val_ids.tolist() == list(range(100032, 100048))


def test_manifest_rejects_corrupt_completed_shard(
    tmp_path: Path,
):
    path = tmp_path / "shard_0000.npz"
    np.savez(
        path,
        state=np.zeros((1, 16), dtype=np.uint8),
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    with pytest.raises(
        RuntimeError,
        match="M5_BLOCKED_DATASET_ARTIFACT_CORRUPT",
    ):
        validate_completed_shard(path, digest)
