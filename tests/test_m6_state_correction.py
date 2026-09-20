from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from game2048.m2_models import ResidualMLP2048
from game2048.m4_compare import make_game_rngs
from game2048.m6_state_correction import (
    GAMES_PER_SHARD,
    M5_CHECKPOINT_SHA256,
    SHARD_STATES,
    TEACHER_SHA256,
    TEACHER_VERSION,
    TRAINING_SEEDS,
    WORK_ORDER_VERSION,
    assert_test_allowed,
    correction_anchor_batch,
    epoch_plan,
    fresh_model_from_state,
    paired_bootstrap,
    parameter_update_checks,
    rollout_student_games,
    select_candidate,
    snapshot_parameters,
    source_attempt_seed,
    split_game_seed,
    teacher_best_action,
    training_label_manifest_sha256,
    validate_completed_shard,
)


class _TieModel(torch.nn.Module):
    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        return torch.zeros((boards.shape[0], 4), dtype=torch.float32, device=boards.device)


def test_student_rollout_is_seed_stable():
    model = _TieModel()
    first = rollout_student_games(model, [20288001], torch.device("cpu"))[0]
    second = rollout_student_games(model, [20288001], torch.device("cpu"))[0]
    for key in (
        "game_seed", "moves", "final_score", "max_tile_exp_terminal"
    ):
        assert first[key] == second[key]
    for key in (
        "state", "student_logits", "student_action",
        "current_score", "max_tile_exp",
    ):
        assert np.array_equal(first[key], second[key])


def test_student_rollout_uses_frozen_policy_and_independent_rngs():
    spawn_a, tie_a = make_game_rngs(123456)
    spawn_b, tie_b = make_game_rngs(123456)
    _ = tie_a.integers(0, 4, size=1000)
    assert np.array_equal(
        spawn_a.integers(0, 2**31, size=64),
        spawn_b.integers(0, 2**31, size=64),
    )
    assert np.array_equal(
        tie_b.integers(0, 4, size=32),
        make_game_rngs(123456)[1].integers(0, 4, size=32),
    )
    result = rollout_student_games(_TieModel(), [20288002], torch.device("cpu"))[0]
    assert result["student_logits"].dtype == np.float32
    assert result["student_action"].dtype == np.uint8
    assert result["state"].shape[1] == 16


def test_teacher_relabel_ignores_illegal_actions():
    values = np.array([[1.0, 1e30, 3.0, np.nan]], dtype=np.float64)
    legal = np.array([[True, False, True, False]], dtype=np.bool_)
    assert teacher_best_action(values, legal).tolist() == [2]
    values[0, 1] = -1e30
    assert teacher_best_action(values, legal).tolist() == [2]


def test_source_splits_are_disjoint_and_test_isolated():
    train = {split_game_seed("train", i)[1] for i in range(1024)}
    validation = {split_game_seed("validation", i)[1] for i in range(64)}
    test = {split_game_seed("test", i)[1] for i in range(64)}
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    gid, canonical, effective = source_attempt_seed("validation", 3, 2)
    assert gid == 100003
    assert canonical == 20290004
    assert effective == 2020290004
    with pytest.raises(RuntimeError, match="M6_BLOCKED_TEST_ISOLATION"):
        assert_test_allowed("TRAIN_DONE")
    assert_test_allowed("CANDIDATE_LOCKED")


def test_correction_anchor_batch_is_exactly_half_and_deterministic():
    plan_a = epoch_plan(TRAINING_SEEDS[0], 0, correction_rows=1024, anchor_rows=2048)
    plan_b = epoch_plan(TRAINING_SEEDS[0], 0, correction_rows=1024, anchor_rows=2048)
    for key in plan_a:
        assert np.array_equal(plan_a[key], plan_b[key])
    assert np.unique(plan_a["anchor_subset"]).size == 1024

    correction_state = np.ones((1024, 16), dtype=np.uint8)
    anchor_state = np.full((2048, 16), 2, dtype=np.uint8)
    correction_target = np.zeros(1024, dtype=np.int64)
    anchor_target = np.ones(2048, dtype=np.int64)
    boards_a, target_a = correction_anchor_batch(
        correction_state, correction_target,
        anchor_state, anchor_target, plan_a, 0, torch.device("cpu")
    )
    boards_b, target_b = correction_anchor_batch(
        correction_state, correction_target,
        anchor_state, anchor_target, plan_b, 0, torch.device("cpu")
    )
    assert boards_a.shape == (1024, 16)
    assert target_a.shape == (1024,)
    assert torch.all(boards_a[:512] == 1)
    assert torch.all(boards_a[512:] == 2)
    assert torch.equal(boards_a, boards_b)
    assert torch.equal(target_a, target_b)


def test_finetune_starts_from_exact_m5_checkpoint_with_fresh_optimizer():
    torch.manual_seed(77)
    baseline = ResidualMLP2048()
    state = {k: v.detach().clone() for k, v in baseline.state_dict().items()}
    model_a, optimizer_a = fresh_model_from_state(
        state, TRAINING_SEEDS[0], torch.device("cpu")
    )
    model_b, optimizer_b = fresh_model_from_state(
        state, TRAINING_SEEDS[1], torch.device("cpu")
    )
    for key, value in state.items():
        assert torch.equal(model_a.state_dict()[key], value)
        assert torch.equal(model_b.state_dict()[key], value)
    assert optimizer_a is not optimizer_b
    assert len(optimizer_a.state) == 0
    assert len(optimizer_b.state) == 0


def test_policy_only_correction_leaves_value_heads_untouched():
    torch.manual_seed(20263101)
    model = ResidualMLP2048()
    before = snapshot_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-4,
        betas=(0.9, 0.999), eps=1e-8,
        amsgrad=False, foreach=False, fused=False, capturable=False,
    )
    boards = torch.randint(0, 12, (16, 16), dtype=torch.uint8)
    target = torch.randint(0, 4, (16,), dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(boards), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    checks = parameter_update_checks(model, before)
    assert np.isfinite(float(loss.item()))
    assert checks["q_head_updated"]
    assert checks["backbone_updated"]
    assert checks["value_heads_unchanged"]


def test_candidate_selection_and_paired_bootstrap_are_deterministic():
    means = {20263101: 100.0, 20263102: 110.0, 20263103: 110.0}
    assert select_candidate(means) == 20263102
    base = np.arange(2000, dtype=np.float64)
    candidate = base + np.sin(np.arange(2000)) + 5.0
    first = paired_bootstrap(candidate, base, seed=20266201)
    second = paired_bootstrap(candidate, base, seed=20266201)
    assert first == second
    assert first["pairs"] == 2000


def test_final_test_cannot_run_before_candidate_lock():
    for state in ("P0_PASSED", "DATA_READY", "TRAIN_DONE", "DEV_DONE"):
        with pytest.raises(RuntimeError, match="M6_BLOCKED_TEST_ISOLATION"):
            assert_test_allowed(state)
    for state in ("CANDIDATE_LOCKED", "GAMEPLAY_VALIDATION_DONE", "PROMOTION_DECIDED"):
        assert_test_allowed(state)


def test_manifest_rejects_corrupt_completed_shard(tmp_path: Path):
    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest().upper()

    def write_manifest(path: Path, payload: dict) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )

    train_source = tmp_path / "source_train_manifest.json"
    validation_source = tmp_path / "source_validation_manifest.json"
    test_source = tmp_path / "source_test_manifest.json"
    train_source.write_text("train-source", encoding="utf-8")
    validation_source.write_text("validation-source", encoding="utf-8")
    test_source.write_text("test-source", encoding="utf-8")

    train_shard = tmp_path / "train_shard.bin"
    validation_shard = tmp_path / "validation_shard.bin"
    train_shard.write_bytes(b"train-shard")
    validation_shard.write_bytes(b"validation-shard")

    manifest = {
        "schema_version": 1,
        "work_order_version": WORK_ORDER_VERSION,
        "student_checkpoint_sha256": M5_CHECKPOINT_SHA256,
        "teacher_checkpoint_sha256": TEACHER_SHA256,
        "teacher_version": TEACHER_VERSION,
        "decision_depth": 3,
        "value_semantics": "SEARCH_VALUE_RAW_LEAF",
        "shard_states": SHARD_STATES,
        "games_per_shard": GAMES_PER_SHARD,
        "splits": {
            "train": {
                "shards": [{
                    "shard_index": 0,
                    "status": "complete",
                    "path": train_shard.name,
                    "sha256": file_sha(train_shard),
                    "rows": SHARD_STATES,
                }],
                "source_manifest_path": train_source.name,
                "source_manifest_sha256": file_sha(train_source),
            },
            "validation": {
                "shards": [{
                    "shard_index": 0,
                    "status": "complete",
                    "path": validation_shard.name,
                    "sha256": file_sha(validation_shard),
                    "rows": SHARD_STATES,
                }],
                "source_manifest_path": validation_source.name,
                "source_manifest_sha256": file_sha(validation_source),
            },
            "test": {"shards": []},
        },
    }
    manifest_path = tmp_path / "label_manifest.json"
    write_manifest(manifest_path, manifest)
    training_sha = training_label_manifest_sha256(manifest_path, tmp_path)

    test_shard = tmp_path / "test_shard_0000.npz"
    np.savez(test_shard, state=np.zeros((1, 16), dtype=np.uint8))
    test_digest = file_sha(test_shard)
    manifest["splits"]["test"] = {
        "shards": [{
            "shard_index": 0,
            "status": "complete",
            "path": test_shard.name,
            "sha256": test_digest,
            "rows": SHARD_STATES,
            "test_only_metadata": "after-candidate-lock",
        }],
        "source_manifest_path": test_source.name,
        "source_manifest_sha256": file_sha(test_source),
    }
    write_manifest(manifest_path, manifest)
    assert training_label_manifest_sha256(manifest_path, tmp_path) == training_sha

    with pytest.raises(
        RuntimeError, match="M6_BLOCKED_DATASET_ARTIFACT_CORRUPT"
    ):
        validate_completed_shard(test_shard, test_digest)

    train_shard.write_bytes(b"train-shard-corrupt")
    with pytest.raises(
        RuntimeError, match="M6_BLOCKED_DATASET_ARTIFACT_CORRUPT"
    ):
        training_label_manifest_sha256(manifest_path, tmp_path)
    train_shard.write_bytes(b"train-shard")

    validation_shard.write_bytes(b"validation-shard-corrupt")
    with pytest.raises(
        RuntimeError, match="M6_BLOCKED_DATASET_ARTIFACT_CORRUPT"
    ):
        training_label_manifest_sha256(manifest_path, tmp_path)
