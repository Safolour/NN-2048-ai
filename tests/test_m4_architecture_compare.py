from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from game2048.m2_models import ResidualMLP2048
from game2048.m2_symmetry import transform_action_batch, transform_board_batch
from game2048.m4_compare import (
    build_training_plan,
    make_game_rngs,
    paired_bootstrap,
    score_summary,
    select_game_actions,
    training_indices,
)
from game2048.reference_env import move_without_spawn, spawn_random


def test_training_plan_is_identical_across_architectures():
    train_indices = np.arange(32, dtype=np.int64)
    transformer_plan = build_training_plan(
        train_indices, 20260919, epochs=3
    )
    mlp_plan = build_training_plan(train_indices, 20260919, epochs=3)
    assert transformer_plan.sha256 == mlp_plan.sha256
    assert np.array_equal(
        transformer_plan.sample_indices, mlp_plan.sample_indices
    )
    assert np.array_equal(
        transformer_plan.transform_ids, mlp_plan.transform_ids
    )


def test_d4_board_and_action_transform_alignment():
    board = np.array(
        [
            1, 1, 0, 0,
            2, 0, 2, 0,
            3, 0, 0, 3,
            0, 4, 4, 0,
        ],
        dtype=np.uint8,
    )
    action = 2
    original = move_without_spawn(board, action)
    assert original.moved
    for transform_id in range(8):
        tids = torch.tensor([transform_id], dtype=torch.long)
        transformed_board = transform_board_batch(
            torch.from_numpy(board[None, :]), tids
        )[0].numpy()
        transformed_action = int(
            transform_action_batch(
                torch.tensor([action], dtype=torch.long), tids
            )[0].item()
        )
        transformed_move = move_without_spawn(
            transformed_board, transformed_action
        )
        expected_afterstate = transform_board_batch(
            torch.from_numpy(original.afterstate[None, :]), tids
        )[0].numpy()
        assert transformed_move.moved
        assert transformed_move.reward == original.reward
        assert np.array_equal(
            transformed_move.afterstate, expected_afterstate
        )


def test_gameplay_masks_illegal_actions():
    logits = np.array(
        [[1000.0, 1.0, 2.0, 999.0]], dtype=np.float32
    )
    legal = np.array([[False, True, True, False]], dtype=np.bool_)
    _spawn_rng, tie_rng = make_game_rngs(20261001)
    actions = select_game_actions(logits, legal, [tie_rng])
    assert actions.tolist() == [2]


def test_spawn_rng_is_independent_from_tie_rng():
    board = np.array([1, 2, 3, 0] + [0] * 12, dtype=np.uint8)
    spawn_a, tie_a = make_game_rngs(20261001)
    spawn_b, tie_b = make_game_rngs(20261001)
    for _ in range(100):
        tie_b.integers(0, 4)
    first = spawn_random(board, spawn_a)
    second = spawn_random(board, spawn_b)
    assert np.array_equal(first.state, second.state)
    assert first.spawn_index == second.spawn_index
    assert first.spawn_exponent == second.spawn_exponent
    assert tie_a.bit_generator.state != tie_b.bit_generator.state


def test_paired_bootstrap_is_deterministic():
    transformer = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    mlp = np.array([8, 21, 27, 40, 45], dtype=np.int64)
    first = paired_bootstrap(
        transformer, mlp, seed=20261101, resamples=200
    )
    second = paired_bootstrap(
        transformer, mlp, seed=20261101, resamples=200
    )
    assert first == second
    assert first["mean_delta"] == float(
        np.mean(transformer - mlp)
    )


def test_game_score_aggregation():
    summary = score_summary(
        np.array([100, 200, 300, 400], dtype=np.int64),
        np.array([10, 11, 12, 16], dtype=np.uint8),
        np.array([50, 60, 70, 80], dtype=np.int64),
    )
    assert summary["games"] == 4
    assert summary["mean_score"] == 250.0
    assert summary["median_score"] == 250.0
    assert summary["mean_moves"] == 65.0
    assert summary["reach"]["2048"] == 0.75
    assert summary["reach"]["65536"] == 0.25
    assert summary["max_tile_distribution"] == {
        "10": 1,
        "11": 1,
        "12": 1,
        "16": 1,
    }


def test_training_path_never_consumes_test_split():
    split = np.array(
        ["train", "test", "validation", "train", "test", "train"],
        dtype="<U10",
    )
    ids = training_indices(split)
    assert ids.tolist() == [0, 3, 5]
    assert np.all(split[ids] == "train")
    assert not np.any(split[ids] == "test")


def test_policy_only_loss_leaves_value_heads_untouched():
    torch.manual_seed(20260919)
    model = ResidualMLP2048()
    initial_value_heads = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("value_head.")
        or name.startswith("afterstate_head.")
    }
    initial_q = model.q_head.weight.detach().clone()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
    )
    boards = torch.tensor(
        [
            [1, 1, 0, 0] + [0] * 12,
            [1, 2, 3, 4] + [0] * 12,
            [0, 1, 0, 2] + [3, 0, 0, 0] + [0] * 8,
            [4, 3, 2, 1] + [0] * 12,
        ],
        dtype=torch.uint8,
    )
    target = torch.tensor([2, 0, 3, 1], dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(boards), target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    assert not torch.equal(initial_q, model.q_head.weight)
    current = dict(model.named_parameters())
    for name, initial in initial_value_heads.items():
        assert torch.equal(initial, current[name])
