import pytest
import torch

from game2048.m2_policy import apply_legal_mask, select_greedy_actions


def test_illegal_highest_q_is_never_selected():
    q = torch.tensor([[100.0, 3.0, 2.0, 1.0]])
    legal = torch.tensor([[False, True, True, True]])
    assert select_greedy_actions(q, legal).item() == 1
    masked = apply_legal_mask(q, legal)
    assert torch.isneginf(masked[0, 0])


def test_single_legal_action_is_selected():
    q = torch.tensor([[99.0, 98.0, 97.0, -100.0]])
    legal = torch.tensor([[False, False, False, True]])
    assert select_greedy_actions(q, legal).item() == 3


def test_unique_maximum_is_selected():
    q = torch.tensor([[1.0, 5.0, 3.0, 4.0]])
    legal = torch.ones((1, 4), dtype=torch.bool)
    assert select_greedy_actions(q, legal).item() == 1


def test_exact_tie_only_selects_tied_legal_actions():
    q = torch.tensor([[7.0, 7.0, 7.0, 1.0]])
    legal = torch.tensor([[True, True, False, True]])
    generator = torch.Generator().manual_seed(123)
    seen = {
        int(select_greedy_actions(q, legal, generator=generator).item())
        for _ in range(100)
    }
    assert seen == {0, 1}


def test_near_tie_within_tolerance_and_outside_tolerance():
    legal = torch.ones((1, 4), dtype=torch.bool)
    g = torch.Generator().manual_seed(9)
    q_near = torch.tensor([[5.0, 5.0 - 5e-7, 0.0, 0.0]])
    seen = {int(select_greedy_actions(q_near, legal, generator=g).item()) for _ in range(100)}
    assert seen == {0, 1}
    q_far = torch.tensor([[5.0, 5.0 - 2e-6, 0.0, 0.0]])
    assert select_greedy_actions(q_far, legal).item() == 0


def test_all_false_legal_mask_is_rejected():
    with pytest.raises(ValueError):
        select_greedy_actions(
            torch.zeros((1, 4)),
            torch.zeros((1, 4), dtype=torch.bool),
        )


def test_fixed_generator_is_reproducible():
    q = torch.zeros((64, 4))
    legal = torch.ones((64, 4), dtype=torch.bool)
    a = select_greedy_actions(q, legal, generator=torch.Generator().manual_seed(77))
    b = select_greedy_actions(q, legal, generator=torch.Generator().manual_seed(77))
    assert torch.equal(a, b)
