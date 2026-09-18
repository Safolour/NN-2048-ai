from __future__ import annotations

import numpy as np

from game2048.m3_search import ExpectimaxTeacher
from game2048.m3_teacher_data import build_teacher_sample, split_game_ids
from game2048.reference_env import ACTIONS, move_without_spawn

class ZeroLeaf:
    def formal_state_leaf(self, state: np.ndarray) -> float:
        return 0.0

def test_teacher_sample_four_action_schema_and_illegal_sentinels():
    state = np.array([1,0,0,0] + [0] * 12, dtype=np.uint8)
    search = ExpectimaxTeacher(ZeroLeaf(), decision_depth=1)
    sample = build_teacher_sample(
        search, state, game_id=7, step_index=3, current_score=12,
        teacher_version="fixture", checkpoint_sha256="abc", data_source="synthetic",
    )
    assert sample.state.shape == (16,)
    assert sample.teacher_value.shape == (4,)
    assert sample.reward.shape == (4,)
    assert sample.afterstate.shape == (4,16)
    assert sample.legal_mask.shape == (4,)
    for action in ACTIONS:
        i = int(action)
        moved = move_without_spawn(state, action)
        if moved.moved:
            assert sample.legal_mask[i]
            assert not np.isnan(sample.teacher_value[i])
            assert np.array_equal(sample.afterstate[i], moved.afterstate)
        else:
            assert not sample.legal_mask[i]
            assert np.isnan(sample.teacher_value[i])
            assert sample.reward[i] == 0
            assert not sample.afterstate[i].any()

def test_game_level_split_is_deterministic_and_disjoint():
    ids = list(range(100))
    first = split_game_ids(ids)
    second = split_game_ids(ids)
    assert first == second
    assert len(first["train"]) == 80
    assert len(first["validation"]) == 10
    assert len(first["test"]) == 10
    assert not (first["train"] & first["validation"])
    assert not (first["train"] & first["test"])
    assert not (first["validation"] & first["test"])
    assert set.union(*first.values()) == set(ids)
