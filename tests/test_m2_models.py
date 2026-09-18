import pytest
import torch

from game2048.m2_models import (
    ResidualMLP2048,
    Transformer2048,
    tokenize_boards,
    trainable_parameter_count,
)


@pytest.mark.parametrize("model_cls", [Transformer2048, ResidualMLP2048])
def test_model_input_contract(model_cls):
    model = model_cls()
    with pytest.raises(TypeError):
        model(torch.zeros((2, 16), dtype=torch.int64))
    with pytest.raises(ValueError):
        model(torch.zeros((16,), dtype=torch.uint8))
    with pytest.raises(ValueError):
        model(torch.zeros((2, 4, 4), dtype=torch.uint8))
    with pytest.raises(ValueError):
        model(torch.zeros((0, 16), dtype=torch.uint8))


def test_overflow_token_mapping():
    values = torch.tensor(
        [[0, 1, 20, 21, 22, 100, 255, 5, 6, 7, 8, 9, 10, 11, 12, 13]],
        dtype=torch.uint8,
    )
    tokens = tokenize_boards(values)
    assert tokens.dtype == torch.long
    assert tokens[0, :7].tolist() == [0, 1, 20, 21, 21, 21, 21]
    assert values[0, 4:7].tolist() == [22, 100, 255]


@pytest.mark.parametrize("model_cls", [Transformer2048, ResidualMLP2048])
@pytest.mark.parametrize("batch", [1, 5])
def test_q_v_a_shapes_and_forward_identity(model_cls, batch):
    torch.manual_seed(1)
    model = model_cls()
    boards = torch.randint(0, 256, (batch, 16), dtype=torch.uint8)
    q = model(boards)
    q_state, value = model.forward_state(boards)
    after = model.forward_afterstate(boards)
    assert q.shape == (batch, 4)
    assert value.shape == (batch,)
    assert after.shape == (batch,)
    assert torch.equal(q, q_state)
    assert torch.isfinite(q).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(after).all()
@pytest.mark.parametrize("model_cls", [Transformer2048, ResidualMLP2048])
def test_all_heads_and_backbone_receive_gradient(model_cls):
    torch.manual_seed(2)
    model = model_cls()
    boards = torch.randint(0, 22, (4, 16), dtype=torch.uint8)
    afterstates = torch.randint(0, 22, (4, 16), dtype=torch.uint8)
    q, value = model.forward_state(boards)
    after = model.forward_afterstate(afterstates)
    loss = q.square().mean() + value.square().mean() + after.square().mean()
    loss.backward()

    assert model.tile_embedding.weight.grad is not None
    assert torch.isfinite(model.tile_embedding.weight.grad).all()
    assert model.q_head.weight.grad is not None
    assert model.value_head.weight.grad is not None
    assert model.afterstate_head.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters())


def test_transformer_position_embedding_exists_and_receives_gradient():
    torch.manual_seed(3)
    model = Transformer2048()
    assert tuple(model.position_embedding.shape) == (1, 16, 256)
    assert model.position_embedding.requires_grad
    boards = torch.randint(0, 22, (3, 16), dtype=torch.uint8)
    model(boards).square().mean().backward()
    assert model.position_embedding.grad is not None
    assert torch.isfinite(model.position_embedding.grad).all()


def test_parameter_count_gate():
    transformer = trainable_parameter_count(Transformer2048())
    mlp = trainable_parameter_count(ResidualMLP2048())
    ratio = abs(transformer - mlp) / transformer
    assert 4_000_000 <= transformer <= 6_000_000
    assert 4_000_000 <= mlp <= 6_000_000
    assert ratio <= 0.15
