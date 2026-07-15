"""Tests for the DSen2-CR model."""

from __future__ import annotations

import torch

from nadir.models.dsen2cr import DSen2CR, paste_clear_pixels


def _tiny_model(**kwargs: object) -> DSen2CR:
    return DSen2CR(features=8, num_blocks=2, **kwargs)  # type: ignore[arg-type]


def test_forward_shape_and_dtype() -> None:
    model = _tiny_model()
    s2 = torch.rand(2, 13, 32, 32) * 2 - 1
    s1 = torch.rand(2, 2, 32, 32) * 2 - 1
    out = model(s2, s1)
    assert out.shape == (2, 13, 32, 32)
    assert out.dtype == torch.float32


def test_long_skip_is_optical_only() -> None:
    # With the tail conv zeroed, the network must return the optical input
    # exactly — proving the long skip adds s2_cloudy, not the SAR channels.
    model = _tiny_model()
    torch.nn.init.zeros_(model.tail.weight)
    torch.nn.init.zeros_(model.tail.bias)
    s2 = torch.rand(1, 13, 16, 16)
    s1 = torch.rand(1, 2, 16, 16)
    torch.testing.assert_close(model(s2, s1), s2)


def test_gradients_flow_to_all_parameters() -> None:
    model = _tiny_model()
    out = model(torch.rand(1, 13, 16, 16), torch.rand(1, 2, 16, 16))
    out.mean().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"


def test_grad_checkpointing_matches() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    model_ckpt = _tiny_model(grad_checkpointing=True)
    model_ckpt.load_state_dict(model.state_dict())
    model.train(), model_ckpt.train()
    s2, s1 = torch.rand(1, 13, 16, 16), torch.rand(1, 2, 16, 16)
    torch.testing.assert_close(model(s2, s1), model_ckpt(s2, s1))


def test_paste_clear_pixels() -> None:
    pred = torch.zeros(1, 13, 4, 4)
    cloudy = torch.ones(1, 13, 4, 4)
    mask = torch.zeros(1, 4, 4)
    mask[:, :2] = 1  # top half cloudy
    out = paste_clear_pixels(pred, cloudy, mask)
    assert (out[:, :, :2] == 0).all()  # masked area: prediction
    assert (out[:, :, 2:] == 1).all()  # clear area: original preserved
