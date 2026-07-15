"""Tests for the Cloud-Adaptive Regularized Loss."""

from __future__ import annotations

import torch

from nadir.losses.carl import CARLLoss


def test_all_clear_reduces_to_input_plus_reg_term() -> None:
    # mask == 0 everywhere: loss = mean|pred - cloudy| + lambda * mean|pred - target|
    loss_fn = CARLLoss(lambda_reg=1.0)
    pred = torch.full((1, 3, 2, 2), 0.5)
    cloudy = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.zeros(1, 2, 2)
    loss = loss_fn(pred, cloudy, target, mask)
    torch.testing.assert_close(loss, torch.tensor(0.5 + 0.5))


def test_all_cloudy_reduces_to_weighted_target_l1() -> None:
    # mask == 1 everywhere: loss = (1 + lambda) * mean|pred - target|
    loss_fn = CARLLoss(lambda_reg=1.0)
    pred = torch.zeros(1, 3, 2, 2)
    cloudy = torch.full((1, 3, 2, 2), 123.0)  # must be ignored
    target = torch.ones(1, 3, 2, 2)
    mask = torch.ones(1, 2, 2)
    loss = loss_fn(pred, cloudy, target, mask)
    torch.testing.assert_close(loss, torch.tensor(2.0))


def test_mixed_mask_hand_computed() -> None:
    # 1x1x1x2 image: pixel0 clear, pixel1 cloudy.
    loss_fn = CARLLoss(lambda_reg=1.0)
    pred = torch.tensor([[[[0.0, 0.0]]]])
    cloudy = torch.tensor([[[[0.2, 9.0]]]])
    target = torch.tensor([[[[1.0, 0.6]]]])
    mask = torch.tensor([[[0.0, 1.0]]])
    # adaptive = mean([|0-0.2|, |0-0.6|]) = 0.4 ; reg = mean([1.0, 0.6]) = 0.8
    loss = loss_fn(pred, cloudy, target, mask)
    torch.testing.assert_close(loss, torch.tensor(1.2))


def test_perfect_behavior_yields_zero_adaptive_loss() -> None:
    # Prediction preserves clear pixels and reconstructs cloudy ones: with
    # lambda_reg=0 the loss must be exactly zero.
    loss_fn = CARLLoss(lambda_reg=0.0)
    cloudy = torch.tensor([[[[0.3, 5.0]]]])
    target = torch.tensor([[[[0.1, 0.7]]]])
    mask = torch.tensor([[[0.0, 1.0]]])
    pred = torch.tensor([[[[0.3, 0.7]]]])  # clear pixel from input, cloudy from target
    torch.testing.assert_close(loss_fn(pred, cloudy, target, mask), torch.tensor(0.0))


def test_multiclass_mask_binarized() -> None:
    # 3-class masks (thin=1, thick=2, shadow=3) must all count as "cloudy".
    loss_fn = CARLLoss(lambda_reg=0.0)
    pred = torch.zeros(1, 1, 1, 3)
    cloudy = torch.full((1, 1, 1, 3), 9.0)
    target = torch.tensor([[[[0.3, 0.6, 0.9]]]])
    mask = torch.tensor([[[1.0, 2.0, 3.0]]])
    loss = loss_fn(pred, cloudy, target, mask)
    torch.testing.assert_close(loss, torch.tensor(0.6))  # mean(|0-.3|,|0-.6|,|0-.9|)
