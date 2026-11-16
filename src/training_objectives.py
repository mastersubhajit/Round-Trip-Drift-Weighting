"""Framework-agnostic differentiable surrogate objectives for robust SD training."""
import torch
import torch.nn.functional as F


def per_example_denoising_loss(pred, target):
    if pred.shape != target.shape:
        raise ValueError('Prediction and noise target have different shapes')
    return (pred.float()-target.float()).square().flatten(1).mean(1)


def contrastive_denoising_margin(pos_loss, neg_loss, margin=0.1):
    if pos_loss.shape != neg_loss.shape:
        raise ValueError('Correct/competing denoising errors must align')
    # Minimize: improve correct-caption fit relative to competing caption fit.
    return F.softplus(float(margin) - (neg_loss - pos_loss)).mean()
