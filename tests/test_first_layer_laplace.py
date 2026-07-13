"""Tests for the first-layer (subnetwork) Laplace method.

Uses SmallCNN + synthetic tensors so no dataset download is required.
"""

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from bnn_medmnist.methods.first_layer_laplace import (
    FirstLayerLaplace,
    mark_bayesian_submodules,
)
from bnn_medmnist.models.small_cnn import SmallCNN


def test_mark_bayesian_submodules_pattern():
    model = SmallCNN(in_channels=1, num_classes=2)
    n = mark_bayesian_submodules(model, ["layer1.0"])
    conv = model.get_submodule("layer1.0")
    assert n == sum(p.numel() for p in conv.parameters())
    for name, p in model.named_parameters():
        assert p.requires_grad == name.startswith("layer1.0.")


def test_fit_and_reload_roundtrip(tmp_path):
    torch.manual_seed(0)
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 2, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    model = SmallCNN(in_channels=1, num_classes=2)
    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)  # stands in for a finished MAP run

    laplace_cfg = OmegaConf.create({
        "hessian_structure": "full",
        "prior_precision_method": 1.0,  # fixed prior: skip marglik for speed
        "n_predictive_samples": 5,
        "fit_batch_size": 16,
    })
    method = FirstLayerLaplace(
        laplace_cfg=laplace_cfg, ckpt_path=ckpt, device="cpu",
        bayesian_modules=["layer1.0"], map_checkpoint=ckpt,
    )
    method.fit(model, loader)

    probs = method.predict(x[:4])
    assert probs.shape == (4, 2)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-5)

    # Reload into a fresh model — must reproduce the identical GLM posterior.
    mu1, var1 = method.glm_logit_distribution(x[:4])
    model2 = SmallCNN(in_channels=1, num_classes=2)
    model2.load_state_dict(torch.load(ckpt))
    la2 = FirstLayerLaplace.load_laplace(model2, method.laplace_path, "cpu")
    mu2, var2 = la2._glm_predictive_distribution(x[:4], diagonal_output=True)
    assert torch.allclose(mu1, mu2, atol=1e-5)
    assert torch.allclose(var1, var2, atol=1e-5)
