"""Uncertainty (second-order) measures from a posterior predictive.

Most inputs are tensors of shape ``(n_samples, batch, num_classes)`` holding
posterior-sample softmax probabilities ``p(y | x, theta_s)``.

The classic information-theoretic decomposition (Depeweg et al. 2018; Kwon
et al. 2020) is

    H[E_q p(y|x,theta)]   =   E_q H[p(y|x,theta)]   +   I[y; theta | x]
    -------------------       ----------------------     ----------------
    total predictive          aleatoric (expected)       epistemic (BALD)

Beyond entropy/MI, the *choice* of summary statistic for the spread of the
posterior predictive matters: Hüllermeier & Waegeman (2021) point out that
Mutual Information is only one way to summarise second-order uncertainty and
need not capture everything the posterior encodes. This module therefore also
provides direct statistical spread measures (variance of the softmax outputs,
variance of the Gaussian logit posterior, quantile/inter-quantile ranges,
expected pairwise KL) so the evaluation can compare different *conceptual
families* of uncertainty scores rather than just entropy-based ones.

References
----------
* Hüllermeier, E. & Waegeman, W. (2021). "Aleatoric and Epistemic Uncertainty
  in Machine Learning: An Introduction to Concepts and Methods." Machine
  Learning 110(3), 457-506.
"""

from __future__ import annotations

import warnings

import torch

_EPS = 1e-12


def _entropy(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(p * p.clamp_min(_EPS).log()).sum(dim=dim)


def _aggregate_per_class(
    per_class: torch.Tensor, mean: torch.Tensor, aggregate: str
) -> torch.Tensor:
    """Reduce a per-class spread tensor ``(B, C)`` to a per-sample score ``(B,)``.

    ``aggregate="sum"`` sums the spread across classes; ``aggregate="max"``
    selects the spread of the *predicted* (argmax of ``mean``) class. ``mean``
    is the quantity whose argmax defines the predicted class (mean softmax for
    softmax variance; logit mean for logit variance).
    """
    if aggregate == "sum":
        return per_class.sum(dim=-1)
    if aggregate == "max":
        pred = mean.argmax(dim=-1, keepdim=True)  # (B, 1)
        return per_class.gather(-1, pred).squeeze(-1)
    raise ValueError(f"aggregate must be 'sum' or 'max', got {aggregate!r}")


def predictive_entropy(probs_samples: torch.Tensor) -> torch.Tensor:
    """Total predictive uncertainty: ``H[ E_q p(y|x,theta) ]``. Shape ``(B,)``."""
    mean = probs_samples.mean(dim=0)
    return _entropy(mean, dim=-1)


def expected_entropy(probs_samples: torch.Tensor) -> torch.Tensor:
    """Aleatoric proxy: ``E_q[ H[ p(y|x,theta) ] ]``. Shape ``(B,)``."""
    return _entropy(probs_samples, dim=-1).mean(dim=0)


def mutual_information(probs_samples: torch.Tensor) -> torch.Tensor:
    """Epistemic uncertainty (BALD): ``I[y; theta | x] = predictive - expected``.
    Shape ``(B,)``.
    """
    return predictive_entropy(probs_samples) - expected_entropy(probs_samples)


def predictive_variance(probs_samples: torch.Tensor) -> torch.Tensor:
    """Per-class variance of the predicted probabilities across MC samples.
    Shape ``(B, C)`` — caller can sum/mean if a scalar per example is wanted.
    """
    return probs_samples.var(dim=0, unbiased=False)


# ---------------------------------------------------------------------------
# second-order spread measures (Hüllermeier & Waegeman 2021)
# ---------------------------------------------------------------------------
def softmax_predictive_variance(
    probs_samples: torch.Tensor, aggregate: str = "sum"
) -> torch.Tensor:
    """Direct statistical spread of the softmax outputs across posterior samples.

    For each input ``x`` and class ``k`` this is the variance
    ``Var_w[softmax(f_w(x))_k]`` taken across the ``n_samples`` posterior draws.
    Unlike Mutual Information — an *information-theoretic* summary of the
    predictive — this is a *direct statistical spread measure*: the plain
    variance of the predicted probabilities. Hüllermeier & Waegeman (2021)
    stress that the choice of summary statistic matters, and a raw variance can
    capture aspects of the posterior spread that MI does not.

    Args:
        probs_samples: ``(n_samples, B, C)`` posterior-sample softmax probs.
        aggregate: ``"sum"`` sums the per-class variances → ``(B,)``; ``"max"``
            takes the variance of the predicted (argmax of mean) class → ``(B,)``.

    Returns:
        Per-sample score of shape ``(B,)``.
    """
    var = probs_samples.var(dim=0, unbiased=False)  # (B, C)
    mean = probs_samples.mean(dim=0)  # (B, C)
    return _aggregate_per_class(var, mean, aggregate)


def predictive_quantiles(probs_samples: torch.Tensor, qs: "list[float] | tuple[float, ...]") -> torch.Tensor:
    """Quantiles of the posterior-predictive softmax across MC samples.

    For each input ``x`` and class ``k``, computes the requested quantiles of
    ``softmax(f_w(x))_k`` over the ``n_samples`` posterior draws — a
    distribution-free description of the predictive spread that, unlike
    variance, is not dominated by outlier samples and directly yields credible
    intervals (e.g. ``qs=[0.05, 0.95]`` for a 90% interval).

    Args:
        probs_samples: ``(n_samples, B, C)`` posterior-sample softmax probs.
        qs: quantile levels in ``[0, 1]``, e.g. ``[0.05, 0.5, 0.95]``.

    Returns:
        Tensor of shape ``(len(qs), B, C)``, one quantile slice per level.
    """
    q_t = torch.as_tensor(list(qs), dtype=probs_samples.dtype, device=probs_samples.device)
    return torch.quantile(probs_samples, q_t, dim=0)


def softmax_predictive_quantile_range(
    probs_samples: torch.Tensor,
    lower: float = 0.05,
    upper: float = 0.95,
    aggregate: str = "sum",
) -> torch.Tensor:
    """Inter-quantile range of the softmax outputs across posterior samples.

    A robust, distribution-free analogue of :func:`softmax_predictive_variance`:
    instead of the second moment, this measures the width of the central
    ``upper - lower`` credible interval of ``softmax(f_w(x))_k`` across
    posterior draws — per Hüllermeier & Waegeman (2021), the choice of spread
    summary matters, and quantile ranges are not skewed by extreme samples the
    way variance is.

    Args:
        probs_samples: ``(n_samples, B, C)`` posterior-sample softmax probs.
        lower: lower quantile level (default 0.05).
        upper: upper quantile level (default 0.95).
        aggregate: ``"sum"`` sums the per-class ranges → ``(B,)``; ``"max"``
            takes the range of the predicted (argmax of mean) class → ``(B,)``.

    Returns:
        Per-sample score of shape ``(B,)``.
    """
    lo, hi = predictive_quantiles(probs_samples, [lower, upper])
    spread = hi - lo  # (B, C)
    mean = probs_samples.mean(dim=0)  # (B, C)
    return _aggregate_per_class(spread, mean, aggregate)


def logit_variance(
    logits, aggregate: str = "sum"
) -> torch.Tensor:
    """Variance of the (Gaussian) posterior over the logits — analytical spread.

    For Laplace methods ``laplace-torch`` can return a Gaussian over the logits
    directly (``pred_type="glm"``) without MC sampling. The variance of that
    Gaussian — induced by the Gaussian posterior over the last-layer weights —
    is a *sampling-free* uncertainty measure and the most direct "variance of a
    Gaussian" interpretation of second-order uncertainty.

    Accepts either input form, auto-detected:
        * ``logits`` a tensor ``(n_samples, B, C)`` — MC logit samples; the
          per-class variance is estimated empirically across samples.
        * ``logits`` a tuple ``(logit_mean, logit_var)`` of ``(B, C)`` each —
          the analytical Gaussian moments. ``logit_var`` may be ``None`` (e.g.
          a deterministic model with no posterior over weights), in which case a
          zero score of shape ``(B,)`` is returned.

    Args:
        logits: see above.
        aggregate: ``"sum"`` sums per-class logit variances → ``(B,)``; ``"max"``
            takes the variance of the predicted (argmax of mean) logit → ``(B,)``.

    Returns:
        Per-sample score of shape ``(B,)``.
    """
    if isinstance(logits, (tuple, list)):
        mean, var = logits
        if var is None:  # deterministic model: no posterior spread to report
            return torch.zeros(mean.shape[0], dtype=mean.dtype, device=mean.device)
        mean = torch.as_tensor(mean)
        var = torch.as_tensor(var)
    else:  # MC logit samples (n_samples, B, C)
        samples = torch.as_tensor(logits)
        var = samples.var(dim=0, unbiased=False)
        mean = samples.mean(dim=0)
    return _aggregate_per_class(var, mean, aggregate)


def expected_pairwise_kl(
    probs_samples: torch.Tensor, min_samples: int = 10
) -> torch.Tensor | None:
    """Average pairwise KL-divergence between the predictive samples.

    For each input, averages ``KL(p_i || p_j)`` over ordered pairs of posterior
    predictive distributions ``p_i = softmax(f_{w_i}(x))``. Captures how much
    the per-sample predictives *disagree* on average — another second-order
    spread measure distinct from entropy/variance.

    Only meaningful with enough samples: if ``n_samples < min_samples`` this
    returns ``None`` (and warns) rather than a noisy estimate.

    Args:
        probs_samples: ``(n_samples, B, C)`` posterior-sample softmax probs.
        min_samples: minimum number of samples required (default 10).

    Returns:
        Per-sample score ``(B,)`` or ``None`` if too few samples.
    """
    n = probs_samples.shape[0]
    if n < min_samples:
        warnings.warn(
            f"expected_pairwise_kl needs >= {min_samples} samples, got {n}; "
            f"returning None.",
            stacklevel=2,
        )
        return None

    log_p = probs_samples.clamp_min(_EPS).log()  # (S, B, C)
    # KL(p_i || p_j) = sum_k p_i,k (log p_i,k - log p_j,k)
    #               = sum_k p_i,k log p_i,k  -  sum_k p_i,k log p_j,k
    self_term = (probs_samples * log_p).sum(dim=-1)  # (S, B): -H[p_i]
    # cross[i, j, b] = sum_k p_i,k log p_j,k
    cross = torch.einsum("ibk,jbk->ijb", probs_samples, log_p)  # (S, S, B)
    kl = self_term.unsqueeze(1) - cross  # (S, S, B): KL(p_i || p_j)
    # average over ordered off-diagonal pairs (diagonal is KL(p||p)=0)
    return kl.sum(dim=(0, 1)) / (n * (n - 1))
