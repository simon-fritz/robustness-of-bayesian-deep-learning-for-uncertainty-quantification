"""Abstract base class for Bayesian inference methods.

Every concrete method implements:

    fit(model, train_loader)
        Estimate (or train) the posterior over the chosen ``bayesian_layers``.

    predict(x, n_samples)
        Return predictive distribution over classes — typically an MC estimate
        ``E_{theta ~ q}[ softmax(f_theta(x)) ]`` of shape ``(batch, num_classes)``.

The ``bayesian_layers`` axis controls *Bayesian extent*:
    - ``["fc"]``                — last layer only (e.g. last-layer Laplace)
    - ``["layer4", "fc"]``      — last N layers
    - ``"all"``                 — full network
    - ``[]``                    — deterministic (MAP) baseline

Concrete methods are responsible for translating these names into a set of
parameters that receive a posterior, leaving all other parameters at their MAP
values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Type alias for the Bayesian-extent parameter.
BayesianLayers = Union[list[str], str]  # list of names, or the literal "all"


class BayesianMethod(ABC):
    """Abstract Bayesian inference method."""

    def __init__(self, bayesian_layers: BayesianLayers, n_samples: int = 1) -> None:
        """
        Args:
            bayesian_layers: Layer names that receive a posterior, or "all".
                Examples: ``["fc"]``, ``["layer4", "fc"]``, ``"all"``, ``[]``.
            n_samples: Default number of MC samples drawn at predict time.
        """
        self.bayesian_layers = bayesian_layers
        self.n_samples = n_samples

    @abstractmethod
    def fit(self, model: nn.Module, train_loader: DataLoader) -> None:
        """Fit the posterior (or train ensemble members, etc.).

        After ``fit``, the method holds whatever state is needed by ``predict``
        (Laplace approximation, dropout flags, ensemble members, ...).
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: torch.Tensor, n_samples: int | None = None) -> torch.Tensor:
        """Return predictive class probabilities of shape ``(batch, num_classes)``.

        Args:
            x: Input batch.
            n_samples: Override for the number of MC samples (defaults to
                ``self.n_samples``).
        """
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------

    def _resolve_bayesian_params(self, model: nn.Module) -> Iterable[nn.Parameter]:
        """Resolve ``self.bayesian_layers`` into an iterable of parameters.

        Sketch of the contract — concrete subclasses may override:
            - ``"all"``  -> all parameters
            - ``[]``     -> empty
            - list[str]  -> parameters of the named submodules
        """
        # TODO: implement
        raise NotImplementedError
