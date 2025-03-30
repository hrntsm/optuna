from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import FloatDistribution
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers.nsgaii._constraints_evaluation import _constrained_dominates
from optuna.samplers.nsgaii._crossover import perform_crossover
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.study._multi_objective import _dominates
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    from optuna.study import Study


class NSGAIIChildGenerationStrategy:
    def __init__(
        self,
        *,
        mutation_prob: float | None = None,
        crossover: BaseCrossover,
        crossover_prob: float,
        swapping_prob: float,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        rng: LazyRandomState,
    ) -> None:
        if not (mutation_prob is None or 0.0 <= mutation_prob <= 1.0):
            raise ValueError(
                "`mutation_prob` must be None or a float value within the range [0.0, 1.0]."
            )

        if not (0.0 <= crossover_prob <= 1.0):
            raise ValueError("`crossover_prob` must be a float value within the range [0.0, 1.0].")

        if not (0.0 <= swapping_prob <= 1.0):
            raise ValueError("`swapping_prob` must be a float value within the range [0.0, 1.0].")

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._swapping_prob = swapping_prob
        self._crossover = crossover
        self._constraints_func = constraints_func
        self._rng = rng

    def _perform_mutation(self, search_space: BaseDistribution, value: float) -> Any:
        if isinstance(search_space, FloatDistribution):
            lb = search_space.low
            ub = search_space.high

            # return self.polynomial_mutation(value, lb, ub, eta=1000)
            return self.gaussian_mutation(value, lb, ub, sigma_factor=1 / 500.0)
            # return self.uniform_mutation(value, lb, ub)
        else:
            raise NotImplementedError

    def uniform_mutation(self, value: float, lb: float, ub: float) -> float:
        child_param = np.random.rand()
        return (ub - lb) * child_param + lb

    def polynomial_mutation(self, value: float, lb: float, ub: float, eta: float) -> float:
        u = self._rng.rng.rand()

        if u <= 0.5:
            delta_l = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            child_param = value + delta_l * (value - lb)
        else:
            delta_r = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            child_param = value + delta_r * (ub - value)

        return np.clip(child_param, lb, ub)

    def gaussian_mutation(self, value: float, lb: float, ub: float, sigma_factor: float) -> float:
        sigma = sigma_factor * (ub - lb)
        child_param = np.random.normal(value, sigma)

        return np.clip(child_param, lb, ub)

    def __call__(
        self,
        study: Study,
        search_space: dict[str, BaseDistribution],
        parent_population: list[FrozenTrial],
    ) -> dict[str, Any]:
        """Generate a child parameter from the given parent population by NSGA-II algorithm.
        Args:
            study:
                Target study object.
            search_space:
                A dictionary containing the parameter names and parameter's distributions.
            parent_population:
                A list of trials that are selected as parent population.
        Returns:
            A dictionary containing the parameter names and parameter's values.
        """
        dominates = _dominates if self._constraints_func is None else _constrained_dominates
        # We choose a child based on the specified crossover method.
        if self._rng.rng.rand() < self._crossover_prob:
            child_params = perform_crossover(
                self._crossover,
                study,
                parent_population,
                search_space,
                self._rng.rng,
                self._swapping_prob,
                dominates,
            )
        else:
            parent_population_size = len(parent_population)
            parent_params = parent_population[self._rng.rng.choice(parent_population_size)].params
            child_params = {name: parent_params[name] for name in search_space.keys()}

        n_params = len(child_params)
        if self._mutation_prob is None:
            mutation_prob = 1.0 / max(1.0, n_params)
        else:
            mutation_prob = self._mutation_prob

        params = {}
        for param_name in child_params.keys():
            if self._rng.rng.rand() >= mutation_prob:
                params[param_name] = child_params[param_name]
            else:
                params[param_name] = self._perform_mutation(
                    search_space[param_name], child_params[param_name]
                )
        params.update()
        return params
