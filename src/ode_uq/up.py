"""Uncertainty propagation tools."""
from jax import vmap
import jax.random as jr
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray, ArrayLike
from typing import Optional
from dataclasses import dataclass

from .model import DynamicalSystem, OutputSystem
from .utils import sample_from_dict_of_distributions, inverse_cdf_from_dict_of_distributions

@dataclass
class SimulationResults:
    times: Optional[Array] = None
    params: Optional[dict] = None
    init_state: Optional[dict] = None
    states: Optional[dict] = None
    pointwise_outputs: Optional[dict] = None
    functional_outputs: Optional[dict] = None

def evaluate_batch(
    dynamical_system: DynamicalSystem,
    param_samples: dict,
    init_state_samples: dict,
    output_system: Optional[OutputSystem] = None
) -> SimulationResults:
    """Evaluate a batch of parameter and initial state samples through the dynamical system.

    Args:
        dynamical_system: A DynamicalSystem instance.
        param_samples: A dictionary of parameter samples.
        init_state_samples: A dictionary of initial state samples.
        output_system: Optional OutputSystem for computing additional outputs.

    Returns:
        A SimulationResults object containing all sampling results.
    """
    state_samples = vmap(lambda *args: dynamical_system.simulate(*args).ys)(param_samples, init_state_samples)
    if output_system is not None:
        pointwise_outputs = vmap(output_system.compute_pointwise_outputs, (None, 0, None))(dynamical_system.times, state_samples, param_samples)
        functional_outputs = vmap(output_system.compute_functional_outputs, (None, 0, None))(dynamical_system.times, state_samples, param_samples)
    else:
        pointwise_outputs = None
        functional_outputs = None

    return SimulationResults(
        times=dynamical_system.times,
        params=param_samples,
        init_state=init_state_samples, 
        states=state_samples,
        pointwise_outputs=pointwise_outputs,
        functional_outputs=functional_outputs
    )

def run_monte_carlo_sampling(
    dynamical_system: DynamicalSystem, 
    rng_key: PRNGKeyArray,
    num_samples: int,
    output_system: Optional[OutputSystem] = None
) -> SimulationResults:
    """Perform Monte Carlo sampling for uncertainty propagation.

    Args:
        dynamical_system: A DynamicalSystem instance.
        rng_key: A JAX random key.
        num_samples: Number of samples to draw.
        output_system: Optional OutputSystem for computing additional outputs.

    Returns:
        A SimulationResults object containing all sampling results.
    """
    key_param, key_init_state = jr.split(rng_key)
    param_samples = sample_from_dict_of_distributions(dynamical_system.param_dists, key_param, num_samples)
    init_state_samples = sample_from_dict_of_distributions(dynamical_system.init_state_dists, key_init_state, num_samples)
    return evaluate_batch(dynamical_system, param_samples, init_state_samples, output_system)

def run_inverse_transform_sampling(
    dynamical_system: DynamicalSystem, 
    sampled_quantiles: dict[ArrayLike],
    output_system: Optional[OutputSystem] = None
) -> SimulationResults:
    """Perform inverse transform sampling for uncertainty propagation.

    Args:
        dynamical_system: A DynamicalSystem instance.
        sampled_quantiles: A dictionary of sampled quantiles (ranging from 0 to 1) for each random parameter and initial state variable.
        output_system: Optional OutputSystem for computing additional outputs.

    Returns:
        A SimulationResults object containing all sampling results.
    """
    param_samples = inverse_cdf_from_dict_of_distributions(sampled_quantiles, dynamical_system.param_dists)
    init_state_samples = inverse_cdf_from_dict_of_distributions(sampled_quantiles, dynamical_system.init_state_dists)
    return evaluate_batch(dynamical_system, param_samples, init_state_samples, output_system)