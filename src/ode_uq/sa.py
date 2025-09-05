from jax import vmap
import jax.random as jr
import equinox as eqx
import numpy as np
from jaxtyping import Array, PRNGKeyArray, ArrayLike
from typing import Optional
from dataclasses import dataclass
from functools import partial

from .model import DynamicalSystem, OutputSystem
from .up import SimulationResults, run_inverse_transform_sampling
from .utils import unflatten_array

from SALib.analyze.sobol import analyze as sobol_analyze
from SALib.sample.sobol import sample as sobol_sample

@dataclass
class SobolSensitivityResults:
    first_order_indices: SimulationResults
    total_order_indices: SimulationResults

num_samples = 2**3

def make_sobol_problem(dynamical_system: DynamicalSystem):
    unit_hypercube_bounds = [[0.0, 1.0] for _ in range(dynamical_system.num_random_inputs)]

    problem = {
        'num_vars': dynamical_system.num_random_inputs,
        'names': dynamical_system.random_input_names,
        'bounds': unit_hypercube_bounds
    }
    return problem

def run_sobol_sampling(
    dynamical_system: DynamicalSystem, 
    log2_num_samples: int,
    output_system: Optional[OutputSystem] = None
):
    """Perform Sobol sampling for uncertainty propagation.

    Args:
        dynamical_system: A DynamicalSystem instance.
        num_samples: A dictionary specifying the number of samples to draw for each random parameter and initial state variable.
        output_system: Optional OutputSystem for computing additional outputs.

    Returns:
        A SimulationResults object containing all sampling results.
    """
    num_samples = 2**log2_num_samples
    problem = make_sobol_problem(dynamical_system)

    sobol_samples_array = sobol_sample(problem, num_samples)
    print('Number of Sobol samples to generate:', sobol_samples_array.shape[0])
    sobol_samples = {name: sobol_samples_array[:, i] for i, name in enumerate(dynamical_system.random_input_names)}
    return run_inverse_transform_sampling(dynamical_system, sobol_samples, output_system)

def analyze_sobol_results(
    sobol_sampling_results: SimulationResults,
    dynamical_system: DynamicalSystem,
):
    """Analyze Sobol sampling results to compute Sobol sensitivity indices."""
    problem = make_sobol_problem(dynamical_system)

    def _sobol_analyze_scalar_output(output_samples):
        si = sobol_analyze(problem, output_samples, print_to_console=False)
        return si

    def _sobol_analyze_vector_output(output_samples):
        sobol_indices = []
        for i in range(output_samples.shape[1]):
            sobol_indices.append(_sobol_analyze_scalar_output(output_samples[:, i]))
        return sobol_indices
    
    def sobol_analyze_dict_of_scalar_outputs(output_samples_dict, sobol_index_names):
        sobol_analysis = {k: _sobol_analyze_scalar_output(v) for k, v in output_samples_dict.items()}
        return {sobol_index_name: {output_name: sobol_analysis[output_name][sobol_index_name] for output_name in output_samples_dict.keys()} for sobol_index_name in sobol_index_names}

    def sobol_analyze_dict_of_vector_outputs(output_samples_dict, sobol_index_names):
        sobol_analysis = {k: _sobol_analyze_vector_output(v) for k, v in output_samples_dict.items()}
        return {sobol_index_name: {output_name: np.array(list(map(lambda x: x[sobol_index_name], sobol_analysis[output_name]))) for output_name in output_samples_dict.keys()} for sobol_index_name in sobol_index_names}

    sobol_index_names = ['S1', 'ST']
    si_states = sobol_analyze_dict_of_vector_outputs(sobol_sampling_results.states, sobol_index_names)
    si_pointwise = sobol_analyze_dict_of_vector_outputs(sobol_sampling_results.pointwise_outputs, sobol_index_names) if sobol_sampling_results.pointwise_outputs is not None else None
    si_functional = sobol_analyze_dict_of_scalar_outputs(sobol_sampling_results.functional_outputs, sobol_index_names) if sobol_sampling_results.functional_outputs is not None else None

    unflatten_random_inputs = partial(unflatten_array, relevant_keys=dynamical_system.random_input_names, all_keys=dynamical_system.all_names)
    
    def construct_inputs_sim_results_vectorized(arr):
        inputs = eqx.filter_vmap(unflatten_random_inputs)(arr)
        return _construct_inputs_sim_results(inputs)
    
    def construct_inputs_sim_results(val):
        inputs = unflatten_random_inputs(val)
        return _construct_inputs_sim_results(inputs)

    def _construct_inputs_sim_results(inputs):
        params  = {k: v for k, v in inputs.items() if k in dynamical_system.all_param_names}
        init_state = {k: v for k, v in inputs.items() if k in dynamical_system.all_state_names}
        return SimulationResults(
            params=params,
            init_state=init_state,
        )
    
    si_states = {sobol_index_name: {output_name: construct_inputs_sim_results_vectorized(output_samples) for output_name, output_samples in output_dict.items()} for sobol_index_name, output_dict in si_states.items()}
    si_pointwise = {sobol_index_name: {output_name: construct_inputs_sim_results_vectorized(output_samples) for output_name, output_samples in output_dict.items()} for sobol_index_name, output_dict in si_pointwise.items()} if si_pointwise is not None else None
    si_functional = {sobol_index_name: {output_name: construct_inputs_sim_results(output_samples) for output_name, output_samples in output_dict.items()} for sobol_index_name, output_dict in si_functional.items()} if si_functional is not None else None

    res = {sobol_index_name: SimulationResults(states=si_states[sobol_index_name], pointwise_outputs=si_pointwise[sobol_index_name], functional_outputs=si_functional[sobol_index_name]) for sobol_index_name in sobol_index_names}
    return res

