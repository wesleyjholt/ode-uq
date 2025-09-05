import yaml
import os
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import dataclass
from typing import Optional
from jaxtyping import Array
import numpyro.distributions as dist

def load_yaml(filepath):
    """Load a YAML file and return its contents as a dictionary"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"YAML file not found: {filepath}")
    
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    
    return data

class FlattenedDict(eqx.Module):
    """A class to handle conversions between dictionaries and arrays."""
    data: Array
    relevant_keys: list
    keys: list
    
    def to_dict(self):
        d = {}
        i = 0
        for key in self.keys:
            if key in self.relevant_keys:
                d[key] = self.data[i]
                i += 1
            else:
                d[key] = None
        return d
    
    def to_partitioned_dicts(self, key_groups: list[list]) -> tuple[dict]:
        dicts = []
        for group in key_groups:
            d = {}
            i = 0
            for key in group:
                if key in self.relevant_keys:
                    d[key] = self.data[i]
                    i += 1
                else:
                    d[key] = None
            dicts.append(d)
        return tuple(dicts)

    @classmethod
    def from_dict(cls, d, keys):
        data = []
        relevant_keys = []
        for key in keys:
            if key in d:
                data.append(d[key])
                relevant_keys.append(key)
        data = jnp.array(data)
        return cls(data=data, relevant_keys=relevant_keys, keys=keys)
    
    def concat(self, other: 'FlattenedDict') -> 'FlattenedDict':
        for key in self.keys:
            if key in other.keys:
                raise ValueError(f"Key '{key}' found in both FlattenedDict instances. Cannot concatenate.")
        new_data = jnp.concatenate([self.data, other.data])
        new_keys = self.keys + other.keys
        new_relevant_keys = self.relevant_keys + other.relevant_keys
        return FlattenedDict(data=new_data, relevant_keys=new_relevant_keys, keys=new_keys)
    
    def update_data(self, new_data: Array) -> 'FlattenedDict':
        if new_data.shape != self.data.shape:
            raise ValueError(f"New data shape {new_data.shape} does not match existing data shape {self.data.shape}.")
        return FlattenedDict(data=new_data, relevant_keys=self.relevant_keys, keys=self.keys)
        

def flatten_dict(d: dict, keys: list, missing_key_okay: bool = True) -> FlattenedDict:
    """Convenience function to create a FlattenedDict from a regular dict."""
    if not missing_key_okay:
        for key in keys:
            if key not in d:
                raise KeyError(f"Key '{key}' not found in the input dictionary.")
    return FlattenedDict.from_dict(d, keys)

def unflatten_array(arr: Array, relevant_keys: list, all_keys: Optional[list] = None, ) -> dict:
    """Convenience function to convert an array back into a dictionary."""
    if all_keys is None:
        all_keys = relevant_keys
    fd = FlattenedDict(data=arr, relevant_keys=relevant_keys, keys=all_keys)
    return fd.to_dict()


def load_yaml_of_distributions(filepath):
    """
    Load parameter distributions from YAML file and create NumPyro distributions.
    
    This function reads a YAML file containing parameter distribution specifications
    and creates a dictionary of initialized NumPyro distribution objects. It also
    supports deterministic values (fixed parameters) specified as single values.
    
    Args:
        filepath (str): Path to the parameter distribution YAML file
        
    Returns:
        dict: Dictionary mapping parameter names to NumPyro distribution instances
              or deterministic values (for fixed parameters)
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If an unsupported distribution type is specified
        KeyError: If required keys are missing from parameter specification
        
    Example:
        >>> distributions = load_yaml_of_distributions('parameter_dist.yaml')
        >>> print(len(distributions))
        22
        
    YAML Format:
        # Probabilistic parameter
        k1:
          dist_name: "LogNormal"
          dist_params:
            loc: -6.91
            scale: 0.5
            
        # Deterministic parameter (fixed value)
        k2: 0.001
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parameter distribution file not found: {filepath}")
    
    with open(filepath, 'r') as file:
        param_specs = yaml.safe_load(file)
    
    distributions = {}
    
    # Mapping of distribution names to NumPyro distribution classes
    dist_mapping = {
        'normal': dist.Normal,
        'lognormal': dist.LogNormal,
        'uniform': dist.Uniform,
        'beta': dist.Beta,
        'gamma': dist.Gamma,
        'exponential': dist.Exponential,
        'student_t': dist.StudentT,
        'cauchy': dist.Cauchy,
        'half_normal': dist.HalfNormal,
        'half_cauchy': dist.HalfCauchy,
        'inverse_gamma': dist.InverseGamma,
        'pareto': dist.Pareto,
        'weibull': dist.Weibull,
        'laplace': dist.Laplace,
        'gumbel': dist.Gumbel,
        'truncated_normal': dist.TruncatedNormal,
    }
    
    for param_key, param_info in param_specs.items():
        param_name = param_key
        
        # Check if this is a deterministic value (single number)
        if isinstance(param_info, (int, float)):
            # Store deterministic value directly
            distributions[param_name] = float(param_info)
            continue
            
        # Otherwise, process as a distribution specification
        try:
            # Extract required fields
            dist_name = param_info['dist_name']
            dist_params = param_info['dist_params']
            
            # Check if distribution type is supported
            if dist_name not in dist_mapping:
                raise ValueError(f"Unsupported distribution type: {dist_name}. "
                               f"Supported types: {list(dist_mapping.keys())}")
            
            # Get the distribution class
            dist_class = dist_mapping[dist_name]
            
            # Create distribution instance with specified parameters
            # Convert parameter names to match NumPyro convention
            numpyro_params = {}
            for param_name_key, param_value in dist_params.items():
                numpyro_params[param_name_key] = param_value
            
            # Create the distribution instance
            distribution = dist_class(**numpyro_params)
            distributions[param_name] = distribution
            
        except KeyError as e:
            raise KeyError(f"Missing required key in parameter '{param_key}': {e}")
        except TypeError as e:
            # Get the distribution name if it was defined
            dist_name_for_error = locals().get('dist_name', 'unknown')
            raise ValueError(f"Invalid parameters for distribution '{dist_name_for_error}' "
                           f"in parameter '{param_key}': {e}")
    
    return distributions


def sample_from_dict_of_distributions(distributions, key, num_samples=1):
    """
    Sample from parameter distributions.
    
    This function generates random samples from a dictionary of NumPyro distributions
    using JAX random keys for reproducible sampling. For deterministic values (fixed
    parameters), it returns the deterministic value directly.
    
    Args:
        distributions (dict): Dictionary of NumPyro distributions or deterministic values
        key: JAX random key for reproducible sampling
        num_samples (int): Number of samples to generate (default: 1)
        
    Returns:
        dict: Dictionary mapping parameter names to sampled values or deterministic values
        
    Example:
        >>> import jax
        >>> key = jax.random.PRNGKey(42)
        >>> samples = sample_from_dict_of_distributions(distributions, key)
        >>> print(samples['k1'])
        0.001036
    """
    samples = {}
    
    # Count how many distributions need random keys
    dist_count = sum(1 for val in distributions.values() if not isinstance(val, (int, float)))
    
    if dist_count > 0:
        # Split the key for each distribution (not deterministic values)
        keys = jax.random.split(key, dist_count)
        key_idx = 0
    
    for param_name, distribution in distributions.items():
        # Check if this is a deterministic value
        if isinstance(distribution, (int, float)):
            # Return deterministic value directly
            if num_samples == 1:
                samples[param_name] = jnp.array(distribution)
            else:
                samples[param_name] = jnp.full((num_samples,), distribution)
        else:
            # Sample from the distribution
            if num_samples == 1:
                samples[param_name] = distribution.sample(keys[key_idx])
            else:
                samples[param_name] = distribution.sample(keys[key_idx], (num_samples,))
            key_idx += 1
    
    return samples

def inverse_cdf_from_dict_of_distributions(quantiles, distributions):
    """
    Compute the inverse CDF (quantile function) for each distribution at the given quantiles.
    
    For deterministic values (fixed parameters), returns the deterministic value directly
    regardless of the quantile input.

    Args:
        quantiles (dict): Quantiles at which to evaluate the inverse CDF, for each random variable
        distributions (dict): Dictionary of NumPyro distributions or deterministic values

    Returns:
        dict: Dictionary mapping parameter names to inverse CDF values or deterministic values
    """
    num_samples = next(iter(quantiles.values())).shape[0]
    inverse_cdfs = {}
    for param_name, distribution in distributions.items():
        # Check if this is a deterministic value
        if isinstance(distribution, (int, float)):
            # Return deterministic value directly (ignore quantile)
            inverse_cdfs[param_name] = jnp.full((num_samples,), distribution)
        else:
            # Compute inverse CDF for the distribution
            inverse_cdfs[param_name] = distribution.icdf(quantiles[param_name])
    return inverse_cdfs
