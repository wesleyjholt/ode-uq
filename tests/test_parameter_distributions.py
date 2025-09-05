"""
Tests for parameter distribution utilities.
"""
import pytest
import os
import yaml
import tempfile
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from pathlib import Path
import sys

# Add the parent directory to the path so we can import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from ode_uq.utils import load_yaml_of_distributions, sample_from_dict_of_distributions, inverse_cdf_from_dict_of_distributions


class TestParameterDistributions:
    """Test class for parameter distribution functionality."""
    
    def create_test_yaml(self, content):
        """Create a temporary YAML file with the given content."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(content, temp_file)
        temp_file.close()
        return temp_file.name
    
    def test_load_simple_normal_distribution(self):
        """Test loading a simple Normal distribution."""
        test_content = {
            'param1': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            assert 'param1' in distributions
            assert isinstance(distributions['param1'], dist.Normal)
            
            # Test that the distribution has correct parameters
            distribution = distributions['param1']
            assert distribution.loc == 0.0
            assert distribution.scale == 1.0
            
        finally:
            os.unlink(temp_file)
    
    def test_load_lognormal_distribution(self):
        """Test loading a LogNormal distribution."""
        test_content = {
            'k1': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': -6.91,
                    'scale': 0.5
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            assert 'k1' in distributions
            assert isinstance(distributions['k1'], dist.LogNormal)
            
            distribution = distributions['k1']
            assert distribution.loc == -6.91
            assert distribution.scale == 0.5
            
        finally:
            os.unlink(temp_file)
    
    def test_load_multiple_distributions(self):
        """Test loading multiple different distribution types."""
        test_content = {
            'normal_param': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 1.0,
                    'scale': 2.0
                }
            },
            'uniform_param': {
                'dist_name': 'Uniform',
                'dist_params': {
                    'low': 0.0,
                    'high': 1.0
                }
            },
            'gamma_param': {
                'dist_name': 'Gamma',
                'dist_params': {
                    'concentration': 2.0,
                    'rate': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            assert len(distributions) == 3
            assert isinstance(distributions['normal_param'], dist.Normal)
            assert isinstance(distributions['uniform_param'], dist.Uniform)
            assert isinstance(distributions['gamma_param'], dist.Gamma)
            
        finally:
            os.unlink(temp_file)
    
    def test_unsupported_distribution(self):
        """Test that unsupported distribution raises ValueError."""
        test_content = {
            'bad_param': {
                'dist_name': 'UnsupportedDistribution',
                'dist_params': {
                    'param1': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            with pytest.raises(ValueError, match="Unsupported distribution type"):
                load_yaml_of_distributions(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_missing_required_fields(self):
        """Test that missing required fields raise KeyError."""
        # Missing 'dist_name' field
        test_content = {
            'param1': {
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            with pytest.raises(KeyError, match="Missing required key"):
                load_yaml_of_distributions(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_invalid_distribution_parameters(self):
        """Test that invalid distribution parameters raise ValueError."""
        test_content = {
            'param1': {
                'dist_name': 'Normal',
                'dist_params': {
                    'invalid_param': 1.0  # Normal doesn't have 'invalid_param'
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            with pytest.raises(ValueError, match="Invalid parameters"):
                load_yaml_of_distributions(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Parameter distribution file not found"):
            load_yaml_of_distributions("nonexistent_file.yaml")
    
    def test_sample_from_distributions(self):
        """Test sampling from parameter distributions."""
        test_content = {
            'param1': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            },
            'param2': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Test single sample
            key = jax.random.PRNGKey(42)
            samples = sample_from_dict_of_distributions(distributions, key, num_samples=1)
            
            assert 'param1' in samples
            assert 'param2' in samples
            assert jnp.isscalar(samples['param1'])
            assert jnp.isscalar(samples['param2'])
            assert samples['param2'] > 0  # LogNormal samples should be positive
            
            # Test multiple samples
            key = jax.random.PRNGKey(123)
            samples = sample_from_dict_of_distributions(distributions, key, num_samples=10)
            
            assert samples['param1'].shape == (10,)
            assert samples['param2'].shape == (10,)
            assert jnp.all(samples['param2'] > 0)  # LogNormal samples should be positive
            
        finally:
            os.unlink(temp_file)
    
    def test_real_parameter_file(self):
        """Test loading the actual parameter_dist.yaml file."""
        # Get the path to the parameter_dist.yaml file
        parent_dir = Path(__file__).parent.parent
        param_file = parent_dir / "parameter_dist.yaml"
        
        if param_file.exists():
            distributions = load_yaml_of_distributions(str(param_file))
            
            # Check that we have the expected parameters
            expected_params = [
                'k1', 'k2', 'k3', 'k4', 'k5',
                'kr1', 'kr2', 'kr3', 'kr4', 'kr5',
                'kt_Peptide_Blood', 'kt_PeptideDimer_Blood',
                'kt_Peptide_albumin_Blood', 'kt_PeptideDimer_albumin_Blood',
                'kt_Peptide_ECM_Blood', 'kt_PeptideDimer_ECM_Blood',
                'kt_Peptide_Tissue', 'kt_PeptideDimer_Tissue',
                'kt_Peptide_albumin_Tissue', 'kt_PeptideDimer_albumin_Tissue',
                'kt_Peptide_ECM_Tissue', 'kt_PeptideDimer_ECM_Tissue'
            ]
            
            for param in expected_params:
                assert param in distributions
                assert isinstance(distributions[param], dist.LogNormal)
            
            # Test sampling from the real distributions
            key = jax.random.PRNGKey(42)
            samples = sample_from_dict_of_distributions(distributions, key)
            
            # All samples should be positive (LogNormal distributions)
            for param, value in samples.items():
                assert value > 0, f"Parameter {param} sample should be positive"
        else:
            pytest.skip("parameter_dist.yaml file not found")
    
    def test_inverse_cdf_from_distributions(self):
        """Test computing inverse CDF from parameter distributions."""
        test_content = {
            'param1': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            },
            'param2': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Test inverse CDF at specific quantiles (using dict format)
            quantiles_dict = {
                'param1': jnp.array([0.25, 0.5, 0.75]),
                'param2': jnp.array([0.25, 0.5, 0.75])
            }
            inverse_cdfs = inverse_cdf_from_dict_of_distributions(quantiles_dict, distributions)
            
            assert 'param1' in inverse_cdfs
            assert 'param2' in inverse_cdfs
            assert inverse_cdfs['param1'].shape == (3,)
            assert inverse_cdfs['param2'].shape == (3,)
            
            # For standard normal, median should be approximately 0
            assert jnp.abs(inverse_cdfs['param1'][1]) < 0.1  # 50th percentile ≈ 0
            
            # LogNormal samples should be positive
            assert jnp.all(inverse_cdfs['param2'] > 0)
            
        finally:
            os.unlink(temp_file)
    
    def test_inverse_cdf_uniform_sampling_equivalence(self):
        """Test that uniform sampling + inverse CDF equals direct sampling."""
        test_content = {
            'normal_param': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 2.0,
                    'scale': 1.5
                }
            },
            'lognormal_param': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': 0.5,
                    'scale': 0.8
                }
            },
            'uniform_param': {
                'dist_name': 'Uniform',
                'dist_params': {
                    'low': -1.0,
                    'high': 3.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Number of samples for statistical test
            num_samples = 5000
            
            # Method 1: Direct sampling
            key1 = jax.random.PRNGKey(42)
            direct_samples = sample_from_dict_of_distributions(distributions, key1, num_samples)
            
            # Method 2: Uniform sampling + inverse CDF
            key2 = jax.random.PRNGKey(42)  # Same key for fair comparison
            uniform_keys = jax.random.split(key2, len(distributions))
            
            inverse_cdf_samples = {}
            for i, (param_name, distribution) in enumerate(distributions.items()):
                # Generate uniform samples
                uniform_samples = jax.random.uniform(uniform_keys[i], (num_samples,))
                
                # Apply inverse CDF
                inverse_cdf_samples[param_name] = distribution.icdf(uniform_samples)
            
            # Statistical tests to verify distributions are equivalent
            for param_name in distributions.keys():
                direct = direct_samples[param_name]
                inverse_cdf = inverse_cdf_samples[param_name]
                
                # Test 1: Sample means should be close
                mean_diff = jnp.abs(jnp.mean(direct) - jnp.mean(inverse_cdf))
                assert mean_diff < 0.1, f"Mean difference too large for {param_name}: {mean_diff}"
                
                # Test 2: Sample standard deviations should be close
                std_diff = jnp.abs(jnp.std(direct) - jnp.std(inverse_cdf))
                assert std_diff < 0.1, f"Std difference too large for {param_name}: {std_diff}"
                
                # Test 3: Quantiles should be close (more robust test)
                quantiles = jnp.array([0.1, 0.25, 0.5, 0.75, 0.9])
                direct_quantiles = jnp.quantile(direct, quantiles)
                inverse_quantiles = jnp.quantile(inverse_cdf, quantiles)
                
                max_quantile_diff = jnp.max(jnp.abs(direct_quantiles - inverse_quantiles))
                assert max_quantile_diff < 0.2, f"Quantile difference too large for {param_name}: {max_quantile_diff}"
                
            # Test the convenience function version
            uniform_dict = {param_name: jax.random.uniform(jax.random.PRNGKey(123), (num_samples,)) 
                           for param_name in distributions.keys()}
            
            convenience_samples = inverse_cdf_from_dict_of_distributions(uniform_dict, distributions)
            
            # Check that convenience function produces same results as manual approach
            for param_name in distributions.keys():
                manual_result = inverse_cdf_samples[param_name]
                # Note: We can't compare directly due to different random keys, 
                # but we can check that the shapes match
                assert convenience_samples[param_name].shape == manual_result.shape
                
        finally:
            os.unlink(temp_file)
    
    def test_inverse_cdf_mathematical_properties(self):
        """Test mathematical properties of inverse CDF function."""
        test_content = {
            'test_normal': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            },
            'test_lognormal': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            }
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Test boundary conditions
            boundary_quantiles = {param_name: jnp.array([0.001, 0.999]) 
                                for param_name in distributions.keys()}
            boundary_results = inverse_cdf_from_dict_of_distributions(boundary_quantiles, distributions)
            
            # For all distributions, inverse CDF should be finite at boundary quantiles
            for param_name, values in boundary_results.items():
                assert jnp.all(jnp.isfinite(values)), f"Non-finite values in {param_name}: {values}"
                assert values[0] < values[1], f"Inverse CDF not monotonic for {param_name}: {values}"
            
            # Test median property (50th percentile)
            median_quantiles = {param_name: jnp.array([0.5]) for param_name in distributions.keys()}
            median_results = inverse_cdf_from_dict_of_distributions(median_quantiles, distributions)
            
            # For standard normal, median should be close to 0
            normal_median = median_results['test_normal'][0]
            assert jnp.abs(normal_median) < 0.01, f"Normal median not close to 0: {normal_median}"
            
            # For log-normal with loc=0, scale=1, median should be close to 1
            lognormal_median = median_results['test_lognormal'][0]
            assert jnp.abs(lognormal_median - 1.0) < 0.01, f"LogNormal median not close to 1: {lognormal_median}"
            
        finally:
            os.unlink(temp_file)

    def test_mixed_deterministic_and_probabilistic_parameters(self):
        """Test handling of mixed deterministic and probabilistic parameters."""
        test_content = {
            # Probabilistic parameters
            'k1': {
                'dist_name': 'LogNormal',
                'dist_params': {
                    'loc': -6.91,
                    'scale': 0.5
                }
            },
            'k2': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 0.0,
                    'scale': 1.0
                }
            },
            # Deterministic parameters (fixed values)
            'k3': 1.2,
            'k4': 0.005,
            'k5': 42.0
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Verify we have both types
            deterministic_params = {k: v for k, v in distributions.items() 
                                   if isinstance(v, (int, float))}
            probabilistic_params = {k: v for k, v in distributions.items() 
                                   if not isinstance(v, (int, float))}
            
            assert len(deterministic_params) == 3, f"Expected 3 deterministic params, got {len(deterministic_params)}"
            assert len(probabilistic_params) == 2, f"Expected 2 probabilistic params, got {len(probabilistic_params)}"
            
            # Check deterministic values are correct
            assert distributions['k3'] == 1.2
            assert distributions['k4'] == 0.005
            assert distributions['k5'] == 42.0
            
            # Check probabilistic parameters are distributions
            import numpyro.distributions as dist
            assert isinstance(distributions['k1'], dist.LogNormal)
            assert isinstance(distributions['k2'], dist.Normal)
            
        finally:
            os.unlink(temp_file)

    def test_sampling_with_mixed_parameters(self):
        """Test sampling from mixed deterministic and probabilistic parameters."""
        test_content = {
            'prob_param': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 5.0,
                    'scale': 2.0
                }
            },
            'det_param1': 1.2,
            'det_param2': 0.005
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            key = jax.random.PRNGKey(123)
            
            # Test single sample
            samples = sample_from_dict_of_distributions(distributions, key, num_samples=1)
            
            # Deterministic parameters should return exact values
            assert samples['det_param1'] == 1.2
            assert samples['det_param2'] == 0.005
            
            # Probabilistic parameter should return a sampled value
            assert jnp.isfinite(samples['prob_param'])
            
            # Test multiple samples
            samples_multi = sample_from_dict_of_distributions(distributions, key, num_samples=5)
            
            # Deterministic parameters should return arrays of the same value
            assert jnp.allclose(samples_multi['det_param1'], jnp.full(5, 1.2))
            assert jnp.allclose(samples_multi['det_param2'], jnp.full(5, 0.005))
            
            # Probabilistic parameter should return an array
            assert samples_multi['prob_param'].shape == (5,)
            
        finally:
            os.unlink(temp_file)

    def test_inverse_cdf_with_mixed_parameters(self):
        """Test inverse CDF computation with mixed deterministic and probabilistic parameters."""
        test_content = {
            'uniform_param': {
                'dist_name': 'Uniform',
                'dist_params': {
                    'low': 0.0,
                    'high': 10.0
                }
            },
            'det_param': 3.14
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            
            # Create quantiles for all parameters (deterministic ones will ignore them)
            quantiles = {
                'uniform_param': 0.25,  # 25th percentile of Uniform[0,10] = 2.5
                'det_param': 0.9       # This will be ignored (deterministic)
            }
            
            results = inverse_cdf_from_dict_of_distributions(quantiles, distributions)
            
            # Deterministic parameter should return its fixed value regardless of quantile
            assert results['det_param'] == 3.14
            
            # Probabilistic parameter should return quantile-based value
            assert jnp.isclose(results['uniform_param'], 2.5)  # Uniform[0,10] 25th percentile is 2.5
            
        finally:
            os.unlink(temp_file)

    def test_mixed_parameters_equivalence(self):
        """Test that uniform sampling + inverse CDF equals direct sampling for mixed parameters."""
        test_content = {
            'normal_param': {
                'dist_name': 'Normal',
                'dist_params': {
                    'loc': 1.0,
                    'scale': 0.5
                }
            },
            'det_param': 7.5
        }
        
        temp_file = self.create_test_yaml(test_content)
        try:
            distributions = load_yaml_of_distributions(temp_file)
            key = jax.random.PRNGKey(456)
            
            num_samples = 1000
            
            # Method 1: Direct sampling
            direct_samples = sample_from_dict_of_distributions(distributions, key, num_samples)
            
            # Method 2: Uniform sampling + inverse CDF
            key_uniform = jax.random.PRNGKey(456)  # Same seed for reproducibility
            
            # Generate uniform samples for probabilistic parameters only
            probabilistic_params = [k for k, v in distributions.items() 
                                   if not isinstance(v, (int, float))]
            
            uniform_samples = {}
            if probabilistic_params:
                uniform_keys = jax.random.split(key_uniform, len(probabilistic_params))
                
                for i, param in enumerate(probabilistic_params):
                    uniform_samples[param] = jax.random.uniform(uniform_keys[i], (num_samples,))
            
            # Add dummy quantiles for deterministic parameters (will be ignored)
            for param, value in distributions.items():
                if isinstance(value, (int, float)):
                    uniform_samples[param] = jnp.full(num_samples, 0.5)  # Dummy values
            
            inverse_cdf_samples = inverse_cdf_from_dict_of_distributions(uniform_samples, distributions)
            
            # Compare results
            for param in distributions.keys():
                if isinstance(distributions[param], (int, float)):
                    # Deterministic parameters should be identical
                    assert jnp.allclose(direct_samples[param], inverse_cdf_samples[param])
                else:
                    # Probabilistic parameters: check statistical properties
                    direct_mean = jnp.mean(direct_samples[param])
                    inverse_mean = jnp.mean(inverse_cdf_samples[param])
                    
                    # Should have similar means (within reasonable tolerance for 1000 samples)
                    rel_diff = abs(direct_mean - inverse_mean) / abs(direct_mean)
                    assert rel_diff < 0.1, f"Mean difference too large for {param}: {rel_diff}"
                    
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    pytest.main([__file__])
