# Tests directory
This directory contains tests for the peptide model utilities and functions.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_parameter_distributions.py
```

To run tests with verbose output:
```bash
pytest tests/ -v
```

## Test Files

- `test_parameter_distributions.py`: Tests for parameter distribution loading, sampling, and inverse CDF functionality
