# Tests for Food Insecurity Synthetic Dataset Generator

This directory contains tests for the food insecurity synthetic dataset generator.

## Structure

- `conftest.py`: Contains shared pytest fixtures used across tests
- `test_location_mapping.py`: Tests for the country-to-location mapping functionality

## Running Tests

To run all tests:

```bash
# From the project root
python -m pytest tests/
```

To run a specific test file with verbose output:

```bash
python -m pytest tests/test_location_mapping.py -v
```

## Adding New Tests

When adding new tests:

1. Create a new file named `test_*.py` in this directory
2. Use the fixtures defined in `conftest.py` when possible
3. Follow the standard pytest format (functions starting with `test_`)
4. Run the tests to ensure they pass 