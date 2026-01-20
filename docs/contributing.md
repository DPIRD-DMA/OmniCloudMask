# Contributing

Contributions to OmniCloudMask are welcome. Please submit a pull request or open an issue to discuss changes.

## Development Setup

Clone the repository and install development dependencies:

```bash
git clone https://github.com/DPIRD-DMA/OmniCloudMask.git
cd OmniCloudMask
uv sync --group dev
```

## Code Style

OmniCloudMask uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting.

Format code:

```bash
ruff format .
```

Check for lint errors:

```bash
ruff check .
```

## Running Tests

```bash
uv run pytest tests/
```

## License

OmniCloudMask is released under the MIT License.

## Acknowledgements

- [CloudSEN12 project](https://cloudsen12.github.io/) for the training dataset used in model versions 1.0-4.0
- [KappaSet authors](https://doi.org/10.5281/zenodo.7100327) for additional training data used in versions 3.0 and 4.0
