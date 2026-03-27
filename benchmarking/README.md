# Benchmarking

This directory contains tools for profiling OmniCloudMask inference performance across devices and scene sizes.

## Files

### `benchmarking.ipynb`

Jupyter notebook that measures inference time for square scenes at a range of sizes across all available devices (CPU, CUDA, MPS) and dtypes (float32, float16).

For each device/dtype/scene-size combination it:
- Warms up the device
- Searches for the optimal batch size (for scenes > 1000 px)
- Times inference over multiple runs and records mean and standard deviation

Results are saved to `results/<hardware>_<date>.json`.

### `generate_docs_table.py`

Script that reads all JSON files in `results/` and regenerates the benchmarks documentation page.

```bash
uv run benchmarking/generate_docs_table.py
```

This writes:
- `docs/benchmarks.md` — a page with a summary plot and per-hardware tables
- `docs/_static/benchmark.png` — the plot embedded in the page

Run this script and commit both outputs whenever new results are added.

### Previewing the docs locally

To build and serve the documentation locally with live reloading:

```bash
uv run sphinx-autobuild docs docs/_build/html
```

The docs will be served at `http://127.0.0.1:8000` and will automatically rebuild when you edit any source files.

## Adding results for your hardware

1. Open `benchmarking.ipynb` and run all cells
2. The results JSON will be saved to `benchmarking/results/`
3. Run `generate_docs_table.py` to update the docs
4. Submit a pull request with the new JSON file and updated docs
