# Project Context

Python library for cloud masking optimized for CUDA MPS and CPU.

# Development Tools

- **Package manager**: uv
- **Formatter**: ruff
- **Testing**: pytest

# Commands

- `uv run pytest tests/` - Run test suite
- `ruff format .` - Format code
- `ruff check .` - Lint code

# Code Style

- Use type hints for all functions and methods
- Follow ruff formatting standards

# Workflow

- User handles git management (no commits/pushes unless requested)
- Batch multiple independent bash commands in parallel to reduce token usage
- Test changes only if requested

# Performance Profiling Findings

## Threading Architecture in predict_from_load_func

When profiling with cProfile, thread lock acquisition time (e.g., 8s spent in `_thread.lock.acquire()`) represents **actual GPU inference work**, not overhead. The threading pattern is:

1. Main thread loads scene N
2. Spawns background thread running `coordinator()` (GPU inference)
3. Immediately continues to load scene N+1
4. Eventually joins background threads to complete

The "waiting" time seen in profilers is the GPU actively processing in background threads. This is **efficient pipeline parallelism**, not wasted overhead.