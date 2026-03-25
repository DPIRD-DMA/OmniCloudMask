"""Generate docs/benchmarks.md from benchmarking/results/*.json."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_PAGE = DOCS_DIR / "benchmarks.md"
PLOT_PATH = DOCS_DIR / "_static" / "benchmark.png"


def load_results() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")
    return [json.loads(f.read_text()) for f in files]


def make_table(data: dict) -> str:
    fp = data["fingerprint"]
    rows = data["results"]

    # Determine columns: unique (device, dtype) combos, cpu first
    combos: list[tuple[str, str]] = []
    for r in rows:
        key = (r["device"], r["dtype"])
        if key not in combos:
            combos.append(key)
    dtype_order = {"float32": 0, "float16": 1, "bfloat16": 2}
    combos.sort(
        key=lambda k: (0 if k[0] == "cpu" else 1, k[0], dtype_order.get(k[1], 99))
    )

    # Index results by (device, dtype, scene_size)
    index: dict[tuple, tuple[float, int]] = {
        (r["device"], r["dtype"], r["scene_size"]): (r["mean_seconds"], r["batch_size"])
        for r in rows
    }

    # All scene sizes across all combos, sorted
    sizes = sorted({r["scene_size"] for r in rows})

    # Non-CPU devices for batch size columns (one per device, not per dtype)
    gpu_devices = list(dict.fromkeys(d for d, _ in combos if d != "cpu"))

    # Header: time columns then one batch size column per GPU device
    col_headers = []
    for device, dtype in combos:
        device_label = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}.get(
            device, device.upper()
        )
        col_headers.append(f"{device_label} {dtype}" if device != "cpu" else "CPU")
    for _ in gpu_devices:
        col_headers.append("Batch size")

    n_cols = len(col_headers)
    header = "| Scene size | " + " | ".join(col_headers) + " |"
    separator = "| --- | " + " | ".join("---" for _ in range(n_cols)) + " |"

    table_rows = []
    for size in sizes:
        cells = [f"**{size}×{size} px**"]
        for combo in combos:
            val = index.get((*combo, size))
            cells.append(f"{val[0]:.3f}s" if val is not None else "—")
        # Batch size columns — use first dtype for each device
        for device in gpu_devices:
            first_dtype = next(dt for dev, dt in combos if dev == device)
            val = index.get((device, first_dtype, size))
            cells.append(str(val[1]) if val is not None else "—")
        table_rows.append("| " + " | ".join(cells) + " |")

    # Build hardware label
    gpu = fp.get("gpu")
    cpu = fp["cpu"]
    if gpu and gpu != cpu and "MPS" not in gpu:
        hw = f"{gpu}, {cpu}"
    elif gpu:
        hw = gpu
    else:
        hw = cpu

    os_str = fp.get("os", "")
    ram = fp.get("ram_gb", "")
    ocm_ver = fp.get("omnicloudmask_version", "")

    lines = [
        f"### {hw}",
        "",
        f"OmniCloudMask {ocm_ver} &middot; {os_str} &middot; {ram} GB RAM",
        "",
        header,
        separator,
        *table_rows,
    ]
    return "\n".join(lines)


COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]


def make_plot(all_data: list[dict]) -> None:
    import matplotlib.pyplot as plt

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_prop_cycle(color=COLORS)

    for data in all_data:
        fp = data["fingerprint"]
        gpu = fp.get("gpu")
        cpu = fp["cpu"]

        # Build a short hardware label
        if gpu and "MPS" in gpu:
            hw = gpu  # e.g. "Apple M4 Pro (MPS)"
        elif gpu:
            hw = gpu  # e.g. "NVIDIA GeForce RTX 4090"
        else:
            hw = cpu

        rows = data["results"]
        dtype_order = {"float32": 0, "float16": 1, "bfloat16": 2}

        # Group by (device, dtype)
        combos: list[tuple[str, str]] = []
        for r in rows:
            key = (r["device"], r["dtype"])
            if key not in combos:
                combos.append(key)
        combos.sort(
            key=lambda k: (0 if k[0] == "cpu" else 1, k[0], dtype_order.get(k[1], 99))
        )

        index = {
            (r["device"], r["dtype"], r["scene_size"]): r["mean_seconds"] for r in rows
        }

        # One line per device: prefer float16, fall back to float32
        devices_seen: list[str] = []
        for device, _ in combos:
            if device not in devices_seen:
                devices_seen.append(device)

        device_label = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}
        for device in devices_seen:
            dtypes_for_device = [dt for dev, dt in combos if dev == device]
            dtype = (
                "float16" if "float16" in dtypes_for_device else dtypes_for_device[0]
            )
            sizes = sorted(
                {
                    r["scene_size"]
                    for r in rows
                    if r["device"] == device and r["dtype"] == dtype
                }
            )
            megapixels = [s**2 / 1e6 for s in sizes]
            times = [index[(device, dtype, s)] for s in sizes]
            dev_str = device_label.get(device, device.upper())
            label = f"{hw} — {dev_str}" if device != "cpu" else f"{cpu} — CPU"
            ax.plot(megapixels, times, "o-", label=label, lw=2, ms=6)

    ax.set_xlabel("Scene size (megapixels)", fontsize=12)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Inference Time vs Scene Size", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.2, color="grey")
    ax.set_xlim(left=0, right=100)
    ax.set_ylim(bottom=0, top=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {PLOT_PATH}")


def main() -> None:
    all_data = load_results()

    make_plot(all_data)
    tables = [make_table(d) for d in all_data]

    page = "\n".join(
        [
            "# Benchmarks",
            "",
            "Inference time for a square scene at various sizes.",
            "Results show mean seconds over multiple runs.",
            "Batch size was selected automatically by searching for the fastest value on each device.",  # noqa: E501
            "",
            "![Benchmark plot](_static/benchmark.png)",
            "",
            *("\n\n".join(tables).splitlines()),
            "",
            "---",
            "",
            "_To add results for your hardware, run `benchmarking/benchmarking.ipynb`"
            " and submit the JSON file in `benchmarking/results/` via a pull request._",
        ]
    )

    DOCS_PAGE.write_text(page + "\n")
    print(f"Written: {DOCS_PAGE}")


if __name__ == "__main__":
    main()
