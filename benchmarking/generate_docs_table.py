"""Generate docs/benchmarks.md from benchmarking/results/*.json."""

import json
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DOCS_DIR = Path(__file__).parent.parent / "docs"
DOCS_PAGE = DOCS_DIR / "benchmarks.md"
PLOT_PATH = DOCS_DIR / "_static" / "benchmark.png"

DTYPE_ORDER = {"float32": 0, "float16": 1, "bfloat16": 2}
DEVICE_LABELS = {"cpu": "CPU", "cuda": "CUDA", "mps": "MPS"}


def load_results() -> list[dict]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {RESULTS_DIR}")
    return [json.loads(f.read_text()) for f in files]


def _clean_cpu_name(name: str) -> str:
    """Strip common suffixes like '16-Core Processor' from CPU names."""
    return re.sub(r"\s+\d+-Core Processor$", "", name)


def _get_hw_label(fp: dict) -> tuple[str, str]:
    """Return (hardware_label, cpu_name) from a fingerprint."""
    gpu = fp.get("gpu")
    cpu = _clean_cpu_name(fp["cpu"])
    if gpu and "MPS" in gpu:
        hw = gpu.replace(" (MPS)", "")
    elif gpu:
        hw = gpu
    else:
        hw = cpu
    return hw, cpu


def _get_combos(rows: list[dict]) -> list[tuple[str, str]]:
    """Return sorted unique (device, dtype) combos, CPU first."""
    combos: list[tuple[str, str]] = []
    for r in rows:
        key = (r["device"], r["dtype"])
        if key not in combos:
            combos.append(key)
    combos.sort(
        key=lambda k: (0 if k[0] == "cpu" else 1, k[0], DTYPE_ORDER.get(k[1], 99))
    )
    return combos


def _make_device_table(
    device: str,
    dtypes: list[str],
    rows: list[dict],
    index: dict[tuple, tuple[float, int]],
) -> list[str]:
    """Build a markdown table for a single device."""
    sizes = sorted(
        {r["scene_size"] for r in rows if r["device"] == device}
    )

    short_dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    col_headers = []
    for dtype in dtypes:
        dt = short_dtype.get(dtype, dtype)
        col_headers.append(dt)
        col_headers.append(f"Batch ({dt})" if len(dtypes) > 1 else "Batch")

    header = "| Scene size | " + " | ".join(col_headers) + " |"
    separator = "| --- | " + " | ".join("---" for _ in col_headers) + " |"

    table_rows = []
    for size in sizes:
        cells = [f"**{size}\u00d7{size} px**"]
        for dtype in dtypes:
            val = index.get((device, dtype, size))
            cells.append(f"{val[0]:.3f}s" if val is not None else "\u2014")
            cells.append(str(val[1]) if val is not None else "\u2014")
        table_rows.append("| " + " | ".join(cells) + " |")

    return [header, separator, *table_rows]


def make_tables(data: dict) -> list[tuple[float, str]]:
    """Return a list of (best_time, section_markdown) for each device.

    best_time is the fastest time at the largest scene size, used for sorting.
    """
    fp = data["fingerprint"]
    rows = data["results"]
    combos = _get_combos(rows)

    index: dict[tuple, tuple[float, int]] = {
        (r["device"], r["dtype"], r["scene_size"]): (r["mean_seconds"], r["batch_size"])
        for r in rows
    }

    # Group dtypes by device
    device_dtypes: dict[str, list[str]] = {}
    for device, dtype in combos:
        device_dtypes.setdefault(device, []).append(dtype)

    # Build hardware labels
    gpu = fp.get("gpu")
    cpu = _clean_cpu_name(fp["cpu"])
    os_str = fp.get("os", "")
    ram = fp.get("ram_gb", "")
    ocm_ver = fp.get("omnicloudmask_version", "")

    gpu_clean = gpu.replace(" (MPS)", "") if gpu else None
    is_mps = gpu is not None and "MPS" in gpu
    meta = f"OmniCloudMask {ocm_ver} &middot; {os_str} &middot; {ram} GB RAM"

    sections: list[tuple[float, str]] = []
    for device, dtypes in device_dtypes.items():
        if device == "cpu":
            name = f"{cpu} (CPU)" if is_mps else cpu
        elif is_mps:
            name = f"{gpu_clean} (MPS)"
        else:
            name = gpu_clean or cpu

        # Best time at 1000x1000 for sorting (fall back to smallest size)
        device_rows = [r for r in rows if r["device"] == device]
        sort_rows = [r for r in device_rows if r["scene_size"] == 1000]
        if not sort_rows:
            sort_rows = [min(device_rows, key=lambda r: r["scene_size"])]
        best_time = min(r["mean_seconds"] for r in sort_rows)

        lines = [
            f"## {name}",
            "",
            meta,
            "",
            *_make_device_table(device, dtypes, rows, index),
        ]
        sections.append((best_time, "\n".join(lines)))

    return sections


def make_plot(all_data: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Collect all series, then sort fastest-first by time at 1000x1000
    series: list[tuple[float, str, list[float], list[float]]] = []
    for data in all_data:
        fp = data["fingerprint"]
        hw, cpu = _get_hw_label(fp)
        rows = data["results"]
        combos = _get_combos(rows)

        index = {
            (r["device"], r["dtype"], r["scene_size"]): r["mean_seconds"] for r in rows
        }

        devices_seen: list[str] = list(dict.fromkeys(d for d, _ in combos))

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

            has_mps = "mps" in devices_seen
            if has_mps and len(devices_seen) > 1:
                dev_str = DEVICE_LABELS.get(device, device.upper())
                label = f"{hw} ({dev_str})"
            elif device == "cpu":
                label = cpu
            else:
                label = hw

            # Sort key: time at 1000x1000, fall back to smallest size
            device_rows = [r for r in rows if r["device"] == device and r["dtype"] == dtype]
            sort_rows = [r for r in device_rows if r["scene_size"] == 1000]
            if not sort_rows:
                sort_rows = [min(device_rows, key=lambda r: r["scene_size"])]
            sort_key = min(r["mean_seconds"] for r in sort_rows)

            series.append((sort_key, label, megapixels, times))

    series.sort(key=lambda s: s[0])

    palette = sns.color_palette("husl", len(series))
    for i, (_, label, megapixels, times) in enumerate(series):
        ax.plot(megapixels, times, "-o", label=label, color=palette[i], lw=2, ms=4)

    ax.set_yscale("log")
    ax.set_xlim(left=-2)
    ax.set_ylim(bottom=0.01)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.ticklabel_format(axis="y", style="plain")
    ax.set_xlabel("Megapixels")
    ax.set_ylabel("Seconds (log)")
    ax.set_title("Inference Time vs Scene Size", fontweight="bold")
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {PLOT_PATH}")


def main() -> None:
    all_data = load_results()

    make_plot(all_data)
    # Collect all device sections and sort fastest-first
    all_sections: list[tuple[float, str]] = []
    for d in all_data:
        all_sections.extend(make_tables(d))
    all_sections.sort(key=lambda s: s[0])
    tables = [text for _, text in all_sections]

    page = "\n".join(
        [
            "# Benchmarks",
            "",
            ":::{note}",
            "To add results for your hardware, see"
            " [`benchmarking/README.md`]"
            "(https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/benchmarking/README.md)"
            " for instructions, then submit the JSON file in"
            " `benchmarking/results/` via a pull request.",
            ":::",
            "",
            "Inference time for a square scene at various sizes.",
            "Results show mean seconds over multiple runs.",
            "Batch size was selected automatically by searching for the fastest value on each device.",  # noqa: E501
            "",
            "![Benchmark plot](_static/benchmark.png)",
            "",
            *("\n\n".join(tables).splitlines()),
        ]
    )

    DOCS_PAGE.write_text(page + "\n")
    print(f"Written: {DOCS_PAGE}")


if __name__ == "__main__":
    main()
