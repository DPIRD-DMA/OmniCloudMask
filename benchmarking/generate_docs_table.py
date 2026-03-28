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
    seen = dict.fromkeys((r["device"], r["dtype"]) for r in rows)
    return sorted(
        seen,
        key=lambda k: (0 if k[0] == "cpu" else 1, k[0], DTYPE_ORDER.get(k[1], 99)),
    )


def _best_time(rows: list[dict], device: str, dtype: str | None = None) -> float:
    """Fastest mean_seconds at 1000×1000 for device/dtype, falling back to smallest."""
    filtered = [
        r for r in rows
        if r["device"] == device and (dtype is None or r["dtype"] == dtype)
    ]
    candidates = [r for r in filtered if r["scene_size"] == 1000]
    if not candidates:
        candidates = [min(filtered, key=lambda r: r["scene_size"])]
    return min(r["mean_seconds"] for r in candidates)


def _make_device_table(
    device: str,
    dtypes: list[str],
    rows: list[dict],
    index: dict[tuple, tuple[float, int]],
) -> list[str]:
    """Build a markdown table for a single device."""
    sizes = sorted({r["scene_size"] for r in rows if r["device"] == device})

    short_dtype = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}
    col_headers = []
    for dtype in dtypes:
        dt = short_dtype.get(dtype, dtype)
        col_headers.append(dt)
        col_headers.append(f"Batch ({dt})" if len(dtypes) > 1 else "Batch")

    header = "| Megapixels | Dimensions | " + " | ".join(col_headers) + " |"
    separator = "| --- | --- | " + " | ".join("---" for _ in col_headers) + " |"

    table_rows = []
    for i, size in enumerate(sizes):
        decimals = 3 if i == 0 else 2
        mp = size ** 2 / 1e6
        if mp >= 1:
            mp_str = f"{mp:.0f}"
        elif i == 0:
            mp_str = f"{mp:.3f}"
        else:
            mp_str = f"{mp:.2f}"
        cells = [f"**{mp_str}**", f"{size}\u00d7{size}"]
        for dtype in dtypes:
            val = index.get((device, dtype, size))
            cells.append(f"{val[0]:.{decimals}f}s" if val is not None else "\u2014")
            cells.append(str(val[1]) if val is not None else "\u2014")
        table_rows.append("| " + " | ".join(cells) + " |")

    return [header, separator, *table_rows]


def make_tables(data: dict) -> list[tuple[float, str]]:
    """Return a list of (best_time, section_markdown) for each device.

    best_time is the fastest time at 1000x1000, used for sorting.
    """
    fp = data["fingerprint"]
    rows = data["results"]
    combos = _get_combos(rows)

    index: dict[tuple, tuple[float, int]] = {
        (r["device"], r["dtype"], r["scene_size"]): (r["mean_seconds"], r["batch_size"])
        for r in rows
    }

    device_dtypes: dict[str, list[str]] = {}
    for device, dtype in combos:
        device_dtypes.setdefault(device, []).append(dtype)

    hw, cpu = _get_hw_label(fp)
    os_str = fp.get("os", "")
    ram = fp.get("ram_gb", "")
    ocm_ver = fp.get("omnicloudmask_version", "")
    meta = f"OmniCloudMask {ocm_ver} &middot; {os_str} &middot; {ram} GB RAM"

    sections: list[tuple[float, str]] = []
    for device, dtypes in device_dtypes.items():
        hw_name = cpu if device == "cpu" else hw
        dev_str = DEVICE_LABELS.get(device, device.upper())
        name = f"{hw_name} ({dev_str})" if len(device_dtypes) > 1 else hw_name

        lines = [
            f"## {name}",
            "",
            meta,
            "",
            *_make_device_table(device, dtypes, rows, index),
        ]
        sections.append((_best_time(rows, device), "\n".join(lines)))

    return sections


def _wrap_hw_name(name: str, max_len: int = 12) -> str:
    """Insert <br> at the last word boundary before max_len chars."""
    if len(name) <= max_len:
        return name
    pos = name.rfind(" ", 0, max_len)
    if pos == -1:
        pos = name.find(" ")
    return f"{name[:pos]}<br>{name[pos + 1:]}" if pos != -1 else name


def make_summary_table(all_data: list[dict]) -> str:
    """One table across all hardware: fastest dtype per device, no batch size."""
    columns: list[tuple[float, str]] = []
    index: dict[tuple[str, int], float] = {}

    for data in all_data:
        fp = data["fingerprint"]
        rows = data["results"]
        hw, cpu = _get_hw_label(fp)
        combos = _get_combos(rows)

        for device in dict.fromkeys(d for d, _ in combos):
            dtypes_for_device = [dt for dev, dt in combos if dev == device]
            best_dtype = min(
                dtypes_for_device, key=lambda dt: _best_time(rows, device, dt)
            )
            hw_name = cpu if device == "cpu" else hw
            dev_str = DEVICE_LABELS.get(device, device.upper())
            label = f"{_wrap_hw_name(hw_name)}<br>({dev_str})"
            columns.append((_best_time(rows, device, best_dtype), label))
            for r in rows:
                if r["device"] == device and r["dtype"] == best_dtype:
                    index[(label, r["scene_size"])] = r["mean_seconds"]

    columns.sort()
    col_labels = [label for _, label in columns]
    all_sizes = sorted({size for _, size in index})

    header = "| Megapixels | Dimensions | " + " | ".join(col_labels) + " |"
    separator = "| --- | --- | " + " | ".join("---" for _ in col_labels) + " |"
    table_rows = []
    for i, size in enumerate(all_sizes):
        decimals = 3 if i == 0 else 2
        mp = size ** 2 / 1e6
        if mp >= 1:
            mp_str = f"{mp:.0f}"
        elif i == 0:
            mp_str = f"{mp:.3f}"
        else:
            mp_str = f"{mp:.2f}"
        cells = [f"**{mp_str}**", f"{size}\u00d7{size}"]
        for label in col_labels:
            val = index.get((label, size))
            cells.append(f"{val:.{decimals}f}s" if val is not None else "\u2014")
        table_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(["## Summary", "", header, separator, *table_rows])


def make_plot(all_data: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import seaborn as sns

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="notebook")

    fig, ax = plt.subplots(figsize=(9, 5))

    # Collect all series, then sort fastest-first by time at 1000x1000
    series: list[tuple[float, str, str, list[float], list[float]]] = []
    for data in all_data:
        fp = data["fingerprint"]
        hw, cpu = _get_hw_label(fp)
        rows = data["results"]
        combos = _get_combos(rows)

        index = {
            (r["device"], r["dtype"], r["scene_size"]): r["mean_seconds"] for r in rows
        }

        for device in dict.fromkeys(d for d, _ in combos):
            dtypes_for_device = [dt for dev, dt in combos if dev == device]
            dtype = "float16" if "float16" in dtypes_for_device else dtypes_for_device[0]
            sizes = sorted(
                r["scene_size"] for r in rows
                if r["device"] == device and r["dtype"] == dtype
            )
            megapixels = [s**2 / 1e6 for s in sizes]
            times = [index[(device, dtype, s)] for s in sizes]

            hw_name = cpu if device == "cpu" else hw
            dev_str = DEVICE_LABELS.get(device, device.upper())
            label = f"{hw_name} ({dev_str})"
            sort_key = _best_time(rows, device, dtype)
            series.append((sort_key, device, label, megapixels, times))

    series.sort(key=lambda s: s[0])

    palette = sns.color_palette("husl", len(series))
    plot_handles: list[tuple[float, object, str]] = []
    for i, (sort_key, _device, label, megapixels, times) in enumerate(series):
        color = palette[i]
        (line,) = ax.plot(megapixels, times, "-", label=label, color=color, lw=2)
        plot_handles.append((sort_key, line, label))
        ax.annotate(
            f"{times[-1]:.3g}s",
            xy=(megapixels[-1], times[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.set_yscale("log")
    ax.set_xlim(left=-2, right=110)
    y_min = min(t for _, _, _, _, times in series for t in times)
    ax.set_ylim(bottom=y_min * 0.5)
    all_x = sorted({mp for _, _, _, megapixels, _ in series for mp in megapixels})
    preferred = [1, 4, 9, 25, 100]
    ax.set_xticks([x for x in preferred if x in all_x])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xlabel("Scene size (megapixels)")
    ax.set_ylabel("Inference time (seconds)")
    ax.set_title("Inference Time vs Scene Size", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(True, axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(False, which="minor")
    ax.legend(
        [e[1] for e in plot_handles],
        [e[2] for e in plot_handles],
        frameon=True, fancybox=False, shadow=False, fontsize=9,
        ncol=2, loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {PLOT_PATH}")


def main() -> None:
    all_data = load_results()

    make_plot(all_data)

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
            make_summary_table(all_data),
            "",
            *("\n\n".join(tables).splitlines()),
        ]
    )

    DOCS_PAGE.write_text(page + "\n")
    print(f"Written: {DOCS_PAGE}")


if __name__ == "__main__":
    main()
