"""Benchmark predict_from_batch across input types and output options.

Each scenario is run once to warm up, then timed over several repeats; the
median and spread are reported. Results can be saved to JSON and compared
between runs, e.g. before and after a change:

    uv run python benchmarking/batch_prediction_benchmark.py --save before.json
    # ...make a change...
    uv run python benchmarking/batch_prediction_benchmark.py --save after.json
    uv run python benchmarking/batch_prediction_benchmark.py \
        --compare before.json after.json

The "gpu_floor" scenario times only the models on batches that are already
normalised and on the device, which is the best the pipeline could achieve.
The "loop_*" scenarios call predict_from_array once per image for comparison:
"loop_default" with its default patch settings, "loop_tuned" with one patch
per image (patch_size = image size, no overlap).
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import rasterio as rio
import torch
from rasterio.transform import from_origin

from omnicloudmask import predict_from_array, predict_from_batch
from omnicloudmask.cloud_mask import collect_models
from omnicloudmask.model_utils import get_torch_dtype, infer_batch

SCENARIOS = [
    "gpu_floor",
    "array_float32",
    "array_uint16",
    "list_uint16",
    "tensor_on_device",
    "files_uint16",
    "confidence_uint16",
    "export_uint16",
    "loop_default",
    "loop_tuned",
]


def make_images(count: int, size: int) -> np.ndarray:
    """Synthetic uint16 images with a Sentinel-2-like value range. Model speed
    does not depend on image content."""
    rng = np.random.default_rng(0)
    return rng.integers(200, 5000, size=(count, 3, size, size), dtype=np.uint16)


def write_tifs(images: np.ndarray, directory: Path) -> list[Path]:
    paths = []
    for i, image in enumerate(images):
        path = directory / f"chip_{i:04d}.tif"
        with rio.open(
            path,
            "w",
            driver="GTiff",
            height=image.shape[1],
            width=image.shape[2],
            count=image.shape[0],
            dtype=image.dtype,
            crs="EPSG:32750",
            transform=from_origin(400000, 6500000, 10, 10),
        ) as dst:
            dst.write(image)
        paths.append(path)
    return paths


def load_tif(input_path: Path) -> tuple[np.ndarray, dict]:
    with rio.open(input_path) as src:
        return src.read(), src.profile


def gpu_floor(
    images: np.ndarray, batch_size: int, device: torch.device, dtype: torch.dtype
) -> Callable[[], None]:
    """Time only the models on pre-normalised batches already on the device."""
    models = collect_models(
        custom_models=None,
        inference_device=device,
        inference_dtype=dtype,
        source="hugging_face",
        patch_size=(images.shape[2], images.shape[3]),
        batch_size=batch_size,
    )
    batches = [
        torch.randn(
            (min(batch_size, len(images) - i), *images.shape[1:]),
            device=device,
            dtype=dtype,
        )
        for i in range(0, len(images), batch_size)
    ]

    def run() -> None:
        for batch in batches:
            infer_batch(models, batch).argmax(dim=1)

    return run


def build_scenario(
    name: str,
    images: np.ndarray,
    paths: list[Path],
    output_dir: Path,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Callable[[], object]:
    common = {
        "batch_size": batch_size,
        "inference_device": device,
        "inference_dtype": dtype,
    }
    if name == "gpu_floor":
        return gpu_floor(images, batch_size, device, dtype)
    if name == "array_float32":
        data = images.astype(np.float32)
        return lambda: predict_from_batch(data, **common)
    if name == "array_uint16":
        return lambda: predict_from_batch(images, **common)
    if name == "list_uint16":
        data = list(images)
        return lambda: predict_from_batch(data, **common)
    if name == "tensor_on_device":
        data = torch.from_numpy(images.astype(np.float32)).to(device)
        return lambda: predict_from_batch(data, **common)
    if name == "files_uint16":
        return lambda: predict_from_batch(paths, load_func=load_tif, **common)
    if name == "confidence_uint16":
        return lambda: predict_from_batch(images, export_confidence=True, **common)
    if name == "loop_default":
        return lambda: [
            predict_from_array(image, inference_device=device, inference_dtype=dtype)
            for image in images
        ]
    if name == "loop_tuned":
        return lambda: [
            predict_from_array(
                image,
                patch_size=images.shape[-1],
                patch_overlap=0,
                inference_device=device,
                inference_dtype=dtype,
            )
            for image in images
        ]
    if name == "export_uint16":
        return lambda: predict_from_batch(
            paths,
            load_func=load_tif,
            export_to_disk=True,
            output_dir=output_dir,
            **common,
        )
    raise ValueError(f"Unknown scenario {name}")


def time_scenario(run: Callable[[], object], repeats: int, device: torch.device):
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    run()  # warm up
    synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        synchronize()
        times.append(time.perf_counter() - start)
    return times


def run_benchmarks(args: argparse.Namespace) -> dict:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = get_torch_dtype(args.dtype)
    images = make_images(args.count, args.size)
    results = {
        "config": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "count": args.count,
            "size": args.size,
            "repeats": args.repeats,
            "torch": torch.__version__,
        },
        "scenarios": {},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        paths = (
            write_tifs(images, tmp_dir)
            if any(s in args.scenarios for s in ("files_uint16", "export_uint16"))
            else []
        )

        for name in args.scenarios:
            run = build_scenario(
                name,
                images,
                paths,
                tmp_dir / "out",
                args.batch_size,
                device,
                dtype,
            )
            times = time_scenario(run, args.repeats, device)
            median = statistics.median(times)
            results["scenarios"][name] = {
                "times": times,
                "median": median,
                "min": min(times),
                "max": max(times),
                "images_per_second": args.count / median,
            }
            print(
                f"{name:18} median {median:6.3f}s  "
                f"min {min(times):6.3f}s  max {max(times):6.3f}s  "
                f"{args.count / median:7.1f} img/s",
                flush=True,
            )
    return results


def compare(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text())
    after = json.loads(after_path.read_text())
    print(f"{'scenario':18} {'before':>8} {'after':>8} {'change':>8}  noise")
    for name, result in after["scenarios"].items():
        if name not in before["scenarios"]:
            continue
        old = before["scenarios"][name]
        change = result["median"] / old["median"] - 1
        # spread of each run relative to its median, to judge whether a change is real
        noise = max(
            (old["max"] - old["min"]) / old["median"],
            (result["max"] - result["min"]) / result["median"],
        )
        print(
            f"{name:18} {old['median']:7.3f}s {result['median']:7.3f}s "
            f"{change:+7.1%}  ±{noise / 2:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--compare", type=Path, nargs=2, default=None)
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    results = run_benchmarks(args)
    if args.save:
        args.save.write_text(json.dumps(results, indent=2))
        print(f"Saved results to {args.save}")


if __name__ == "__main__":
    main()
