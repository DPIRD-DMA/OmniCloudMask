# Benchmarks

To add results for your hardware, run [`benchmarking/benchmarking.ipynb`](https://github.com/DPIRD-DMA/OmniCloudMask/blob/main/benchmarking/benchmarking.ipynb) and submit the JSON file in `benchmarking/results/` via a pull request.

Inference time for a square scene at various sizes.
Results show mean seconds over multiple runs.
Batch size was selected automatically by searching for the fastest value on each device.

![Benchmark plot](_static/benchmark.png)

## NVIDIA GeForce RTX 4090

OmniCloudMask 1.7.1 &middot; Linux 6.8.0-106-generic &middot; 125.7 GB RAM

| Scene size | fp32 | Batch (fp32) | fp16 | Batch (fp16) |
| --- | --- | --- | --- | --- |
| **50×50 px** | 0.013s | 1 | 0.015s | 1 |
| **100×100 px** | 0.013s | 1 | 0.015s | 1 |
| **200×200 px** | 0.015s | 1 | 0.016s | 1 |
| **300×300 px** | 0.016s | 1 | 0.016s | 1 |
| **400×400 px** | 0.016s | 1 | 0.016s | 1 |
| **500×500 px** | 0.018s | 1 | 0.020s | 1 |
| **750×750 px** | 0.029s | 1 | 0.023s | 1 |
| **1000×1000 px** | 0.042s | 1 | 0.036s | 1 |
| **2000×2000 px** | 0.211s | 2 | 0.164s | 4 |
| **3000×3000 px** | 0.404s | 4 | 0.293s | 4 |
| **5000×5000 px** | 1.084s | 4 | 0.846s | 4 |
| **10000×10000 px** | 4.106s | 4 | 3.136s | 4 |

## Apple M4 Pro (MPS)

OmniCloudMask 1.7.1 &middot; Darwin 25.2.0 &middot; 64.0 GB RAM

| Scene size | fp32 | Batch (fp32) | fp16 | Batch (fp16) |
| --- | --- | --- | --- | --- |
| **50×50 px** | 0.039s | 1 | 0.036s | 1 |
| **100×100 px** | 0.040s | 1 | 0.036s | 1 |
| **200×200 px** | 0.036s | 1 | 0.036s | 1 |
| **300×300 px** | 0.039s | 1 | 0.040s | 1 |
| **400×400 px** | 0.042s | 1 | 0.041s | 1 |
| **500×500 px** | 0.046s | 1 | 0.045s | 1 |
| **750×750 px** | 0.077s | 1 | 0.064s | 1 |
| **1000×1000 px** | 0.119s | 1 | 0.099s | 1 |
| **2000×2000 px** | 0.946s | 2 | 0.725s | 2 |
| **3000×3000 px** | 1.665s | 4 | 1.278s | 2 |
| **5000×5000 px** | 4.758s | 4 | 3.772s | 1 |
| **10000×10000 px** | 18.750s | 2 | 14.915s | 1 |

## Apple M4 Pro (CPU)

OmniCloudMask 1.7.1 &middot; Darwin 25.2.0 &middot; 64.0 GB RAM

| Scene size | fp32 | Batch |
| --- | --- | --- |
| **50×50 px** | 0.105s | 1 |
| **100×100 px** | 0.147s | 1 |
| **200×200 px** | 0.162s | 1 |
| **300×300 px** | 0.210s | 1 |
| **400×400 px** | 0.237s | 1 |
| **500×500 px** | 0.338s | 1 |
| **750×750 px** | 0.523s | 1 |
| **1000×1000 px** | 0.785s | 1 |
| **2000×2000 px** | 6.896s | 1 |
| **3000×3000 px** | 14.408s | 1 |

## AMD Ryzen 9 5950X

OmniCloudMask 1.7.1 &middot; Linux 6.8.0-106-generic &middot; 125.7 GB RAM

| Scene size | fp32 | Batch |
| --- | --- | --- |
| **50×50 px** | 0.026s | 1 |
| **100×100 px** | 0.046s | 1 |
| **200×200 px** | 0.054s | 1 |
| **300×300 px** | 0.095s | 1 |
| **400×400 px** | 0.153s | 1 |
| **500×500 px** | 0.226s | 1 |
| **750×750 px** | 0.614s | 1 |
| **1000×1000 px** | 1.233s | 1 |
| **2000×2000 px** | 10.726s | 1 |
| **3000×3000 px** | 17.872s | 1 |
