# Benchmarks

Inference time for a square scene at various sizes.
Results show mean seconds over multiple runs.
Batch size was selected automatically by searching for the fastest value on each device.

![Benchmark plot](_static/benchmark.png)

### Apple M4 Pro (MPS)

OmniCloudMask 1.7.1 &middot; Darwin 25.2.0 &middot; 64.0 GB RAM

| Scene size | CPU | MPS float32 | MPS float16 | Batch size |
| --- | --- | --- | --- | --- |
| **50×50 px** | 0.105s | 0.039s | 0.036s | 1 |
| **100×100 px** | 0.147s | 0.040s | 0.036s | 1 |
| **200×200 px** | 0.162s | 0.036s | 0.036s | 1 |
| **300×300 px** | 0.210s | 0.039s | 0.040s | 1 |
| **400×400 px** | 0.237s | 0.042s | 0.041s | 1 |
| **500×500 px** | 0.338s | 0.046s | 0.045s | 1 |
| **750×750 px** | 0.523s | 0.077s | 0.064s | 1 |
| **1000×1000 px** | 0.785s | 0.119s | 0.099s | 1 |
| **2000×2000 px** | 6.896s | 0.946s | 0.725s | 2 |
| **3000×3000 px** | 14.408s | 1.665s | 1.278s | 4 |
| **5000×5000 px** | — | 4.758s | 3.772s | 4 |
| **10000×10000 px** | — | 18.750s | 14.915s | 2 |

### NVIDIA GeForce RTX 4090, AMD Ryzen 9 5950X 16-Core Processor

OmniCloudMask 1.7.1 &middot; Linux 6.8.0-106-generic &middot; 125.7 GB RAM

| Scene size | CPU | CUDA float32 | CUDA float16 | Batch size |
| --- | --- | --- | --- | --- |
| **100×100 px** | 0.030s | 0.013s | 0.014s | 1 |
| **250×250 px** | 0.070s | 0.014s | 0.015s | 1 |
| **500×500 px** | 0.200s | 0.017s | 0.018s | 1 |
| **1000×1000 px** | 1.133s | 0.038s | 0.036s | 1 |
| **2000×2000 px** | — | 0.216s | 0.169s | 2 |
| **3000×3000 px** | — | 0.389s | 0.283s | 4 |
| **5000×5000 px** | — | 1.065s | 0.869s | 4 |

---

_To add results for your hardware, run `benchmarking/benchmarking.ipynb` and submit the JSON file in `benchmarking/results/` via a pull request._
