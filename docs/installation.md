# Installation

OmniCloudMask requires Python 3.9 or higher.

## Install from PyPI

```bash
pip install omnicloudmask
```

Or using [uv](https://docs.astral.sh/uv/):

```bash
uv add omnicloudmask
```

## Install from conda-forge

```bash
conda install conda-forge::omnicloudmask
```

## Install from source

```bash
pip install git+https://github.com/DPIRD-DMA/OmniCloudMask.git
```

## Docker

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) installed
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Build the Docker image:

```bash
git clone https://github.com/DPIRD-DMA/OmniCloudMask
cd OmniCloudMask
docker build -f docker/Dockerfile -t omnicloudmask:local .
```

Run with GPU support:

```bash
docker run --gpus all -p 127.0.0.1:8888:8888 omnicloudmask:local
```

Or CPU only:

```bash
docker run -p 127.0.0.1:8888:8888 omnicloudmask:local
```

Mount a data directory:

```bash
docker run --gpus all -p 127.0.0.1:8888:8888 -v /path/to/data:/workspace/data omnicloudmask:local
```

Access Jupyter Lab at http://localhost:8888 (no password required).

## Legacy Model Support

Model versions 1.0 through 3.0 were built with fastai. To use these older models, install the legacy extra:

```bash
pip install omnicloudmask[legacy]
```

Or with uv:

```bash
uv add omnicloudmask --extra legacy
```

Or with conda:

```bash
conda install conda-forge::omnicloudmask conda-forge::fastai
```

See the [model changelog](model-changelog.md) for differences between model versions.
