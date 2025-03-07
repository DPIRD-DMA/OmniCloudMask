# OmniCloudMask Docker with Jupyter

This directory contains the Docker configuration for running OmniCloudMask with Jupyter Lab and GPU support.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

## Building Docker Image from local Dockerfile

```bash
# Clone the repository (if not already done)
git clone https://github.com/DPIRD-DMA/OmniCloudMask
cd OmniCloudMask

# Build the image locally
docker build -f docker/Dockerfile -t omnicloudmask:local .
```

## Running the Container

### Basic Usage (CPU Only)

```bash
docker run -p 127.0.0.1:8888:8888 omnicloudmask:local
```

### With GPU Support

```bash
docker run --gpus all -p 127.0.0.1:8888:8888 omnicloudmask:local
```

### Binding a directory to access data on host machine

```bash
docker run --gpus all -p 127.0.0.1:8888:8888 -v /path/to/your/data:/workspace/data omnicloudmask:local
```

## Accessing Jupyter Lab

Once the container is running, access Jupyter Lab by opening your browser to:

```
http://localhost:8888
```

No password or token is required by default.

## Example Notebooks

The container includes example notebooks located in the default directory:

- Sentinel-2 example
- Maxar example

## Troubleshooting

### GPU Issues

If you encounter GPU-related issues, ensure:
1. NVIDIA drivers are properly installed on your host
2. NVIDIA Container Toolkit is correctly installed

### Port Conflicts

If port 8888 is already in use, change the host port mapping:
```bash
docker run -p 127.0.0.1:9999:8888 omnicloudmask:local
```
Then access Jupyter Lab at http://localhost:9999