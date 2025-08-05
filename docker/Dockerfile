# Arguments to pass to the image
ARG VERSION_DATE=24.09
ARG FROM_IMAGE=nvcr.io/nvidia/pytorch

# Import NVIDIA PyTorch container as the base image
FROM ${FROM_IMAGE}:${VERSION_DATE}-py3

# Install OmniCloudMask
RUN pip --no-cache-dir install --upgrade omnicloudmask jupyter matplotlib s2dl h5py

# Clean up default example directories that come with the base image
RUN rm -rf /workspace/docker-examples /workspace/tutorials /workspace/examples /workspace/README.md

# Set up a working directory
WORKDIR /workspace

# Set up a directory for data
RUN mkdir -p /workspace/data

# Copy example notebooks
COPY ["examples/Maxar.ipynb", "/workspace/"]
COPY ["examples/Sentinel-2.ipynb", "/workspace/"]
COPY ["examples/PlanetScope Hyperspectral.ipynb", "/workspace/"]

# Copy README file
COPY docker/README.md /workspace

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

LABEL maintainer="Nick Wright <nicholas.wright@dpird.wa.gov.au>"
LABEL description="OmniCloudMask container with PyTorch and Jupyter"
LABEL version="1.1"