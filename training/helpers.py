import math
import platform
from typing import Optional

import fastai
import numpy as np
import torch
from fastai.torch_core import default_device
from matplotlib import pyplot as plt
from torch import version


def plot_batch(batch, image_num=0, labels: Optional[list[str]] = None):
    # Load one batch of data
    x, y = batch  # x: images, y: masks

    # Print the shape of the image tensor and the mask
    print(f"Image tensor shape: {x.shape}")
    print(f"Label shape: {y.shape}")

    _, channels, _, _ = x.shape

    if image_num >= x.shape[0]:
        print(f"Image number {image_num} is out of range for this batch.")
        return

    # Ensure there are at least 3 channels to form an RGB image
    if channels < 3:
        print("There are less than 3 channels available. Cannot form an RGB image.")
        return

    # Extract the first three channels to form an RGB image
    rgb_img = x[image_num, :3].cpu().numpy()
    rgb_img = np.transpose(
        rgb_img, (1, 2, 0)
    )  # Rearrange dimensions to height x width x channels
    rgb_img = (rgb_img - rgb_img.min()) / (
        rgb_img.max() - rgb_img.min()
    )  # Normalize to [0, 1] for displaying

    fig, axs = plt.subplots(1, channels + 2, figsize=(25, 10))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    # Display the RGB image
    axs[0].imshow(rgb_img)
    if labels:
        axs[0].set_title(labels[0])
    else:
        axs[0].set_title("RGB Image")
    axs[0].axis("off")

    # Plot each channel for the specified image number and print mean and std
    for ch in range(channels):
        img_channel = x[image_num, ch].cpu().numpy()

        axs[ch + 1].imshow(img_channel, cmap="gray")
        if labels:
            axs[ch + 1].set_title(labels[ch + 1])
        else:
            axs[ch + 1].set_title(f"Channel {ch + 1}")
        axs[ch + 1].axis("off")

    # Plot the label mask for the specified image number
    axs[-1].imshow(
        y[image_num].cpu().numpy(),
        cmap="tab20b",
        interpolation="nearest",
        vmin=0,
        vmax=3,
    )
    axs[-1].set_title("Label")
    axs[-1].axis("off")

    plt.show()


def show_histo(batch, image_num=0, labels: Optional[list[str]] = None):
    tensor = batch[0][image_num].cpu()
    num_channels = tensor.shape[0]

    # Calculate the number of rows and columns for subplots
    # aiming for a square-ish layout
    num_rows = math.ceil(math.sqrt(num_channels))
    num_cols = math.ceil(num_channels / num_rows)

    # Create a figure with dynamic subplots
    _, axs = plt.subplots(num_rows, num_cols, figsize=(10, 8))

    # Ensure axs is always a 2D array for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1 or num_cols == 1:
        axs = axs.reshape(num_rows, num_cols)

    # Flatten each band and plot its histogram
    for i in range(num_channels):
        values = tensor[i].flatten().numpy()  # Convert to NumPy array for plotting
        row, col = divmod(i, num_cols)

        ax = axs[row, col]
        ax.hist(values, bins=100, alpha=0.75)
        if labels:
            ax.set_title(labels[i])
        else:
            ax.set_title(f"Band {i + 1} Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        ax.set_xlim(-2, 2)

    # Adjust layout to prevent overlap and hide empty subplots
    for ax in axs.flatten()[num_channels:]:
        ax.set_visible(False)  # Hide unused subplots

    plt.tight_layout()
    plt.show()


def print_system_info():
    # Gather information
    info = {
        "PyTorch Version": torch.__version__,
        "CUDA Available": "Yes" if torch.cuda.is_available() else "No",
        "CUDA Version": version.cuda if torch.cuda.is_available() else "N/A",
        "Python Version": platform.python_version(),
        "Fastai Version": fastai.__version__,
        "Default Device": default_device(),
    }

    # Find the maximum key length for alignment
    max_key_length = max(len(key) for key in info.keys())

    # Print the table
    print("System Information")
    print("-" * (max_key_length + 20))  # Adjusting based on expected value lengths
    for key, value in info.items():
        print(f"{key.ljust(max_key_length)} : {value}")
    print("-" * (max_key_length + 20))
