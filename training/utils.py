import math
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastai.torch_core import TensorImage, TensorMask
from fastai.vision.augment import DisplayedTransform, RandTransform
from torch import Tensor
from torchvision.transforms.functional import adjust_sharpness


class BatchResample(RandTransform):
    """
    Randomly resample images and masks to different scales using a plateau distribution.

    This transform applies random scaling to entire batches, where the scale factor is sampled
    from a three-zone plateau distribution that allows fine control over scale bias:
    - Linear fade-in from min_scale to plateau_min
    - Uniform sampling from plateau_min to plateau_max
    - Linear fade-out from plateau_max to max_scale

    The plateau distribution allows you to bias sampling towards specific scale ranges with
    linear probability transitions at the boundaries. When plateau bounds equal the min/max
    scales, the distribution becomes uniform (default behaviour).

    Images are resampled using random interpolation modes (bilinear/nearest) with randomly
    applied antialiasing, whilst masks use nearest neighbour to preserve discrete values.

    Probability Density

         │       ┌────────────┐
         │      ╱              ╲
         │    ╱  │            │  ╲
         │  ╱                      ╲
         │╱      │            │      ╲
         └────────────────────────────── Scale Factor
         │       │            │       │
    min_scale plat_min    plat_max  max_scale

    """

    order = 2
    split_idx = 0  # only apply to the training set

    def __init__(
        self,
        p: float = 1.0,
        min_scale=0.2,
        max_scale=1.111,
        plateau_min=None,
        plateau_max=None,
    ):
        super().__init__(p=p)
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._image_modes = [
            "bilinear",
            "nearest",
        ]
        self._antialias_modes = {"bilinear", "bicubic"}
        if plateau_min is None:
            plateau_min = min_scale
        if plateau_max is None:
            plateau_max = max_scale
        self.plateau_min = plateau_min
        self.plateau_max = plateau_max

    def _select_scale_factor(self) -> float:
        """Sample from plateau distribution: linear fade-in, uniform plateau, linear fade-out"""

        # Calculate ranges and areas
        lower_range = self.plateau_min - self.min_scale
        plateau_range = self.plateau_max - self.plateau_min
        upper_range = self.max_scale - self.plateau_max

        lower_area = lower_range / 2
        plateau_area = plateau_range
        upper_area = upper_range / 2

        total_area = lower_area + plateau_area + upper_area

        # Sample zone
        rand = random.random() * total_area

        if rand < lower_area:
            # Lower triangle
            u = random.random()
            return self.min_scale + lower_range * np.sqrt(u)
        elif rand < lower_area + plateau_area:
            # Plateau
            return random.uniform(self.plateau_min, self.plateau_max)
        else:
            # Upper triangle
            u = random.random()
            return self.max_scale - upper_range * np.sqrt(u)

    def before_call(self, batch: Tuple[TensorImage, TensorMask], split_idx: int):
        """Determine the target size before processing the batch"""
        original_size = batch[0].shape[-1]
        scale_factor = self._select_scale_factor()
        self.target_size = round(original_size * scale_factor)

    def _resample_image(self, image: TensorImage) -> TensorImage:
        """Resample image with random interpolation mode and antialiasing"""
        interpolation_mode = random.choice(self._image_modes)

        # Randomly apply antialiasing for modes that support it
        use_antialiasing = (
            interpolation_mode in self._antialias_modes and random.choice([True, False])
        )

        return F.interpolate(
            image,
            size=(self.target_size, self.target_size),
            mode=interpolation_mode,
            antialias=use_antialiasing,
        )

    def _resample_mask(self, mask: TensorMask) -> TensorMask:
        """Resample mask using nearest neighbour to preserve discrete values"""
        # Add batch dimension, interpolate, then remove batch dimension
        resampled = F.interpolate(
            mask.unsqueeze(0), size=(self.target_size, self.target_size), mode="nearest"
        )
        return resampled.squeeze(0)

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        """Apply appropriate resampling based on input type"""
        if isinstance(x, TensorImage):
            return self._resample_image(x)
        elif isinstance(x, TensorMask):
            return self._resample_mask(x)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")


class RandomClipLargeImages(RandTransform):
    """
    Randomly crop batches of images and masks to a smaller size when they exceed the target dimensions.

    This transform applies random cropping to entire batches where images are larger than the desired
    output size. It first selects a random crop size between min_size and max_size, then chooses a
    random location within the image to extract the crop from. If the input images are smaller than
    the target crop size, no cropping is performed and the original batch is returned unchanged.

    The same crop size and coordinates are applied to all items in the batch to maintain spatial
    correspondence between images and masks. This transform is particularly useful after resampling
    operations that may produce varying image sizes, and helps speed up training by reducing the
    size of larger images.

    Cropping behaviour:

    ┌───────────────┐
    │  ┌──────────┐ │
    │  │ Random   │ │
    │  │ Crop     │ │
    │  │ Location │ │
    │  │          │ │
    │  └──────────┘ │
    └───────────────┘
    """

    order = 3  # after resampling
    split_idx = 0  # only apply to the training set

    def __init__(self, p: float = 1.0, min_size: int = 256, max_size: int = 256):
        super().__init__(p=p)

        self.min_size = min_size
        self.max_size = max_size

    def before_call(self, b: Tuple[TensorImage, TensorMask], split_idx: int):
        image_size = b[0].shape[-1]

        new_size = random.randint(self.min_size, self.max_size)
        self.new_size = new_size

        if image_size < self.new_size:
            self.clip_x = 0
            self.clip_y = 0
        else:
            clip_max = image_size - self.new_size
            self.clip_x = random.randint(0, clip_max)
            self.clip_y = random.randint(0, clip_max)

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        if isinstance(x, TensorImage):
            clipped_images = x[
                :,
                :,
                self.clip_x : self.clip_x + self.new_size,
                self.clip_y : self.clip_y + self.new_size,
            ]

            return TensorImage(clipped_images)
        if isinstance(x, TensorMask):
            clipped_masks = x[
                :,
                self.clip_x : self.clip_x + self.new_size,
                self.clip_y : self.clip_y + self.new_size,
            ]
            return TensorMask(clipped_masks)


class BatchRot90(RandTransform):
    """
    Randomly rotate entire batches of images and masks by 0, 90, 180, or 270 degrees.

    This transform applies random 90-degree rotations to batches, where all items in the batch
    receive the same rotation to maintain spatial correspondence between images and masks.
    The rotation is applied around the centre of the image using PyTorch's rot90 function.

    0° (Original):      90° Rotated:        180° Rotated:       270° Rotated:
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │███████       │   │████ ████ ████│   │██████████████│   │          ████│
    │███████       │   │████ ████ ████│   │██████████████│   │          ████│
    │███████████   │   │████ ████ ████│   │   ███████████│   │     ████ ████│
    │███████████   │   │████ ████     │   │   ███████████│   │████ ████ ████│
    │██████████████│   │████          │   │       ███████│   │████ ████ ████│
    │██████████████│   │████          │   │       ███████│   │████ ████ ████│
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
    """

    order = 4
    split_idx = 0  # only apply to the training set

    def __init__(self, p: float = 1.0):
        self.p = p
        super().__init__(p=p)

    def before_call(self, b: Tuple[TensorImage, TensorMask], split_idx: int):
        if random.random() < self.p:
            self.rot = random.choice([0, 1, 2, 3])
        else:
            self.rot = 0

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        return type(x)(x.rot90(self.rot, [-2, -1]))


class RandomRectangle(RandTransform):
    """
    Randomly erases rectangular regions in images by filling them with random values.

    This transform implements a variant of Random Erasing augmentation that helps improve
    model robustness by forcing it to rely on multiple features rather than specific regions.
    It randomly selects rectangular areas in the image and fills them with random pixel
    values, simulating occlusion or missing data.

    The transform can create multiple rectangles per image, with configurable size ranges,
    aspect ratios, and fill values. Only a random subset of colour channels are affected
    per rectangle.d
    Original Image:          After RandRect Effect:
    ┌──────────────┐         ┌──────────────┐
    │██████████████│         │██████████████│
    │██████████████│         │███####███████│
    │██████████████│ ----->  │███####███████│
    │██████████████│         │███####███████│
    │██████████████│         │██████████████│
    │██████████████│         │██████████████│
    └──────────────┘         └──────────────┘
    """

    order = 100
    split_idx = 0  # only apply to the training set

    def __init__(
        self,
        p: float = 0.1,  # Probability of appying Random Erasing
        sl: float = 0.0,  # Minimum proportion of erased area
        sh: float = 0.3,  # Maximum proportion of erased area
        min_aspect: float = 0.3,  # Minimum aspect ratio of erased area
        max_count: int = 1,  # Maximum number of erasing blocks per image, area per box is scaled by count
        max_fill_value: int = 10000,  # Maximum value to fill in the erased area
    ):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.min_aspect = min_aspect
        self.max_count = max_count
        self.max_fill_value = max_fill_value
        super().__init__(p=p)
        self.log_ratio = (math.log(min_aspect), math.log(1 / min_aspect))

    def cutout_values(
        self,
        x: TensorImage,  # Input image
        areas: list,  # List of areas to cutout. Order rl,rh,cl,ch
        value: float,  # Value to set in the cutout areas
    ):
        "Replace all `areas` in `x` with `value` in some or all channels."
        chan = x.shape[-3]
        value_chan_count = random.randint(
            1, chan
        )  # Determine how many channels to modify
        value_chans = random.sample(
            range(chan), value_chan_count
        )  # Select random channels
        for rl, rh, cl, ch in areas:
            for c in value_chans:
                x[:, c, rl:rh, cl:ch] = (
                    value  # Set the specified area to the given value
                )
        return x

    def _slice(self, area, sz: int) -> Tuple[int, int]:
        bound = int(round(math.sqrt(area)))
        loc = random.randint(0, max(sz - bound, 0))
        return loc, loc + bound

    def _bounds(self, area, img_h, img_w):
        r_area = random.uniform(self.sl, self.sh) * area
        aspect = math.exp(random.uniform(*self.log_ratio))
        return self._slice(r_area * aspect, img_h) + self._slice(r_area / aspect, img_w)

    def encodes(self, x: TensorImage):
        count = random.randint(1, self.max_count)
        _, img_h, img_w = x.shape[-3:]
        area = img_h * img_w / count
        areas = [self._bounds(area, img_h, img_w) for _ in range(count)]
        return self.cutout_values(x, areas, random.randint(0, self.max_fill_value))


class DynamicZScoreNormalize(DisplayedTransform):
    """
    Dynamically normalize images using Z-score normalization on non-no-data pixels only.

    This transform applies per-channel Z-score normalization (mean=0, std=1) but only
    considers pixels that are not equal to the no_data_value when calculating statistics.

    The normalization is computed independently for each channel and each image in the
    batch, making it adaptive to the content and dynamic range of individual images.
    No-data pixels are set to zero after normalization.
    """

    order = 101

    def __init__(self, no_data_value: float = 0.0):
        super().__init__(split_idx=None)
        self.no_data_value = no_data_value

    def encodes(self, x: TensorImage):
        # Mask for non-zero elements
        mask = x != self.no_data_value

        # Calculate mean and std only on non-zero elements, keeping channel dimension
        mean = (x * mask).sum(dim=(2, 3)) / mask.sum(dim=(2, 3))
        std = torch.sqrt(
            ((x - mean[:, :, None, None]) ** 2 * mask).sum(dim=(2, 3))
            / mask.sum(dim=(2, 3))
        )
        epsilon = 1e-8  # Small value to prevent division by zero
        normalized_tensor = torch.where(
            mask, (x - mean[:, :, None, None]) / (std[:, :, None, None] + epsilon), x
        )

        # Replace zero values with the channel's mean (after normalization, mean is 0 for non-zero elements)
        normalized_tensor = torch.where(
            mask, normalized_tensor, torch.zeros_like(normalized_tensor)
        )

        return normalized_tensor


class SceneEdge(RandTransform):
    """
    Simulates scene edges by adding random linear no-data regions to images.

    This transform mimics the irregular boundaries found in satellite imagery
    where scenes have natural edges due to acquisition geometry.

    The transform generates a random line that divides the image and sets all pixels
    on one side of that line to zero.

    Original Image:          After Scene Edge Effect:
    ┌──────────────┐         ┌──────────────┐
    │██████████████│         │██████████████│
    │██████████████│         │██████████████│
    │██████████████│ ----->  │██████████████│
    │██████████████│         │████████████xx│
    │██████████████│         │██████████xxxx│
    │██████████████│         │████████xxxxxx│
    └──────────────┘         └──────────────┘

    """

    order = 99
    split_idx = 0

    def __init__(self, p: float = 0.5):
        # Initialize with the probability of applying the mask
        self.p = p

    def add_zero_sliver(self, image: Tensor) -> Tensor:
        C, H, W = image.shape
        x0, y0 = np.random.uniform(0, W), np.random.uniform(0, H)
        angle = np.random.uniform(0, 2 * np.pi)
        dx, dy = np.cos(angle), np.sin(angle)
        a, b = -dy, dx
        c = -(a * x0 + b * y0)
        xx, yy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
        mask = a * xx.float() + b * yy.float() + c > 0
        mask = mask.unsqueeze(0).repeat(C, 1, 1)
        image_out = image.clone()
        image_out[mask] = 0
        return image_out

    def encodes(self, batch: TensorImage) -> TensorImage:
        "Applies the transform to each image in the batch with probability p"
        batch = batch.clone()  # Clone to avoid modifying the original batch
        for i in range(batch.size(0)):
            if torch.rand(1).item() < self.p:  # Check if the mask should be applied
                batch[i] = self.add_zero_sliver(batch[i])
        return batch


class BatchTear(RandTransform):
    """
    Creates a 'tear' effect in images by applying local displacement along a random line.

    This transform simulates visual distortions that can occur in imagery due to sensor
    movement or processing artifacts. It creates a realistic tearing effect by displacing
    pixels along a randomly oriented line through the image.

    The transform works by generating a displacement map based on distance from a random
    line, then shifting pixels in the direction perpendicular to that line. The result
    is a localized distortion that looks like the image has been "torn" and slightly
    offset along the tear line.

    Original Image:          After Tear Effect:
    ┌──────────────┐         ┌──────────────┐
    │██████████████│         │█████████████ │
    │              │         │             █│
    │██████████████│ ----->  │███████████   │
    │              │         │           ███│
    │██████████████│         │█████████     │
    │              │         │         █████│
    └──────────────┘         └──────────────┘
    """

    split_idx = 0
    order = 5

    def __init__(self, p: float = 1.0, displacement: int = 10):
        super().__init__(p=p)
        self.p = p
        self.displacement = displacement

    def before_call(self, b, split_idx):
        if random.random() < self.p:
            self.do = True
            self.H, self.W = b[0].shape[-2], b[0].shape[-1]
            self.angle = np.random.uniform(0, 180)
            self.x0, self.y0 = (
                np.random.uniform(0, self.W),
                np.random.uniform(0, self.H),
            )
        else:
            self.do = False

    def create_tear_augmentation(
        self,
        image_tensor: TensorImage | TensorMask,
        angle: float,
        x0: float,
        y0: float,
        displacement: int = 5,
    ) -> TensorImage | TensorMask:
        """Create a tear augmentation on the input image tensor."""
        mask_tensor = isinstance(image_tensor, TensorMask)
        if mask_tensor:
            image_tensor = TensorMask(image_tensor.unsqueeze(1))

        _, C, H, W = image_tensor.shape

        angle_rad = np.deg2rad(angle)

        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)

        xx, yy = torch.meshgrid(
            torch.arange(W, device=image_tensor.device),
            torch.arange(H, device=image_tensor.device),
            indexing="xy",
        )
        displacement_map = torch.abs(dx * (yy - y0) - dy * (xx - x0))
        displacement_map = torch.clamp(
            displacement_map / displacement_map.max() * displacement, 0, displacement
        )
        # Using advanced indexing to avoid explicit looping over channels
        channels = [
            torch.roll(
                image_tensor[:, c],
                shifts=(int(dy * displacement), int(dx * displacement)),
                dims=(1, 2),
            )
            for c in range(C)
        ]
        displaced_image = torch.stack(channels, dim=1)

        mask = displacement_map.unsqueeze(0) > (displacement / 2)
        displaced_image = torch.where(mask, displaced_image, image_tensor)

        # Adjust cropping to minimize information loss
        crop_size = int(displacement)
        displaced_image = displaced_image[
            :, :, crop_size:-crop_size, crop_size:-crop_size
        ]

        displaced_image = torch.nn.functional.interpolate(displaced_image, size=(H, W))
        if mask_tensor:
            displaced_image = displaced_image.squeeze(1)

        return displaced_image

    def encodes(self, x: TensorImage | TensorMask):
        if not self.do:
            return x
        return self.create_tear_augmentation(
            x, self.angle, self.x0, self.y0, self.displacement
        )


class BatchFlip(RandTransform):
    """
    Randomly flip images and masks horizontally and/or vertically.

    This transform applies random flipping augmentation to entire batches, where all items
    receive the same flip operations to maintain spatial correspondence between images and
    masks.

    The transform can apply horizontal flips, vertical flips, both, or neither, depending
    on the configuration and random selection.

    Original Image:     Horizontal Flipped: Vertical Flipped:   Both Flipped:
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │███████       │   │██████████████│   │       ███████│   │██████████████│
    │███████       │   │██████████████│   │       ███████│   │██████████████│
    │███████████   │   │███████████   │   │   ███████████│   │   ███████████│
    │███████████   │   │███████████   │   │   ███████████│   │   ███████████│
    │██████████████│   │███████       │   │██████████████│   │       ███████│
    │██████████████│   │███████       │   │██████████████│   │       ███████│
    └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
    """

    split_idx = 0  # Apply only on the training set
    order = 5  # Order of application

    def __init__(self, p: float = 1.0, flip_vert: bool = True, flip_horiz: bool = True):
        """
        p: Probability of applying the transform.
        flip_vert: Whether to allow vertical flips.
        flip_horiz: Whether to allow horizontal flips.
        """
        super().__init__(p=p)
        self.flip_vert = flip_vert
        self.flip_horiz = flip_horiz
        self.p = p

    def before_call(self, b, split_idx):
        "Decide randomly whether to flip vertically and/or horizontally."
        if random.random() < self.p:
            self.do = True
            self.do_horiz = self.flip_horiz and random.choice([True, False])
            self.do_vert = self.flip_vert and random.choice([True, False])
        else:
            self.do = False

    def encodes(self, x: TensorImage | TensorMask) -> TensorImage | TensorMask:
        "Apply the selected flips to the image or mask tensor."
        if not self.do:
            return x
        # Horizontal flip: flip along the width dimension (last dimension)
        if self.do_horiz:
            x = type(x)(torch.flip(x, dims=[-1]))
        # Vertical flip: flip along the height dimension (second-to-last dimension)
        if self.do_vert:
            x = type(x)(torch.flip(x, dims=[-2]))
        return x


class ClipHighAndLow(RandTransform):
    """
    Simulates sensor limitations by randomly clipping pixel values at high and low extremes.

    This transform mimics real-world sensor behaviour where very bright areas cause saturation
    (clipping high values) and very dark areas fall below the noise floor or sensor sensitivity
    threshold (clipping low values). Both effects are applied independently and randomly
    to each band of each image in the batch.

    The clipping is applied as a percentage of each band's dynamic range, making it
    adaptive to the actual data distribution.
    """

    order = 50
    split_idx = 0

    def __init__(self, p: float = 0.1, max_pct: float = 0.1):
        self.p = p
        self.max_pct = max_pct
        super().__init__(p=p)

    def encodes(self, x: TensorImage):

        for image_num, image in enumerate(x):
            for band_num, band in enumerate(image):
                band_min, band_max = band.min(), band.max()
                band_range = band_max - band_min

                # High clipping (sensor saturation)
                if random.random() < self.p:
                    clip_pct = random.uniform(0, self.max_pct)
                    new_max = band_max - (band_range * clip_pct)
                    band = torch.clip(band, band_min, new_max)

                # Low clipping (noise floor/poor sensitivity)
                if random.random() < self.p:
                    clip_pct = random.uniform(0, self.max_pct)
                    new_min = band_min + (band_range * clip_pct)
                    band = torch.clip(band, new_min, band_max)

                x[image_num][band_num] = band

        return x


class RandomSharpenBlur(RandTransform):
    """
    Randomly applies sharpening or blurring effects to images with varying intensity.

    This transform simulates different image acquisition conditions such as camera focus
    variations, atmospheric effects, motion blur, or post-processing artifacts. It can
    both sharpen images (factor > 1.0) to simulate enhanced edge definition, or blur
    images (factor < 1.0) to simulate out-of-focus conditions or atmospheric distortion.

    The transform uses PyTorch's adjust_sharpness function and preserves the original
    dynamic range of each image by normalizing to [0,1] during processing, then
    rescaling back to the original min/max values.
    """

    order = 3
    split_idx = 0

    def __init__(
        self,
        p: float = 1.0,
        min_factor: float = 0.0,
        max_factor: float = 2.0,
        per_sample_probability: float = 0.1,
    ):
        super().__init__(p=p)
        self.min_factor: float = min_factor
        self.max_factor: float = max_factor
        self.per_sample_probability: float = per_sample_probability

    def encodes(self, x: TensorImage) -> TensorImage:
        x_blur_sharpen = x.clone()
        for idx, image in enumerate(x_blur_sharpen):
            if random.random() < self.per_sample_probability:
                sharpness_factor = random.uniform(self.min_factor, self.max_factor)
                image_min = image.min()
                image_max = image.max()
                # Normalize the image to [0, 1] range for sharpening
                image = (image - image_min) / (image_max - image_min + 1e-8)
                image = adjust_sharpness(image, sharpness_factor)
                # Rescale back to original range
                image = image * (image_max - image_min) + image_min

            x_blur_sharpen[idx] = image

        return x_blur_sharpen
