import math
import random
from typing import Tuple

import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as F
from PIL import Image as PILImage

from sam3.train.data.sam3_image_dataset import Datapoint
from sam3.train.transforms import basic_for_api


def _is_pil_image(img) -> bool:
    return isinstance(img, PILImage.Image)


def _to_float_tensor(img: torch.Tensor) -> torch.Tensor:
    if img.dtype == torch.uint8:
        return img.float().div(255.0)
    return img.float()


def _image_to_tensor(img):
    if _is_pil_image(img):
        return F.to_tensor(img), "pil"
    if isinstance(img, torch.Tensor):
        return _to_float_tensor(img), "tensor"
    raise TypeError(f"Unsupported image type: {type(img)}")


def _tensor_to_image(tensor: torch.Tensor, image_type: str):
    tensor = tensor.clamp(0.0, 1.0)
    if image_type == "pil":
        return F.to_pil_image(tensor)
    return tensor


def _sample_uniform(range_tuple: Tuple[float, float]) -> float:
    return random.uniform(range_tuple[0], range_tuple[1])


def _to_list_if_needed(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [float(v) for v in list(value)]
    return value


class ZScoreNormalize:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def _apply(self, img):
        tensor, _ = _image_to_tensor(img)
        mean = tensor.mean()
        std = tensor.std().clamp(min=self.eps)
        return (tensor - mean) / std

    def __call__(self, datapoint: Datapoint, **kwargs):
        for img in datapoint.images:
            img.data = self._apply(img.data)
        return datapoint


class RandomAffineNNUNet:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.2,
        degrees: Tuple[float, float] = (-10.0, 10.0),
        scale: Tuple[float, float] = (0.85, 1.15),
        translate: Tuple[float, float] = (0.03, 0.03),
        shear=None,
        fill_value: float = 0.0,
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.degrees = degrees
        self.scale = scale
        self.translate = translate
        self.shear = shear
        self.fill_value = fill_value

    def __call__(self, datapoint: Datapoint, **kwargs):
        if random.random() >= self.p:
            return datapoint
        degrees = _to_list_if_needed(self.degrees)
        scale = _to_list_if_needed(self.scale)
        translate = _to_list_if_needed(self.translate)
        shear = _to_list_if_needed(self.shear)
        transformer = basic_for_api.RandomAffine(
            degrees=degrees,
            consistent_transform=self.consistent_transform,
            scale=scale,
            translate=translate,
            shear=shear,
            image_mean=(self.fill_value,) * 3,
            image_interpolation="bilinear",
        )
        return transformer(datapoint, **kwargs)


class RandomGaussianNoise:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.1,
        variance_range: Tuple[float, float] = (0.0, 0.02),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.variance_range = variance_range

    def _apply(self, img, variance: float):
        tensor, image_type = _image_to_tensor(img)
        std = math.sqrt(max(variance, 0.0))
        noise = torch.randn_like(tensor) * std
        tensor = (tensor + noise).clamp(0.0, 1.0)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            variance = _sample_uniform(self.variance_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, variance)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                variance = _sample_uniform(self.variance_range)
                img.data = self._apply(img.data, variance)
        return datapoint


class RandomGaussianBlur:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.15,
        sigma_range: Tuple[float, float] = (0.3, 1.0),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.sigma_range = sigma_range

    def _apply(self, img, sigma: float):
        kernel_size = max(3, int(round(sigma * 3)) * 2 + 1)
        tensor, image_type = _image_to_tensor(img)
        tensor = F.gaussian_blur(tensor, kernel_size=kernel_size, sigma=sigma)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            sigma = _sample_uniform(self.sigma_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, sigma)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                sigma = _sample_uniform(self.sigma_range)
                img.data = self._apply(img.data, sigma)
        return datapoint


class RandomMotionBlur:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.05,
        kernel_size_range: Tuple[int, int] = (3, 7),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.kernel_size_range = kernel_size_range

    def _apply(self, img, kernel_size: int, horizontal: bool):
        tensor, image_type = _image_to_tensor(img)
        kernel_size = max(3, int(kernel_size) | 1)
        if horizontal:
            kernel = torch.zeros(1, 1, 1, kernel_size, device=tensor.device)
            kernel[0, 0, 0, :] = 1.0 / kernel_size
            padding = (kernel_size // 2, kernel_size // 2, 0, 0)
        else:
            kernel = torch.zeros(1, 1, kernel_size, 1, device=tensor.device)
            kernel[0, 0, :, 0] = 1.0 / kernel_size
            padding = (0, 0, kernel_size // 2, kernel_size // 2)
        tensor = tensor.unsqueeze(0)
        channels = tensor.shape[1]
        kernel = kernel.repeat(channels, 1, 1, 1)
        tensor = torch_F.pad(tensor, padding, mode="replicate")
        tensor = torch_F.conv2d(tensor, kernel, groups=channels)
        tensor = tensor.squeeze(0)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
            horizontal = random.random() < 0.5
            for img in datapoint.images:
                img.data = self._apply(img.data, kernel_size, horizontal)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                kernel_size = random.randint(self.kernel_size_range[0], self.kernel_size_range[1])
                horizontal = random.random() < 0.5
                img.data = self._apply(img.data, kernel_size, horizontal)
        return datapoint


class RandomMultiplicativeBrightness:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.15,
        multiplier_range: Tuple[float, float] = (0.9, 1.1),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.multiplier_range = multiplier_range

    def _apply(self, img, multiplier: float):
        tensor, image_type = _image_to_tensor(img)
        tensor = (tensor * multiplier).clamp(0.0, 1.0)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            multiplier = _sample_uniform(self.multiplier_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, multiplier)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                multiplier = _sample_uniform(self.multiplier_range)
                img.data = self._apply(img.data, multiplier)
        return datapoint


class RandomContrast:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.15,
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.contrast_range = contrast_range

    def _apply(self, img, contrast: float):
        tensor, image_type = _image_to_tensor(img)
        tensor = F.adjust_contrast(tensor, contrast)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            contrast = _sample_uniform(self.contrast_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, contrast)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                contrast = _sample_uniform(self.contrast_range)
                img.data = self._apply(img.data, contrast)
        return datapoint


class RandomGamma:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.3,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        invert: bool = False,
        retain_stats: bool = True,
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.gamma_range = gamma_range
        self.invert = invert
        self.retain_stats = retain_stats

    def _apply(self, img, gamma: float):
        tensor, image_type = _image_to_tensor(img)
        if self.invert:
            tensor = 1.0 - tensor
        orig_mean = tensor.mean()
        orig_std = tensor.std()
        tensor = F.adjust_gamma(tensor, gamma)
        if self.retain_stats:
            new_mean = tensor.mean()
            new_std = tensor.std().clamp(min=1e-6)
            tensor = (tensor - new_mean) / new_std * orig_std + orig_mean
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            gamma = _sample_uniform(self.gamma_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, gamma)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                gamma = _sample_uniform(self.gamma_range)
                img.data = self._apply(img.data, gamma)
        return datapoint


class RandomSimulateLowResolution:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.25,
        scale_range: Tuple[float, float] = (0.85, 1.0),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.scale_range = scale_range

    def _apply(self, img, scale: float):
        tensor, image_type = _image_to_tensor(img)
        _, h, w = tensor.shape
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        tensor = tensor.unsqueeze(0)
        tensor = torch_F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        tensor = torch_F.interpolate(tensor, size=(h, w), mode="bilinear", align_corners=False)
        tensor = tensor.squeeze(0)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            scale = _sample_uniform(self.scale_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, scale)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                scale = _sample_uniform(self.scale_range)
                img.data = self._apply(img.data, scale)
        return datapoint


class RandomBiasField:
    def __init__(
        self,
        consistent_transform: bool,
        p: float = 0.2,
        coeff_range: Tuple[float, float] = (0.05, 0.25),
        blur_sigma_range: Tuple[float, float] = (12.0, 36.0),
    ):
        self.consistent_transform = consistent_transform
        self.p = p
        self.coeff_range = coeff_range
        self.blur_sigma_range = blur_sigma_range

    def _apply(self, img, coeff: float, blur_sigma: float):
        tensor, image_type = _image_to_tensor(img)
        _, h, w = tensor.shape
        field = torch.rand(1, 1, h, w, device=tensor.device)
        kernel_size = max(3, int(round(blur_sigma * 3)) * 2 + 1)
        field = F.gaussian_blur(field, kernel_size=kernel_size, sigma=blur_sigma)
        field = field.squeeze(0)
        field = (field - field.min()) / (field.max() - field.min() + 1e-6)
        field = (1.0 - coeff) + field * (2.0 * coeff)
        tensor = (tensor * field).clamp(0.0, 1.0)
        return _tensor_to_image(tensor, image_type)

    def __call__(self, datapoint: Datapoint, **kwargs):
        if self.consistent_transform:
            if random.random() >= self.p:
                return datapoint
            coeff = _sample_uniform(self.coeff_range)
            blur_sigma = _sample_uniform(self.blur_sigma_range)
            for img in datapoint.images:
                img.data = self._apply(img.data, coeff, blur_sigma)
            return datapoint
        for img in datapoint.images:
            if random.random() < self.p:
                coeff = _sample_uniform(self.coeff_range)
                blur_sigma = _sample_uniform(self.blur_sigma_range)
                img.data = self._apply(img.data, coeff, blur_sigma)
        return datapoint
