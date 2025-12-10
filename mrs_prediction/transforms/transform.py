from monai.transforms.transform import Transform 
import torch
from monai.transforms.compose import Compose
from monai.transforms.spatial.array import Spacing, Resize, Orientation
from monai.transforms.utility.array import EnsureChannelFirst
from monai.transforms.croppad.array import CropForeground
from monai.transforms.croppad.batch import PadListDataCollate
from monai.transforms.spatial.array import RandFlip
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.transform import MapTransform
from monai.transforms.croppad.array import SpatialPad

class WindowCT(Transform):
    def __init__(self, level, width):
        super().__init__()
        self.level = level
        self.width = width
    def __call__(self, img):
        min_val = self.level - self.width // 2
        max_val = self.level + self.width // 2
        return torch.clamp(img, min_val, max_val, out=img)

class WindowCTd(MapTransform):
    def __init__(self, keys, level, width, allow_missing_keys= False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.level = level
        self.width = width
        self.min_val = self.level - self.width // 2
        self.max_val = self.level + self.width // 2

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.clamp(d[key], self.min_val, self.max_val)
        return d

def ncct_train_transforms(**kwargs):
    window_level, window_width = kwargs["window_level"], kwargs["window_width"]
    mean_intensity, std_intensity = kwargs["mean_intensity"], kwargs["std_intensity"]
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    min_hu_threshold = window_level - window_width // 2
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        WindowCT(window_level, window_width),
        CropForeground(select_fn=lambda x: x > min_hu_threshold, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        RandFlip(prob=0.5, spatial_axis=[0]),
        RandFlip(prob=0.5, spatial_axis=[1]),
        RandFlip(prob=0.5, spatial_axis=[2]),
        NormalizeIntensity(mean_intensity, std_intensity)
    ], lazy=True)

def ncct_test_transforms(**kwargs):
    window_level, window_width = kwargs["window_level"], kwargs["window_width"]
    mean_intensity, std_intensity = kwargs["mean_intensity"], kwargs["std_intensity"]
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    min_hu_threshold = window_level - window_width // 2
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        WindowCT(window_level, window_width),
        CropForeground(select_fn=lambda x: x > min_hu_threshold, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        NormalizeIntensity(mean_intensity, std_intensity)
    ], lazy=True)

def cta_train_transforms(**kwargs):
    window_level, window_width = kwargs["window_level"], kwargs["window_width"]
    mean_intensity, std_intensity = kwargs["mean_intensity"], kwargs["std_intensity"]
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    min_hu_threshold = window_level - window_width // 2
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        WindowCT(window_level, window_width),
        CropForeground(select_fn=lambda x: x > min_hu_threshold, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        RandFlip(prob=0.5, spatial_axis=[0]),
        RandFlip(prob=0.5, spatial_axis=[1]),
        RandFlip(prob=0.5, spatial_axis=[2]),
        NormalizeIntensity(mean_intensity, std_intensity)
    ], lazy=True)

def cta_test_transforms(**kwargs):
    window_level, window_width = kwargs["window_level"], kwargs["window_width"]
    mean_intensity, std_intensity = kwargs["mean_intensity"], kwargs["std_intensity"]
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    min_hu_threshold = window_level - window_width // 2
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        WindowCT(window_level, window_width),
        CropForeground(select_fn=lambda x: x > min_hu_threshold, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        NormalizeIntensity(mean_intensity, std_intensity)
    ], lazy=True)

def dwmri_train_transforms(**kwargs):
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        CropForeground(select_fn=lambda x: x > 0, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        NormalizeIntensity()
    ], lazy=True)

def dwmri_test_transforms(**kwargs):
    voxel_spacing, image_size = kwargs["voxel_spacing"], kwargs["image_size"]
    return Compose([
        EnsureChannelFirst(),
        Orientation("RAS"),
        CropForeground(select_fn=lambda x: x > 0, margin=0),
        Spacing(voxel_spacing),
        Resize(image_size, mode="bilinear"),
        NormalizeIntensity()
    ], lazy=True)

def get_train_transforms(modality, **kwargs):
    transforms_dict = {
        "B0_NCCT": ncct_train_transforms,
        "B0_CTA": cta_train_transforms,
        "24h_DWMRI": dwmri_train_transforms
    }
    return transforms_dict[modality](**kwargs)

def get_test_transforms(modality, **kwargs):
    transforms_dict = {
        "B0_NCCT": ncct_test_transforms,
        "B0_CTA": cta_test_transforms,
        "24h_DWMRI": dwmri_test_transforms
    }
    return transforms_dict[modality](**kwargs)
