import cv2
import math
import torch
import random
import numpy as np
import ml_collections
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
import albumentations as A
import torchvision.transforms as T
from Exploring.utils import to_tensor, to_numpy
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as tsfunc
from typing import List, Union, Tuple, Dict, Optional


def get_contrastive_transform(size) -> List[A.Compose]:
    transform = A.Compose(
        [
            A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=[0.1, 2.0], p=1.0),
            A.GaussNoise(p=0.6),
            A.Solarize(p=0.2),
            A.ToGray(p=0.2),
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2()
        ]
    )

    return transform


def _location_to_NxN_grid(location, N=7):
    h, w, i, j, flip = location

    size_h_case = h / N
    size_w_case = w / N
    half_size_h_case = size_h_case / 2
    half_size_w_case = size_w_case / 2
    final_grid_x = torch.zeros(N, N)
    final_grid_y = torch.zeros(N, N)

    final_grid_x[0][0] = i + half_size_h_case
    final_grid_y[0][0] = j + half_size_w_case
    for k in range(1, N):
        final_grid_x[k][0] = final_grid_x[k - 1][0] + size_h_case
        final_grid_y[k][0] = final_grid_y[k - 1][0]
    for l in range(1, N):
        final_grid_x[0][l] = final_grid_x[0][l - 1]
        final_grid_y[0][l] = final_grid_y[0][l - 1] + size_w_case
    for k in range(1, N):
        for l in range(1, N):
            final_grid_x[k][l] = final_grid_x[k - 1][l] + size_h_case
            final_grid_y[k][l] = final_grid_y[k][l - 1] + size_w_case

    final_grid = torch.stack([final_grid_x, final_grid_y], dim=-1)
    if flip:
        # start_grid = final_grid.clone()
        for k in range(0, N):
            for l in range(0, N // 2):
                swap = final_grid[k, l].clone()
                final_grid[k, l] = final_grid[k, N - 1 - l]
                final_grid[k, N - 1 - l] = swap

    return final_grid


class HorizontalFlipIfTrue(A.DualTransform):
        def apply(self, img, **params):
            if img.ndim == 3 and img.shape[2] > 1 and img.dtype == np.uint8:
                # Opencv is faster than numpy only in case of
                # non-gray scale 8bits images
                return A.hflip_cv2(img), True

            return A.hflip(img), True

        def apply_to_bbox(self, bbox, **params):
            return A.bbox_hflip(bbox, **params)

        def apply_to_keypoint(self, keypoint, **params):
            return A.keypoint_hflip(keypoint, **params)

        def get_transform_init_args_names(self):
            return ()


class _BaseRandomSizedCropWithLocation(A.DualTransform):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(_BaseRandomSizedCropWithLocation, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = A.random_crop(img, crop_height, crop_width, h_start, w_start)
        return A.resize(crop, self.height, self.width, interpolation), [crop_height, crop_width, h_start, w_start]

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return A.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = A.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        keypoint = A.keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint


class RandomResizedCropWithLocation(_BaseRandomSizedCropWithLocation):
    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):

        super(RandomResizedCropWithLocation, self).__init__(
            height=height, width=width, interpolation=interpolation, always_apply=always_apply, p=p
        )
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "height", "width", "scale", "ratio", "interpolation"


class EvalTransform:
    def __init__(self,
                 img_size: Optional[Union[int, Tuple]] = 224):
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.resize_trans = RandomResizedCropWithLocation(height=self.img_size, width=self.img_size, scale=(0.3, 1.0))
        self.flip_trans = HorizontalFlipIfTrue(p=0.5)
        self.transform = A.Compose([
            A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
            ToTensorV2()
        ]
        )

    def __call__(self, img: np.ndarray, mask=None):
        if isinstance(img, Image.Image):
            img = np.asarray(img)

        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.asarray(mask)

            resiezd_trans = self.resize_trans(image=img, mask=mask)
            img, location = resiezd_trans["image"]
            mask = resiezd_trans["mask"]

            fliped_trans = self.flip_trans(image=img, mask=mask)
            img = fliped_trans["image"]
            mask = fliped_trans["mask"]

            if isinstance(img, Tuple):
                img, flip = img
            else:
                flip = False

            transed = self.transform(image=img, mask=mask)
            location.append(flip)
            return transed["image"], transed["mask"], location
        else:
            img, location = self.resize_trans(image=img)["image"]
            img = self.flip_trans(image=img)["image"]
            if isinstance(img, Tuple):
                img, flip = img
            else:
                flip = False

            location.append(flip)
            return self.transform(image=img)["image"], location


class MaskTransform:
    def __init__(self,
                 img_size: Optional[Union[int, Tuple]] = 224,
                 hp_filter_sigma: float = 0.5,
                 grid_mask_prob: float = 0.8,
                 grid_mask_ratio: float = 0.8,
                 focal_mask_ratio: float = 0.3,
                 channel_mask_ratio: float = 0.3,
                 channel_mask_prob: float = 0.7):
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.spatial_mask_prob = grid_mask_prob
        self.channel_mask_prob = channel_mask_prob
        self.standard_transform = A.Compose(
            [
                A.RandomResizedCrop(self.img_size, self.img_size, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(0.4, 0.4, 0.2, 0.1, p=0.8),
                A.ToGray(p=0.2),
                A.GaussianBlur(sigma_limit=(hp_filter_sigma, hp_filter_sigma), always_apply=True),
                A.Normalize(mean=[0.33797, 0.3605, 0.3348], std=[0.1359, 0.1352, 0.1407]),
                ToTensorV2()
            ]
        )

        self.grid_mask_generator = MaskGenerator(input_size=self.img_size, mask_ratio=grid_mask_ratio)
        self.channel_mask_generator = MaskGenerator(input_size=self.img_size, mask_ratio=channel_mask_ratio,
                                                    mask_channel=3)
        self.focal_mask_generator = FocalMaskGenerator(img_size=img_size, focal_mask_ratio=focal_mask_ratio)
        self.gaussian_noise = GaussianNoiseGenerator(img_size=img_size)

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        img = self.standard_transform(image=img)["image"]
        if np.random.random() <= self.spatial_mask_prob:
            mask = self.grid_mask_generator()
            gauss = self.gaussian_noise(False)
            img = img * mask + mask * gauss
        else:
            img = self.focal_mask_generator(img)

        if np.random.random() <= self.channel_mask_prob:
            mask = self.channel_mask_generator()
            gauss = self.gaussian_noise(True)
            img = img * mask + mask * gauss

        return img.float()


class BlockwiseMaskGenerator(object):
    """Generate random block for the image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6,
                 mask_color='zero',
                 ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero', 'rand', ]

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0  # model patch size应该是stage的那个个数

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)  # scale 是上采样的个数, ratio保持不变
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == 'mean':
            if isinstance(img, Image.Image):
                img = np.array(img)
                mask_ = to_numpy(mask).reshape((self.rand_size * self.scale, -1, 1))
                mask_ = mask_.repeat(
                    self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
                mean = img.reshape(-1, img.shape[2]).mean(axis=0)
                img = np.where(mask_ == 1, img, mean)
                img = Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                mask_ = to_tensor(mask)
                mask_ = mask_.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                    self.model_patch_size, 1).contiguous()
                img = img.clone()
                mean = img.mean(dim=[1, 2])
                for i in range(img.size(0)):
                    img[i, mask_ == 1] = mean[i]

        return img, mask


class ContrastiveTransform:
    def __init__(self,
                 size: Union[Tuple, List],
                 transform_func=get_contrastive_transform):
        self.resize_trans = RandomResizedCropWithLocation(height=size[0], width=size[1], scale=(0.3, 1.0), always_apply=True)
        self.flip_trans = HorizontalFlipIfTrue(p=0.5)
        self.transform = transform_func(size)

    def __call__(self, img, mask=None):
        img = to_numpy(img, ToCHW=False)

        img, location = self.resize_trans(image=img)["image"]
        img = self.flip_trans(image=img)["image"]
        if isinstance(img, Tuple):
            img, flip = img
        else:
            flip = False

        img = self.transform(image=img)["image"]
        location.append(flip)
        return img, location


class ParallelTransform:
    MODEL_PATCH_SIZE = {
        0: 4,
        1: 8,
        2: 16,
        3: 32,
    }

    def __init__(self,
                 config: ml_collections.ConfigDict):
        self.size = tuple(config.size)
        self.contras_transform = ContrastiveTransform(config.size)
        self.mask_transform = BlockwiseMaskGenerator(input_size=config.size[0],
                                                     mask_patch_size=config.mask_patch_size,
                                                     model_patch_size=self.MODEL_PATCH_SIZE[config.mask_stage],
                                                     mask_ratio=config.mask_ratio,
                                                     mask_color=config.mask_color)
        self.eval_transform = EvalTransform(config.size[0])

    def __call__(self, img) -> Union[Dict[str, torch.Tensor], List]:
        if isinstance(img, dict):
            img, target = img["img"], img["label"]
        else:
            target = None

        if isinstance(img, Image.Image):
            img_shape = img.size[0]
        else:
            img_shape = img.shape[0]

        if img_shape != self.size[0]:
            img = img.resize(self.size, resample=Image.BICUBIC)
            if target is not None:
                target = target.resize(self.size, resample=Image.NEAREST)

        view, view_location = self.contras_transform(img)

        if target is not None:
            img, seg_map, location = self.eval_transform(img, target)
        else:
            img, location = self.eval_transform(img)

        mim_img, mask = self.mask_transform(img)
        view_mim_img, view_mask = self.mask_transform(view)

        locations = []

        for loc in [location, view_location]:
            locations.append(_location_to_NxN_grid(loc, N = self.size[0] // 32))

        img_dict = dict(img=img, mim_img=mim_img, view=view, mask=mask, locations=locations, view_mim_img=view, view_mask=view_mask)
        return img_dict if target is None else [img_dict, seg_map]