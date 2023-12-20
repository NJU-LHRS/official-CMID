import PIL
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .transform import ContrastiveTransform, ParallelTransform, EvalTransform


def build_transform(config):
    transform_type = config.transform_type.lower()
    assert transform_type in ["contrast", "parallel"], "Not Support transform type, Please choose from [mask, \
                                                                    contrast, parallel, bigearthnet]"
    if transform_type == "contrast":
        return ContrastiveTransform(config.size)
    elif transform_type == "parallel":
        return ParallelTransform(config)


def build_cls_transform(config, is_train=True):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        transform = create_transform(
           input_size=config.input_size,
            is_training=True,
            color_jitter=config.color_jitter,
            auto_augment=config.aa,
            interpolation='bicubic',
            re_prob=config.reprob,  # re means random erasing
            re_mode=config.remode,
            re_count=config.recount,
            mean=mean,
            std=std,
        )
        return transform

    t = []
    crop_pct = 224 / 256
    size = int(config.input_size[0] / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(config.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)