import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_transforms(is_train=True, image_size=(256, 256)):
    """
    Albumentations transforms for CT segmentation.

    Image:
        H x W x 3 -> 3 x H x W

    Mask:
        H x W -> H x W

    Mask interpolation is nearest-neighbor to preserve
    the integer segmentation labels 0-4.
    """

    transforms = [
        A.Resize(
            height=image_size[0],
            width=image_size[1],
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        )
    ]

    if is_train:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                fill_mask=255,
                p=0.5
            )
        ])

    transforms.extend([
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    return A.Compose(transforms)