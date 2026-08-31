import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms(is_train=True, image_size=(256, 256)):
    """
    Returns albumentations transforms.
    Mask interpolation is STRICTLY cv2.INTER_NEAREST.
    Image is duplicated across 3 channels in the dataset loader to be compatible with ResNet34, 
    but Albumentations handles the (H, W, C) input.
    """
    if is_train:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, 
                               interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=255, p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2()
        ], is_check_shapes=False, additional_targets={})
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1], interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2()
        ], is_check_shapes=False)
