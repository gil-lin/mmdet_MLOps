

import numpy as np
from typing import List, Union
import albumentations as A
import mmcv
import cv2
from clearml import StorageManager


class PixelLevelDA:
    def __init__(self, method, target_images_ann):
        self.method = method
        ann_dict = mmcv.load(target_images_ann)
        img_paths = ann_dict['images']
        self.target_images = self.get_imgs(img_paths)

    @staticmethod
    def get_imgs(img_paths):
        return [path['coco_url'] for path in img_paths]

    def get_random_img(self, flag):
        image_path = np.random.choice(self.target_images)
        filename = StorageManager.get_local_copy(image_path, force_download=False)
        if filename is None:
            self.get_random_img(flag='grayscale')
        target_image = mmcv.imread(filename, flag=flag)
        return [target_image]

    def fda(self, image: List[Union[str, np.ndarray]], beta_limit=0.13):
        augmentation = A.Compose(
            [A.FDA(self.get_random_img(flag='grayscale'), beta_limit=beta_limit, p=1, read_fn=lambda x: x)])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transfer_image = augmentation(image=image_gray)
        transfer_image = np.dstack([transfer_image['image']] * 3)
        return transfer_image

    def pixel_distribution(self, image: List[Union[str, np.ndarray]], transform_type='pca'):
        augmentation = A.Compose(
            [A.PixelDistributionAdaptation(self.get_random_img(flag='unchanged'), transform_type=transform_type, p=1, read_fn=lambda x: x)])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.dstack([image_gray] * 3)
        transfer_image = augmentation(image=image_gray)['image']
        return transfer_image

    def hist_matching(self, image: List[Union[str, np.ndarray]]):
        augmentation = A.Compose(
            [A.HistogramMatching(self.get_random_img('grayscale'), blend_ratio=(0.5, 1.), p=1, read_fn=lambda x: x)])
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transfer_image = augmentation(image=image_gray)['image']
        transfer_image = np.dstack([transfer_image] * 3)
        return transfer_image

    def apply(self, source_img):
        if self.method == 'fda':
            return self.fda(source_img)
        if self.method == 'pixel_distribution':
            return self.pixel_distribution(source_img)
        else:
            return self.hist_matching(source_img)
