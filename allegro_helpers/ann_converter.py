import json
import os
import re
import time
from collections import defaultdict, Counter
from typing import List

import mmcv
import numpy as np

from clearml import StorageManager
from tqdm import tqdm


class BaseAnnConverter:
    """
    General : Base class for annotation converter
              input: Dataview
              output: annotation file in "COCO" format
    args:
        dataview: (type: DataView) will be used to iterate overt
        classes: (type: tuple) (category_1, category_2, ... category_n)
        test_mode: (type: bool) indicates training mode , "False" if mode='train'
        logger: logger object that will be used during training
        path: (type: str) path to save annotation files
    """
    max_idx = 0

    def __init__(self,
                 dataview,
                 classes,
                 test_mode,
                 ignored_categories,
                 logger,
                 path,
                 ):
        super().__init__()
        self.frame_groups = dataview.get_iterator()
        self.ignore = ignored_categories
        self.logger = logger
        self.annotation_path = path
        self.CLASSES = classes
        self.test_mode = test_mode
        self.categories = [{'id': i + 1, 'name': name} for i, name in enumerate(self.CLASSES)]
        BaseAnnConverter.max_idx = 0

    def create_dataset(self, save_results=False):
        raise NotImplementedError

    def ann_from_single_frame(self, idx, single_frame, ann_file=None):
        """
        General: Generated "coco" format annotation file from frame_group
        :param idx: running index of objects in DataView
        :param single_frame: object in DataView
        :param ann_file: DefaultDict[List(Dict)] object to contain annotation data
        :return: List of dictionaries
        """
        assert isinstance(ann_file, defaultdict)
        image_id = idx + 1
        if idx == 0:
            ann_file['info'] = single_frame.origin
            ann_file['licences'] = [{'url': 'http://www.sightx.ai', 'id': 1, 'name': 'SightX-ai'}]
            ann_file['categories'] = self.categories

        """calculating mean/std per channel"""
        img = mmcv.imread(StorageManager.get_local_copy(single_frame.source)).astype(np.uint8)
        mean, std = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))

        images = dict(
            licence=1,
            file_name=single_frame.source.split('/')[-1],
            coco_url=single_frame.source,
            height=single_frame.height,
            width=single_frame.width,
            id=image_id,
            mean=list(mean),
            std=list(std),
        )

        def parse_frame_group(single, img_id: int) -> List[dict]:
            """
            This method gets a "single_frame" objects and parses it to build "annotations" dict
            :param img_id:
            :param single: a single argument from a FrameGroup
            :return: COCO taxonomy annotation dictionary from each frame

            Example:
            collecting all annotations ids by image id
            [{...'image_id': 283, ....'id': 1 ...}, {...'image_id': 283, ....'id': 2 ...}]
            """
            ann = []
            ann_idx = AnnGeneratorTrain.max_idx
            for idx in range(len(single.get_annotations())):
                ann_idx += 1
                iscrowd = 0
                category_id = self.CLASSES.index(single_frame.get_annotations()[idx].labels[0]) + 1
                assert category_id in [i + 1 for i in range(len(self.CLASSES))]
                w, h = single_frame.get_annotations()[idx].bounding_box_xywh[2:]

                """iscrowd is treated the same as "ignore_region" in which case there is no contribution to the Loss"""
                if single_frame.get_annotations()[idx].labels[0] in self.ignore or min(w, h) <= 10:
                    iscrowd = 1

                annotations = dict(
                    image_id=img_id,
                    iscrowd=iscrowd,
                    bbox=single_frame.get_annotations()[idx].bounding_box_xywh,
                    category_id=category_id,
                    area=int(w * h),
                    id=ann_idx,
                    ignore=0,
                )

                ann.append(annotations)
            AnnGeneratorTrain.max_idx = ann_idx
            return ann

        annotation = parse_frame_group(single_frame, img_id=image_id)

        return images, annotation

    def save_result(self, ann_file):
        self.logger.info(f'\n\nSerializing annotation file ...\nThis might take a while ...')
        assert isinstance(self.annotation_path, str), 'no annotation file was given'
        tic = time.time()

        with open(self.annotation_path, "w") as file:
            json.dump(ann_file, file)

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

        try:
            with open(self.annotation_path, "r") as output_file:
                temp_dict = json.load(output_file)
        except FileNotFoundError as e:
            self.logger.info("annotation file was not Serialized")
        else:
            file_name = re.split('[/ .]', self.annotation_path)[-2]
            self.logger.info(f'annotation file "{file_name}" was successfully Serialized to {self.annotation_path}')

        remote_path = f"s3://data-playground-target/rafael_styleTransfer/{'/'.join(self.annotation_path.split('/')[1:])}"
        StorageManager.upload_file(
            self.annotation_path,
            remote_path
        )
        self.logger.info(f'annotation file was successfully Serialized to {remote_path}')
        return self.annotation_path


class AnnGeneratorTrain(BaseAnnConverter):
    r""" annotation generator - converts framegroup into coco format dict

    """

    def __init__(self, **kwargs):
        super(AnnGeneratorTrain, self).__init__(**kwargs)

    def create_dataset(self, save_results=False):
        self.logger.info(f'Start generating annotations...')
        corrupted_imgs = dict()
        ann_file = defaultdict(list)
        for idx, frame_group in enumerate(tqdm(self.frame_groups, desc='Converting FrameGroups to Json')):
            single_frame = list(frame_group.values())[0]
            """
            checks for corrupted or un-available images. if corrupted skips a frame 
            and corrects the index
            """
            valid_path = StorageManager.get_local_copy(single_frame.source)
            if valid_path is None:
                corrupted_imgs[idx] = single_frame.source
                valid_path = False
            if not valid_path:
                continue
            current_idx = idx - len(corrupted_imgs)
            img, anns = self.ann_from_single_frame(current_idx, single_frame, ann_file)
            ann_file['images'].append(img)
            for an in anns:
                ann_file['annotations'].append(an)
        self.logger.info(f'The Dataset has: {len(corrupted_imgs)} un-valid images')
        # logging corrupted images path
        if not self.test_mode:
            corrupted_img_path = f'{os.path.dirname(self.annotation_path)}/corrupted_train.json'
        else:
            corrupted_img_path = f'{os.path.dirname(self.annotation_path)}/corrupted_val.json'

        with open(corrupted_img_path, "w") as f:
            json.dump(corrupted_imgs, f)

        ann_count = [x['category_id'] for x in ann_file['annotations']]
        self.logger.info(f'annotations count: {Counter(ann_count)}')

        if save_results:
            return self.save_result(ann_file)
        else:
            return self.annotation_path
