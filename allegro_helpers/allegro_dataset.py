# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from collections import OrderedDict

import mmcv
import numpy as np

from clearml import StorageManager
from clearml.config import running_remotely
from mmcv.utils import print_log

from terminaltables import AsciiTable
from torch.utils.data import Dataset
from typing import Tuple, List

from allegro_helpers.allegro_utills import get_train_and_val_dataviews

from mmdet.core import eval_map, eval_recalls
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose


@DATASETS.register_module(name="CustomAllegroDataset")
class CustomAllegroDataset(Dataset):
    """Custom dataset for detection.

    The annotation format is shown as follows. The `ann` field is optional for
    testing.

    .. code-block:: none

        [
            {
                'filename': 'a.jpg',
                'width': 1280,
                'height': 720,
                'ann': {
                    'bboxes': <np.ndarray> (n, 4) in (x1, y1, x2, y2) order.
                    'labels': <np.ndarray> (n, ),
                    'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                    'labels_ignore': <np.ndarray> (k, 4) (optional field)
                }
            },
            ...
        ]

    Args:
        dataview_cfg (dict): Dataview configuration.
        pipeline (list[dict]): Processing pipeline.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Default: None.
        data_root (str, optional): Data root for ``ann_file``,
            ``img_prefix``, ``seg_prefix``, ``proposal_file`` if specified.
        test_mode (bool, optional): If set True, annotation will not be loaded.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests.
        max_frame_size (int): filter all frame sizes that either their width or
        height is lower than this threshold
    """
    PALETTE = [(250, 0, 30), (30, 255, 0), (50, 100, 255), (0, 255, 0)]

    def __init__(self,
                 dataview_cfg: dict,
                 pipeline: List[dict],
                 CLASSES: Tuple,
                 remote: dict,
                 test_mode: bool = False,
                 **kwargs,
                 ):

        super().__init__()
        self.read_from_file = kwargs['args'].read_from_file
        self.data_root = kwargs['args'].data_root
        self.to_serialize = kwargs['args'].to_serialize
        self.min_size = kwargs['args'].max_frame_size
        self.filter_by_size = kwargs['args'].filter_by_size
        self.filter_by_context_id = kwargs['args'].filter_by_context_id
        self.filter_empty_gt = kwargs['args'].filter_empty_gt
        self.frame_type = dataview_cfg['frame_type']
        self.ignored_categories = dataview_cfg['ignore_region']
        self.generate_overfit = kwargs['args'].generate_overfit
        self.CLASSES = CLASSES
        self.PALETTE = CustomAllegroDataset.PALETTE
        self.test_mode = test_mode
        workflow = kwargs['args'].workflow

        if not self.read_from_file.get('read', None):
            self.dataview = get_train_and_val_dataviews(
                dataview_cfg=dataview_cfg,
                categories=self.CLASSES,
                use_train_iterable_dataset=False,
                prefetch=remote['prefetch'] and (not remote['skip_local_run'] or running_remotely()),
            )
        # for the case of mapping we modify original categories to a subset of them
        if dataview_cfg['cat_mapping']['*'] is not None:
            keys = [key for key, val in dataview_cfg['cat_mapping']['*'].items()]
            self.CLASSES = [cls for cls in self.CLASSES if cls not in keys]
        # checks if we wish to use an existing annotation_file
        if not remote['skip_local_run']:
            # generate annotation file if not exists
            assert self.data_root, f'data_root cannot be empty: {self.data_root}'
            if os.path.basename(os.getcwd()) == 'AllegroSt' and not os.path.exists(os.path.join(self.data_root, f'annotations/{self.frame_type}')):
                os.makedirs(os.path.join(self.data_root, f'annotations/{self.frame_type}'))

            # construct dataset object from existing ann_file
            if self.read_from_file.get('read', False):
                self.annotation_path = os.path.join(
                    self.data_root,
                    f'annotations/{self.frame_type}/{self.read_from_file["path_name"]}'
                )

            else:  # construct new dataset object from dataview in case of [new_data or overfit]
                if self.test_mode and workflow == 2:
                    self.annotation_path = os.path.join(
                        self.data_root,
                        f'annotations/{self.frame_type}/{self.read_from_file["path_name"]}'
                    )
                else:
                    self.annotation_path = self.generate_ann()
        else:  # running remote
            ann_path_s3 = os.path.join(
                remote['s3_root_dir'],
                f'{self.frame_type}/{remote["url"]}'
            )
            print(f"Downloading annotation file from: {ann_path_s3}")
            self.annotation_path = StorageManager.get_local_copy(
                ann_path_s3,
                force_download=False
            )

        quit() if self.generate_overfit else print("generating data infos")
        self.data_infos = self.load_annotations(self.annotation_path)

        # filter images too small and containing no annotations
        if not test_mode:
            valid_inds = self._filter_imgs(min_size=self.min_size)
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()
        # switching from [person, vehicle, two-wheeled, dilemma_zone]
        # to [person, vehicle, two-wheeled]
        self.CLASSES = [cat for cat in self.CLASSES if cat not in self.ignored_categories]
        self.PALETTE = self.PALETTE[: len(self.CLASSES)]
        # processing pipeline
        self.pipeline = Compose(pipeline)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.data_infos[idx]['ann']

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = None
        results['bbox_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `name` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
         """
        img_info = self.data_infos[idx]
        container = StorageManager.get_local_copy(img_info["filename"], force_download=False)
        if not container:
            self.__getitem__(self._rand_another(idx))
        img_info["filename"] = container
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        container = StorageManager.get_local_copy(img_info["filename"], force_download=False)
        if not container:
            self.__getitem__(self._rand_another(idx))
        img_info["filename"] = container
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def get_cat2imgs(self):

        """Get a dict with class as key and img_ids as values, which will be
        used in :class:`ClassAwareSampler`.

        Returns:
            dict[list]: A dict of per-label image list,
            the item of the dict indicates a label index,
            corresponds to the image index that contains the label.
        """
        if self.CLASSES is None:
            raise ValueError('self.CLASSES can not be None')
        # sort the label index
        cat2imgs = {i: [] for i in range(len(self.CLASSES))}
        for i in range(len(self)):
            cat_ids = set(self.get_cat_ids(i))
            for cat in cat_ids:
                cat2imgs[cat].append(i)
        return cat2imgs

    def format_results(self, results, **kwargs):
        """ Placeholder to format result to dataset specific output."""

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    dataset=[str(i) for i in self.CLASSES],
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results

    def __repr__(self):
        """Print the number of instance number."""
        dataset_type = 'test' if self.test_mode else 'train'
        result = (f'\n{self.__class__.__name__} {dataset_type} dataset '
                  f'with number of images {len(self)}, '
                  f'and instance counts: \n')
        if self.CLASSES is None:
            result += 'Category names are not provided. \n'
            return result
        instance_count = np.zeros(len(self.CLASSES) + 1).astype(int)
        # count the instance number in each image
        for idx in range(len(self)):
            label = self.get_ann_info(idx)['labels']
            unique, counts = np.unique(label, return_counts=True)
            if len(unique) > 0:
                # add the occurrence number to each class
                instance_count[unique] += counts
            else:
                # background is the last index
                instance_count[-1] += 1
        # create a table with category count
        table_data = [['category', 'count'] * 5]
        row_data = []
        for cls, count in enumerate(instance_count):
            if cls < len(self.CLASSES):
                row_data += [f'{cls} [{self.CLASSES[cls]}]', f'{count}']
            else:
                # add the background number
                row_data += ['-1 background', f'{count}']
            if len(row_data) == 10:
                table_data.append(row_data)
                row_data = []
        if len(row_data) >= 2:
            if row_data[-1] == '0':
                row_data = row_data[:-2]
            if len(row_data) >= 2:
                table_data.append([])
                table_data.append(row_data)

        table = AsciiTable(table_data)
        result += table.table
        return result
