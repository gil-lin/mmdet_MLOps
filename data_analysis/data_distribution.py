# Copyright (c) OpenMMLab. All rights reserved.
"""Optimize anchor settings on a specific dataset.

Example:

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --output-dir ${OUTPUT_DIR}
    Use differential evolution to optimize anchors::

        python tools/analysis_tools/optimize_anchors.py ${CONFIG} \
        --output-dir ${OUTPUT_DIR}
"""

import argparse
import os

import mmcv
import numpy as np
import pandas as pd
import seaborn
from clearml import StorageManager
from mmcv import Config

import matplotlib.pyplot as plt
import scipy.stats as stats

from mmdet.datasets import build_dataset
from mmdet.utils import get_root_logger, update_data_root
import allegro_helpers.allegro_coco_format


def parse_args():
    parser = argparse.ArgumentParser(description='Optimize anchor parameters.')
    parser.add_argument('config', help='Train config file path.')
    parser.add_argument(
        '--execute',
        type=str,
        nargs='+',
        help='executes a method ["stat", "countplot"]')
    # parser.add_argument('--execute', type=str, help='executes a method ["stat", "countplot"]')
    parser.add_argument('--show', default=False, help='plot statistics')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for calculating.')
    parser.add_argument('--output-dir', default=None, type=str, help='Path to save anchor optimize result.')
    args = parser.parse_args()
    return args


class BaseDistribution:
    """Base class for anchor optimizer.

    Args:
        dataset (obj:`Dataset`): Dataset object.
        logger (obj:`logging.Logger`): The logger for logging.
        device (str, optional): Device used for calculating.
            Default: 'cuda:0'
        out_dir (str, optional): Path to save anchor optimize result.
            Default: None
    """

    def __init__(self,
                 dataset,
                 logger,
                 mode,
                 device='cuda:0',
                 out_dir=None,
                 show=False):
        self.dataset = dataset
        self.logger = logger
        self.device = device
        self.out_dir = out_dir
        self.mode = mode
        self.show = show

        if not os.path.exists(os.path.join(self.out_dir, self.mode)):
            os.makedirs(os.path.join(self.out_dir, self.mode))

    def execute(self):
        raise NotImplementedError

    def save_result(self, result, path=None):
        self.logger.info(f'Statistics result:{result.keys()}')
        for name, values in result.items():
            if name == 'stat':
                json_path = os.path.join(path, f'{self.mode}/{name}_results.json')
                mmcv.dump(values, json_path)
                self.logger.info(f'{name} Result were saved in {json_path}')
                if self.show:
                    self.plot_data(values, os.path.dirname(json_path))
            else:
                pkl_path = os.path.join(path, f'{self.mode}/{name}_results.pkl')
                mmcv.dump(values, pkl_path)
                self.logger.info(f'{name} Results were saved in {pkl_path}')

    def plot_data(self, mean_std, path):
        mean, std = mean_std
        label = ['Red', 'Green', 'Blue']
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        plt.plot(x[:, 0], stats.norm.pdf(x[:, 0], mean[0], std[0]), color='b')
        plt.plot(x[:, 1], stats.norm.pdf(x[:, 1], mean[1], std[1]), color='g')
        plt.plot(x[:, 2], stats.norm.pdf(x[:, 2], mean[2], std[2]), color='r')
        plt.legend(label)
        plt.title("MEAN-STD DATASET PDF")
        plt.savefig(os.path.join(path, 'dist.jpg'))
        plt.close()

    def plot_hist(self, results: list, path: str):

        mapping_dict = {
            1: 'person',
            2: 'vehicle',
            3: 'two-wheeled',
        }

        df = pd.DataFrame(results, columns=['category_id', 'size'])
        df['label_names'] = df['category_id'].map(mapping_dict)
        ax = seaborn.countplot(x='label_names', order=['person', 'vehicle', 'two-wheeled'], hue='size', hue_order=['small', 'medium', 'large'], data=df)
        plt.title(f"{self.mode} - dataset distribution by bbox size")
        for p in ax.patches:
            ax.annotate('{:.1f}'.format(p.get_height()), (p.get_x(), p.get_height() + 0.01))
        # if self.show:
        #     plt.show()
        plt.savefig(os.path.join(path, f'{self.mode}/countplot.jpg'))
        plt.close()
        del df


class StatTrain(BaseDistribution):
    r""" get mean and std of train and val datasets

    """

    def __init__(self, **kwargs):
        super(StatTrain, self).__init__(**kwargs)

    def execute(self):
        stat = self.create_dataset_stat()
        self.save_result(stat, self.out_dir)

    def create_dataset_stat(self):
        self.logger.info(f'Start gathering image data...')
        mean_, std_ = [], []
        prog_bar = mmcv.ProgressBar(len(self.dataset))

        for idx in range(len(self.dataset)):
            filename = self.dataset.data_infos[idx]['filename']
            valid_path = StorageManager.get_local_copy(filename)
            valid_path = False if valid_path is None else valid_path
            if not valid_path:
                continue
            img = mmcv.imread(valid_path).astype(np.uint8)
            mean, std = np.mean(img, axis=(0, 1)), np.std(img, axis=(0, 1))
            mean_.append(mean)
            std_.append(std)

            prog_bar.update()
        print('\n')

        stat_results = dict(
            mean_arr=np.array(mean_),
            std_arr=np.array(std_),
            stat=(np.mean(mean_, axis=0), np.mean(std_, axis=0))
        )

        return stat_results


class SizeDistribution(BaseDistribution):
    r""" get mean and std of train and val datasets

    """

    def __init__(self, **kwargs):
        super(SizeDistribution, self).__init__(**kwargs)

    def execute(self):
        stat = self.create_dataset_size_stat()
        self.plot_hist(stat, self.out_dir)

    def create_dataset_size_stat(self):
        self.logger.info(f'Start gathering size data...')
        prog_bar = mmcv.ProgressBar(len(self.dataset))
        dataset = mmcv.load(self.dataset.annotation_path, 'json')
        stat_results = []
        for data in dataset['annotations']:
            if data['iscrowd'] == 1:
                continue
            if data['area'] <= 32**2:
                stat_results.append([data['category_id'], 'small'])
            elif 32**2 < data['area'] <= 96**2:
                stat_results.append([data['category_id'], 'medium'])
            elif data['area'] > 96**2:
                stat_results.append([data['category_id'], 'large'])
            prog_bar.update()
        print('\n')

        return stat_results


def main():
    logger = get_root_logger()
    args = parse_args()
    cfg = args.config
    cfg = Config.fromfile(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    train_data_cfg = cfg.data.train.dataset
    val_data_cfg = cfg.data.val
    datasets = {'train': train_data_cfg, 'val': val_data_cfg}

    for mode, dataset_cfg in datasets.items():
        dataset = build_dataset(dataset_cfg)

        if "stat" in args.execute:
            stat_gen = StatTrain(
                dataset=dataset,
                device=args.device,
                logger=logger,
                out_dir=os.path.join(args.output_dir, "stat"),
                mode=mode,
                show=args.show)
            stat_gen.execute()

        if "countplot" in args.execute:
            count_plot = SizeDistribution(
                dataset=dataset,
                device=args.device,
                logger=logger,
                out_dir=os.path.join(args.output_dir, "countplot"),
                mode=mode,
                show=args.show)
            count_plot.execute()
        else:
            print(f"you have chosen an invalid argument {args.execute} should be [stat, countplot]")


if __name__ == '__main__':
    main()
