# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, Optional
from allegroai import Task, OutputModel
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module(name='AllegroAiLLoggerHook')
class AllegroAiLLoggerHook(LoggerHook):
    """Class to log metrics with clearml.

    It requires `clearml`_ to be installed.


    Args:
        init_kwargs (dict): A dict contains the `clearml.Task.init`
            initialization keys. See `taskinit`_  for more details.
        interval (int): Logging interval (every k iterations). Default 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.

    .. _clearml:
        https://clear.ml/docs/latest/docs/
    .. _taskinit:
        https://clear.ml/docs/latest/docs/references/sdk/task/#taskinit
    """

    val_max = 0

    def __init__(self,
                 init_kwargs: Optional[Dict] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.init_kwargs = init_kwargs

    @master_only
    def before_run(self, runner) -> None:
        super().before_run(runner)
        self.task = Task.current_task()
        self.task_name = self.task.name.split('--')[1]
        self.task_logger = self.task.get_logger()

    @master_only
    def log(self, runner) -> None:

        out_dir = runner.work_dir
        frame_type = runner.meta['frame_type']
        upload_uri = os.path.join(runner.meta['checkpoints_uri'], frame_type)
        assert upload_uri, 'no path was given to save model checkpoint'
        category_dict = {cat: i for i, cat in enumerate(runner.meta['config_dict']['categories'])}
        output_model = OutputModel(
            task=self.task,
            config_dict=runner.meta['config_dict'],
            tags=[runner.meta['task_type'], os.path.splitext(os.path.basename(runner.meta['exp_name']))[0]],
            framework='PyTorch',
            name='best',
            label_enumeration=category_dict
        )
        output_model.connect(task=self.task)

        tags = self.get_loggable_tags(runner)
        for tag, val in tags.items():
            if output_model.name == 'best':
                if tag == 'val/bbox_mAP' and val > AllegroAiLLoggerHook.val_max:
                    checkpoints = os.path.join(out_dir, f'best_bbox_mAP_epoch_{runner.epoch + 1}.pth')
                    assert os.path.exists(checkpoints), f"File {checkpoints} doesn't exists"
                    upload_uri = os.path.join(upload_uri, f'best_bbox_mAP_epoch_{runner.epoch+1}.pth')
                    output_model.update_weights(weights_filename=checkpoints, upload_uri=upload_uri, auto_delete_file=True, iteration=runner.iter)
                    AllegroAiLLoggerHook.val_max = val
            if output_model.name == 'last':
                checkpoints = os.path.join(out_dir, 'latest.pth')
                assert os.path.exists(checkpoints), f"File {checkpoints} doesn't exists"
                upload_uri = os.path.join(upload_uri, f'latest_{runner.epoch+1}.pth')
                output_model.update_weights(weights_filename=checkpoints, upload_uri=upload_uri, auto_delete_file=True, iteration=runner.iter)
            self.task_logger.report_scalar(tag, tag, val, self.get_iter(runner))

