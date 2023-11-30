
import argparse
import os
import warnings
import time
import re

import mmcv
import torch
from allegroai import StorageManager
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (init_dist, wrap_fp16_model)
from mmdet.apis.inference import init_detector

from mmdet.apis import single_gpu_test
from mmdet.datasets import (build_dataset, replace_ImageToTensor)
from mmdet.utils import get_device
from Image_Style_Transfer.basic_style_transfer import PixelLevelDA


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--target_config', help='test config file path')
    parser.add_argument('--source_ckps', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--transfer_method',
        type=str,
        help='basic domain adaptation methods: ["fda", "pixel_distribution", "hist_matching"]')
    parser.add_argument(
        '--out',
        help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_false',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', type=int, default=0, help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--target_data_type',
        default='rgb',
        help='defines image wavelength')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.target_data_type == re.split('[/ .]', args.target_config)[-3], f"target config and data type " \
    f"must be the same instead got: {args.target_data_type} and {re.split('[/ .]', args.target_config)[-2]}"

    img_norm_cfg = dict(
        mean=[110.675, 113.167, 113.311],
        std=[47.832, 49.947, 51.754],
        to_rgb=True
    )

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(768, 576),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=False),
                dict(type='RandomFlip'),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ]

    s3_path = 's3://data-playground-target/rafael_styleTransfer/checkpoints/rgb/' \
              'best_bbox_mAP_epoch_44.pth/YoLoV3/Experiments_6/' \
              'train--VIS_VIS_UpperLimit_768_576_4heads_new_Aug.71addd79bd8944758c65463f5b83521c/' \
              'models/best_bbox_mAP_epoch_44.pth'

    if args.source_ckps.split('/')[-1] == 'mwir':

        s3_path = 's3://data-playground-target/rafael_styleTransfer/checkpoints/mwir/' \
                  'best_bbox_mAP_epoch_49.pth/YoLoV3/Experiments_6/' \
                  'train--IR_IR_UpperLimit_640_512_4heads_new_.eef34a15533944b49d892b6944215b84/' \
                  'models/best_bbox_mAP_epoch_49.pth'

        if args.target_data_type == 'mwir':

            img_norm_cfg = dict(
                type='Normalize',
                mean=[125.882, 125.882, 125.882],
                std=[59.887, 59.887, 59.887],
                to_rgb=True,
            )

            test_pipeline[1]["img_scale"] = (640, 512)
            test_pipeline[1]["transforms"] = img_norm_cfg

        else:
            test_pipeline[1]["img_scale"] = (768, 576)

    elif args.target_data_type == 'mwir':
        test_pipeline[1]["img_scale"] = (640, 512)

    local_path = StorageManager.get_local_copy(s3_path)
    s3_checkpoint = torch.load(local_path)
    args.source_ckps = f"{args.source_ckps}/{re.split('[/ .]', s3_path)[-4]}_{os.path.basename(s3_path)}"
    torch.save(s3_checkpoint, args.source_ckps)
    args.show = bool(args.show)

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.target_config)
    cfg.data.test.pipeline = test_pipeline

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.backbone.init_cfg = None

    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')

    cfg.device = get_device()
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # allows not to create
    if args.work_dir:
        mmcv.mkdir_or_exist(os.path.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = os.path.join(args.work_dir, f'eval_{timestamp}.json')

    # 1) build "test" dataset
    dataset = build_dataset(cfg.data.test)

    cfg.model.train_cfg = None

    # 2) build the model and load checkpoint
    model = init_detector(cfg, args.source_ckps, device='cuda:0')

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    target_annotations = f'./annotations/{args.target_data_type}/{cfg.data.test.args.read_from_file.path_name}'
    if args.transfer_method is not None:
        assert args.transfer_method in ["fda", "pixel_distribution", "hist_matching"], 'No such methods'
        aug = PixelLevelDA(method=args.transfer_method, target_images_ann=target_annotations)

    if not distributed:

        outputs = single_gpu_test(model,
                                  cfg.data.test.args.filter_by_context_id,
                                  dataset,
                                  aug,
                                  args.show,
                                  args.show_dir,
                                  args.show_score_thr)

        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        if args.format_only:
            results, _ = dataset.format_results(outputs, f"{args.work_dir}/results")
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            kwargs = {} if args.eval_options is None else args.eval_options
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # 6) get evaluation results "allegro_coco_format"
            metric = dataset.evaluate(outputs, **eval_kwargs)

            print(metric)
            metric_dict = dict(
                config=args.target_config,
                metric=metric,
                artifacts=args.source_ckps,
                cfg_file=dict(cfg),
            )
            if args.work_dir:
                mmcv.dump(metric_dict, json_file)


if __name__ == '__main__':
    main()
