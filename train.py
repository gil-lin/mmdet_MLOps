import torch
import yaml

cuda_available = torch.cuda.is_available()
print(f"CUDA AVAILABLE: {cuda_available}")
if not cuda_available:
    print(f"{torch.cuda.current_device()}")
    raise Exception("CUDA VERSION ERROR")

import argparse
import copy
import os
import os.path as osp
import time
import warnings
from allegro_helpers.allegro_utills import init_task

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--skip_local_run', help='enable remote training or not',
                        required=True, type=int, default=0)
    parser.add_argument('--work_dir',
                        help='the dir to save logs and models')
    parser.add_argument('--root_dir', dest='root_dir',
                        help="Root directory to locally save experiments.",
                        required=True, type=str)
    parser.add_argument('--data_root', help="Project Root directory.",
                        required=True, default='.', type=str)
    parser.add_argument('--project', dest='project',
                        help='name of project as organized in allegro',
                        required=True, type=str)
    parser.add_argument('--task_type', dest='task_type',
                        help='which CV task is the model performing? (cls, det, seg, ect)',
                        default='det', type=str)
    parser.add_argument('--experiment_tag', dest='experiment_tag',
                        help='experiment to load model',
                        required=True, type=str)
    parser.add_argument('--checkpoints_uri', help='path in //S3: to save checkpoints',
                        default='', type=str)
    parser.add_argument('--s3_root_dir', help='root dir to saved annotations at s3',
                        default='s3://data-playground-target/rafael_styleTransfer/annotations', type=str)
    parser.add_argument('--generate_data', type=int, default=0,
                        help='changes config file to support overfit [generate, from_file]')
    parser.add_argument('--run_overfit_data', type=int, default=0,
                        help='changes config file to support overfit [generate, from_file]')
    parser.add_argument('--resume-from',
                        help='the checkpoint full file path ".ph" to resume from')
    parser.add_argument('--auto-resume', type=str, default=False,
                        help='resume from the latest checkpoint automatically')
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int,
                            help='(Deprecated, please use --gpu-id) number of gpus to use '
                                 '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+',
                            help='(Deprecated, please use --gpu-id) ids of gpus to use '
                                 '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-id', type=int, default=0,
                            help='id of gpu to use '
                                 '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff-seed', action='store_true',
                        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', type=bool, default=False,
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--auto-scale-lr', action='store_true',
                        help='enable automatically scaling LR.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main(mode: str = "train"):
    # Parse arguments ###############################################
    args = parse_args()

    # Initialize AllegroAI ##########################################
    task, experimental_dir = init_task(args, mode=mode)

    # add Parse arguments to cfg ####################################
    cfg_ = Config.fromfile(args.config)
    cfg = copy.deepcopy(cfg_)

    #######################################################
    # anchors = mmcv.load('./data_analysis/rgb/report_416_416_iter1/416_416/train/anchor_optimize_result.json', file_format='json')
    # pr = [anchors[i:i + 4][::-1] for i in range(0, len(anchors), 4)][::-1]
    # cfg.model.bbox_head.anchor_generator.base_sizes = pr
    #######################################################
    args.work_dir = f'{args.work_dir}/{cfg_.data.val.dataview_cfg.frame_type}'
    cfg.data.train.dataset.remote.s3_root_dir = cfg.data.val.remote.s3_root_dir = args.s3_root_dir
    cfg.data.train.dataset.args.workflow = cfg.data.val.args.workflow = len(cfg_.workflow)
    cfg.data.train.dataset.args.generate_overfit = False \
        if cfg.data.train.dataset.args.generate_overfit is None else True  # initializing param generate_overfit

    """ treating CLASSES per different configs """
    categories = cfg.data.train.dataset.CLASSES
    if cfg.data.train.dataset.dataview_cfg['cat_mapping']['*'] is not None:
        keys = [key for key, val in cfg.data.train.dataset.dataview_cfg['cat_mapping']['*'].items()]
        categories = [cls for cls in cfg.data.train.dataset.CLASSES if cls not in keys]
    cfg.model.bbox_head.num_classes = \
        len([cat for cat in categories if cat not in cfg.data.train.dataset.dataview_cfg.ignore_region])

    """ pre-sets 'remote training' parameters to config file """
    cfg.data.train.dataset.remote.skip_local_run = cfg.data.val.remote.skip_local_run = bool(args.skip_local_run)
    if cfg.data.train.dataset.remote.skip_local_run:
        cfg.data.samples_per_gpu = 16

    """ pre-sets for generating overfit annotation file """
    if bool(args.generate_data):
        cfg.data.train.dataset.args.read_from_file.read = False
        cfg.data.train.dataset.args.to_serialize.save = True
        cfg.data.train.dataset.dataview_cfg.max_num_frames = 1000 \
            if cfg.data.train.dataset.dataview_cfg.max_num_frames is None else \
            cfg.data.train.dataset.dataview_cfg.max_num_frames

    """ pre-sets 'overfitting' parameters to config file in case annotation file exists """
    if bool(args.run_overfit_data):
        assert not bool(args.generate_data), '"generate-data" should be False but instead got True}'
        cfg.data.train.dataset.args.generate_overfit = False
        if cfg.data.train.dataset.remote.skip_local_run:
            cfg.data.val.remote.url = \
                cfg.data.train.dataset.remote.url
        else:
            cfg.data.train.dataset.args.read_from_file.read = cfg.data.val.args.read_from_file.read = True
            cfg.data.val.args.read_from_file.path_name = \
                cfg.data.train.dataset.args.read_from_file.path_name

    # merge args
    args_to_merge = {k: v for k, v in dict(args._get_kwargs()).items() if k != 'auto_scale_lr'}
    cfg.merge_from_dict(args_to_merge)

    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update all the '
                          'configuration files to mmdet >= 2.24.1.')

    # add configuration file to allegro GUI task ######################
    with open(args.config, 'r') as f:
        _cfg = yaml.full_load(f)

    _cfg = task.connect(_cfg, name="config")
    ###################################################################

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.auto_resume = args.auto_resume
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['config_dict'] = dict(cfg)
    meta['checkpoints_uri'] = cfg['checkpoints_uri']
    meta['frame_type'] = cfg.data.val.dataview_cfg.frame_type
    meta['task_type'] = cfg['task_type']
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()
    # set random seeds
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    print("Building train dataset")
    if cfg.data.train['type'] is not None:
        datasets = [build_dataset(cfg.data.train)]
    else:
        datasets = [build_dataset(cfg.data.train.dataset)]

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.dataset.pipeline
        print("Building val dataset")
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    if cfg.data.train.dataset.remote['skip_local_run']:
        assert not bool(args.generate_data), '"generate-overfit-data" should be False but instead got True}'
        task.execute_remotely(queue_name=cfg.data.train.dataset.remote['queue_name'])

    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
