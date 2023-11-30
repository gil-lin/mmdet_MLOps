import argparse
import os

import mmcv
import torch
from clearml import StorageManager
from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
from mmdet.datasets import build_dataset
from allegro_helpers import allegro_coco_format
from mmcv import Config
import re

parser = argparse.ArgumentParser()
parser.add_argument('--target_config', help='test config file path')
parser.add_argument('--source_ckps', help='checkpoint file')
parser.add_argument(
    '--device', default='cuda:0', help='Device used for inference')
parser.add_argument(
    '--score-thr', type=float, default=0.3, help='bbox score threshold')
parser.add_argument(
    '--target_data_type', default='rgb', help='defines image wavelength')
args = parser.parse_args()

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

# 1) build "test" dataset
cfg = Config.fromfile(args.target_config)
cfg.data.test.pipeline = test_pipeline
# build the model from a config file and a checkpoint file
model = init_detector(cfg, args.source_ckps, device=args.device)
model.eval()
dataset = build_dataset(cfg.data.test)
for idx in range(len(dataset)):

    data_info = dataset.data_infos[idx]
    filename = StorageManager.get_local_copy(data_info["coco_url"], force_download=False)
    img = mmcv.imread(filename)

    result = inference_detector(model, img)
    print(result)
    # show the results
    show_result_pyplot(model, img, result, score_thr=args.score_thr)



































































































































































































































































































































































































































































































































































































































































show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
