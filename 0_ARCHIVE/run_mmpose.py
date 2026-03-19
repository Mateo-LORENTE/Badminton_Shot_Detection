import os
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
import numpy as np

import cv2
import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
# from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

    
def assert_file_exists(file_path):
    p = Path(file_path)
    assert p.is_file()

def run_mmpose(img, frame_id, det_model, pose_model, visualizer, df_predict_mmpose):
        
    scope = det_model.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(det_model, img)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]
    # mmdet_results = inference_detector(det_model, img)

    # predict keypoints
    pose_results = inference_topdown(pose_model, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    visualizer.add_datasample(
        'result',
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=True,
        show=False,
        wait_time=0,
        out_file="./videos_matchs/test.jpg",
        kpt_thr=0.3)
    
    keypoints = data_samples.pred_instances['keypoints']
    keypoints_scores = data_samples.pred_instances['keypoint_scores']

    # Expand the second array's dimensions to (10, 17, 1) for broadcasting
    keypoints_scores = keypoints_scores[:, :, np.newaxis]

    # Merge the arrays by concatenating along the last dimension

    keypoints_with_scores = np.concatenate((keypoints, keypoints_scores), axis=2)

    df_predict_mmpose.loc[len(df_predict_mmpose)] = [frame_id, keypoints_with_scores]

    return mmcv.rgb2bgr(visualizer.get_image())
 
DET_CONFIG = "./configs/faster_rcnn_r50_fpn_coco.py"
POSE_CONFIG = './configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
#DET_CHECKPOINT = "./checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
DET_CHECKPOINT = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
POSE_CHECKPOINT = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
#POSE_CHECKPOINT = "./checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"

# optional
DEVICE="cuda:0"
#DEVICE="cpu"
DET_CAT_ID=1
BBOX_THR=0.3
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# assert_file_exists(DET_CONFIG)
# # assert_file_exists(DET_CHECKPOINT)
# assert_file_exists(POSE_CONFIG)
# # assert_file_exists(POSE_CHECKPOINT)

# build the detector and pose model from a config file and a checkpoint file
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=DEVICE.lower())
pose_model = init_pose_estimator(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE.lower(),cfg_options=cfg_options)

# init visualizer
pose_model.cfg.visualizer.radius = 3
pose_model.cfg.visualizer.line_width = 1
visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
# the dataset_meta is loaded from the checkpoint and
# then pass to the model in init_pose_estimator
visualizer.set_dataset_meta(pose_model.dataset_meta)


