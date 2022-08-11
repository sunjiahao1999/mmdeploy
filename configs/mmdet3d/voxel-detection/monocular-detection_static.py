_base_ = ['../../_base_/onnx_config.py']

onnx_config = dict(
    output_names=[
        'cls_scores_0',
        'cls_scores_1',
        'cls_scores_2',
        'cls_scores_3',
        'cls_scores_4',
        'bbox_preds_0',
        'bbox_preds_1',
        'bbox_preds_2',
        'bbox_preds_3',
        'bbox_preds_4',
        'dir_cls_preds_0',
        'dir_cls_preds_1',
        'dir_cls_preds_2',
        'dir_cls_preds_3',
        'dir_cls_preds_4',
        'attr_preds_0',
        'attr_preds_1',
        'attr_preds_2',
        'attr_preds_3',
        'attr_preds_4',
        'centerness_0',
        'centerness_1',
        'centerness_2',
        'centerness_3',
        'centerness_4',
    ],
    input_shape=None,
)
codebase_config = dict(
    type='mmdet3d',
    task='MonocularDetection',
    model_type='end2end',
    ann_file='/home/sjh/Projects/mmdetection3d/demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json',
)
