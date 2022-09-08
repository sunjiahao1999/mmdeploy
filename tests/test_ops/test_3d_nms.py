# Copyright (c) OpenMMLab. All rights reserved.
import onnx
import pytest
import torch
import torch.nn as nn
from mmcv import Config
from onnx.helper import (make_graph, make_model, make_node,
                         make_tensor_value_info)

from mmdeploy.core import RewriterContext
from mmdeploy.utils.test import WrapFunction, assert_allclose
from utils import TestNCNNExporter, TestOnnxRTExporter, TestTensorRTExporter

TEST_ONNXRT = TestOnnxRTExporter()
TEST_TENSORRT = TestTensorRTExporter()
TEST_NCNN = TestNCNNExporter()

TEST_TENSORRT.check_env()
# nms_boxes = torch.tensor([[[291.1746, 316.2263, 343.5029, 347.7312, 1.],
#                             [288.4846, 315.0447, 343.7267, 346.5630, 2.],
#                             [288.5307, 318.1989, 341.6425, 349.7222, 3.],
#                             [918.9102, 83.7463, 933.3920, 164.9041, 4.],
#                             [895.5786, 78.2361, 907.8049, 172.0883, 5.],
#                             [292.5816, 316.5563, 340.3462, 352.9989, 6.],
#                             [609.4592, 83.5447, 631.2532, 144.0749, 7.],
#                             [917.7308, 85.5870, 933.2839, 168.4530, 8.],
#                             [895.5138, 79.3596, 908.2865, 171.0418, 9.],
#                             [291.4747, 318.6987, 347.1208, 349.5754, 10.]]])
# scores = torch.tensor([[[0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
#                         [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
#                         [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
#                         [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
#                         [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
#                         [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
#                         [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
#                         [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
#                         [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
#                         [0.4385, 0.6035, 0.0508, 0.0662, 0.5938]]])
nms_boxes = torch.load('./bboxes.pth')
scores = torch.load('./scores.pth')
num_classes=10
pre_topk=-1
after_topk=200
iou_threshold=0.8
score_threshold=0.05
background_label_id=-1
return_index=True

from mmdeploy.mmcv.ops import ONNXBEVNMSOp
scores_onnx = scores.permute(0, 2, 1)
selected_indices = ONNXBEVNMSOp.apply(nms_boxes, scores_onnx, iou_threshold, score_threshold)
from mmdeploy.codebase.mmdet3d.core.post_processing.box3d_nms import select_nms_index
results = select_nms_index(scores_onnx, nms_boxes, selected_indices, after_topk)

from mmdeploy.mmcv.ops import TRTBatchedBEVNMSop
batched_bev_nms = TRTBatchedBEVNMSop.apply

def wrapped_function(nms_boxes, scores):
    return batched_bev_nms(nms_boxes.unsqueeze(2), scores, num_classes, pre_topk,
                            after_topk, iou_threshold, score_threshold,
                            background_label_id,return_index)

wrapped_model = WrapFunction(wrapped_function)

with RewriterContext(cfg={}, backend=TEST_TENSORRT.backend_name, opset=11):
    TEST_TENSORRT.run_and_validate(
        wrapped_model, [nms_boxes, scores],
        'batched_rotated_nms',
        input_names=['boxes', 'scores'],
        output_names=['dets', 'labels', 'inds'],
        expected_result=None,
        save_dir=None)

@pytest.mark.parametrize('backend', [TEST_TENSORRT])
@pytest.mark.parametrize('num_classes,pre_topk,after_topk,iou_threshold,'
                         'score_threshold,background_label_id',
                         [(5, 6, 3, 0.7, 0.1, -1)])
def test_batched_rotated_nms(backend,
                             num_classes,
                             pre_topk,
                             after_topk,
                             iou_threshold,
                             score_threshold,
                             background_label_id,
                             input_list=None,
                             save_dir=None):
    backend.check_env()
    pytest.importorskip('mmrotate', reason='mmrorate is not installed.')

    if input_list is None:
        nms_boxes = torch.tensor(
            [[[291.1746, 316.2263, 343.5029, 347.7312, 1.],
              [288.4846, 315.0447, 343.7267, 346.5630, 2.],
              [288.5307, 318.1989, 341.6425, 349.7222, 3.],
              [918.9102, 83.7463, 933.3920, 164.9041, 4.],
              [895.5786, 78.2361, 907.8049, 172.0883, 5.],
              [292.5816, 316.5563, 340.3462, 352.9989, 6.],
              [609.4592, 83.5447, 631.2532, 144.0749, 7.],
              [917.7308, 85.5870, 933.2839, 168.4530, 8.],
              [895.5138, 79.3596, 908.2865, 171.0418, 9.],
              [291.4747, 318.6987, 347.1208, 349.5754, 10.]]])
        scores = torch.tensor([[[0.9577, 0.9745, 0.3030, 0.6589, 0.2742],
                                [0.1618, 0.7963, 0.5124, 0.6964, 0.6850],
                                [0.8425, 0.4843, 0.9489, 0.8068, 0.7340],
                                [0.7337, 0.4340, 0.9923, 0.0704, 0.4506],
                                [0.3090, 0.5606, 0.6939, 0.3764, 0.6920],
                                [0.0044, 0.7986, 0.2221, 0.2782, 0.4378],
                                [0.7293, 0.2735, 0.8381, 0.0264, 0.6278],
                                [0.7144, 0.1066, 0.4125, 0.4041, 0.8819],
                                [0.4963, 0.7891, 0.6908, 0.1499, 0.5584],
                                [0.4385, 0.6035, 0.0508, 0.0662, 0.5938]]])
    else:
        nms_boxes = torch.tensor(input_list[0], dtype=torch.float32)
        scores = torch.tensor(input_list[1], dtype=torch.float32)

    from mmdeploy.codebase.mmrotate.core.post_processing.bbox_nms import \
        _multiclass_nms_rotated
    expected_result = _multiclass_nms_rotated(
        nms_boxes,
        scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        pre_top_k=pre_topk + 1,
        keep_top_k=after_topk + 1)
    expected_result = (expected_result[0][:,
                                          0:-1, :], expected_result[1][:,
                                                                       0:-1])

    boxes = nms_boxes.unsqueeze(2).tile(num_classes, 1)

    from mmdeploy.mmcv.ops.nms_rotated import TRTBatchedRotatedNMSop
    batched_rotated_nms = TRTBatchedRotatedNMSop.apply

    def wrapped_function(boxes, scores):
        return batched_rotated_nms(boxes, scores, num_classes, pre_topk,
                                   after_topk, iou_threshold, score_threshold,
                                   background_label_id)

    wrapped_model = WrapFunction(wrapped_function)

    with RewriterContext(cfg={}, backend=backend.backend_name, opset=11):
        backend.run_and_validate(
            wrapped_model, [boxes, scores],
            'batched_rotated_nms',
            input_names=['boxes', 'scores'],
            output_names=['batched_rotated_nms_bboxes', 'inds'],
            expected_result=expected_result,
            save_dir=save_dir)