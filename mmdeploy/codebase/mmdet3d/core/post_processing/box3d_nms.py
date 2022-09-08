import numba
import numpy as np
import torch
from mmcv.ops import nms, nms_rotated
from torch import Tensor

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.mmcv.ops import ONNXBEVNMSOp, TRTBatchedBEVNMSop


def select_nms_index(scores, bboxes, nms_index, keep_top_k, dir_scores=None, attr_scores=None):
    """Transform NMSRotated output.

    Args:
        scores (Tensor): The detection scores of shape
            [N, num_classes, num_boxes].
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 6].
        nms_index (Tensor): NMS output of bounding boxes indexing.
        batch_size (int): Batch size of the input image.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(0)
    bboxes = bboxes[batch_inds, box_inds, ...].unsqueeze(0)
    labels = cls_inds.unsqueeze(0)

    # sort
    is_use_topk = keep_top_k > 0 and (torch.onnx.is_in_onnx_export() or keep_top_k < scores.shape[1])
    if is_use_topk:
        scores, topk_inds = scores.topk(keep_top_k, dim=1)
    else:
        scores, topk_inds = scores.sort(dim=1, descending=True)
    bboxes = torch.gather(bboxes, 1, topk_inds.unsqueeze(2).repeat(1, 1, bboxes.shape[2]))
    labels = torch.gather(labels, 1, topk_inds)
    if dir_scores is not None:
        dir_scores = torch.gather(dir_scores, 1, topk_inds)
    if attr_scores is not None:
        attr_scores = torch.gather(attr_scores, 1, topk_inds)
    nms_index = nms_index[topk_inds.reshape(-1),2]

    return (bboxes, scores, labels, dir_scores, attr_scores, nms_index)


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.codebase.mmdet3d.core.post_processing.box3d_nms._box3d_multiclass_nms', backend='tensorrt'
)
# This function duplicates functionality of mmcv.ops.iou_3d.nms_bev
# from mmcv<=1.5, but using cuda ops from mmcv.ops.nms.nms_rotated.
# Nms api will be unified in mmdetection3d one day.
def box3d_multiclass_nms__tensorrt(ctx,
    mlvl_bboxes,
    mlvl_bboxes_for_nms,
    mlvl_scores,
    score_thr,
    nms_thr,
    max_num,
    mlvl_dir_scores=None,
    mlvl_attr_scores=None,
    mlvl_bboxes2d=None,
):
    """Multi-class NMS for 3D boxes. The IoU used for NMS is defined as the 2D
    IoU between BEV boxes.

    Args:
        mlvl_bboxes (torch.Tensor): Multi-level boxes with shape (N, M).
            M is the dimensions of boxes.
        mlvl_bboxes_for_nms (torch.Tensor): Multi-level boxes with shape
            (N, 5) ([x1, y1, x2, y2, ry]). N is the number of boxes.
            The coordinate system of the BEV boxes is counterclockwise.
        mlvl_scores (torch.Tensor): Multi-level boxes with shape
            (N, C + 1). N is the number of boxes. C is the number of classes.
        score_thr (float): Score threshold to filter boxes with low
            confidence.
        max_num (int): Maximum number of boxes will be kept.
        cfg (dict): Configuration dict of NMS.
        mlvl_dir_scores (torch.Tensor, optional): Multi-level scores
            of direction classifier. Defaults to None.
        mlvl_attr_scores (torch.Tensor, optional): Multi-level scores
            of attribute classifier. Defaults to None.
        mlvl_bboxes2d (torch.Tensor, optional): Multi-level 2D bounding
            boxes. Defaults to None.

    Returns:
        tuple[torch.Tensor]: Return results after nms, including 3D
            bounding boxes, scores, labels, direction scores, attribute
            scores (optional) and 2D bounding boxes (optional).
    """
    # do multi class nms
    # the fg class id range: [0, num_classes-1]
    num_classes = int(mlvl_scores.shape[-1])
    torch.save(mlvl_bboxes_for_nms,'./bboxes.pth')
    torch.save(mlvl_scores,'./scores.pth')
    mlvl_bboxes_for_nms = mlvl_bboxes_for_nms.unsqueeze(2)
    dets, labels, selected = TRTBatchedBEVNMSop.apply(
        mlvl_bboxes_for_nms, mlvl_scores, num_classes, -1, max_num, nms_thr, score_thr
    )
    selected = selected.squeeze(0)
    bboxes = mlvl_bboxes[:, selected, :]
    scores = mlvl_scores[:, selected, labels.squeeze(0)]
    dir_scores = mlvl_dir_scores[:, selected]
    attr_scores = mlvl_attr_scores[:, selected]

    results = (bboxes, scores, labels)

    if mlvl_dir_scores is not None:
        results = results + (dir_scores,)
    if mlvl_attr_scores is not None:
        results = results + (attr_scores,)
    # if mlvl_bboxes2d is not None:
    #     results = results + (bboxes2d, )

    return results


def _box3d_multiclass_nms(
    bboxes,
    bboxes_for_nms,
    scores,
    score_thr,
    nms_thr,
    max_num,
    dir_scores=None,
    attr_scores=None,
):
    """NMSRotated for multi-class bboxes.

    This function helps exporting to onnx with batch and multiclass NMSRotated
    op. It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 5).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 5].
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes].
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5.
        score_threshold (float): bbox threshold, bboxes with scores lower than
            it will not be considered.
        pre_top_k (int): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.

    Returns:
        tuple[Tensor, Tensor]: (dets, labels), `dets` of shape [N, num_det, 6]
            and `labels` of shape [N, num_det].
    """
    scores = scores.permute(0, 2, 1)
    selected_indices = ONNXBEVNMSOp.apply(bboxes_for_nms, scores, nms_thr, score_thr)

    return select_nms_index(scores, bboxes, selected_indices, max_num, dir_scores, attr_scores)


@FUNCTION_REWRITER.register_rewriter(func_name='mmdet3d.core.post_processing.box3d_multiclass_nms')
def box3d_multiclass_nms(*args, **kwargs):
    """Wrapper function for `_multiclass_nms`."""
    return mmdeploy.codebase.mmdet3d.core.post_processing.box3d_nms._box3d_multiclass_nms(*args, **kwargs)
