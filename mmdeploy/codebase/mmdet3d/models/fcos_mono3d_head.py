# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.codebase.mmdet.core.ops import ncnn_detection_output_forward
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.utils import Backend, is_dynamic_shape
from mmdet3d.core import limit_period, xywhr2xyxyr
from mmdeploy.codebase.mmdet3d.core.post_processing import box3d_multiclass_nms
from mmdeploy.codebase.mmdet3d.core.bbox import points_img2cam


@FUNCTION_REWRITER.register_rewriter('mmdet3d.models.dense_heads.fcos_mono3d_head.FCOSMono3DHead._get_bboxes_single')
def fcosmono3dhead___get_bboxes_single(
    ctx,
    self,
    cls_scores,
    bbox_preds,
    dir_cls_preds,
    attr_preds,
    centernesses,
    mlvl_points,
    input_meta,
    cfg,
    rescale=False,
):
    """Transform outputs for a single batch item into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for a single scale level
            Has shape (num_points * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for a single scale
            level with shape (num_points * bbox_code_size, H, W).
        dir_cls_preds (list[Tensor]): Box scores for direction class
            predictions on a single scale level with shape
            (num_points * 2, H, W)
        attr_preds (list[Tensor]): Attribute scores for each scale level
            Has shape (N, num_points * num_attrs, H, W)
        centernesses (list[Tensor]): Centerness for a single scale level
            with shape (num_points, H, W).
        mlvl_points (list[Tensor]): Box reference for a single scale level
            with shape (num_total_points, 2).
        input_meta (dict): Metadata of input image.
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.

    Returns:
        tuples[Tensor]: Predicted 3D boxes, scores, labels and attributes.
    """
    view = torch.tensor(input_meta['cam2img'], dtype=cls_scores[0].dtype, device=cls_scores[0].device)
    scale_factor = input_meta['scale_factor']
    cfg = self.test_cfg if cfg is None else cfg
    assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
    mlvl_centers2d = []
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []
    mlvl_attr_scores = []
    mlvl_centerness = []

    for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, points in zip(
        cls_scores, bbox_preds, dir_cls_preds, attr_preds, centernesses, mlvl_points
    ):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
        dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        attr_pred = attr_pred.permute(1, 2, 0).reshape(-1, self.num_attrs)
        attr_score = torch.max(attr_pred, dim=-1)[1]
        centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, sum(self.group_reg_dims))
        bbox_pred = bbox_pred[:, : self.bbox_code_size]
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[0] > nms_pre:
            max_scores, _ = (scores * centerness[:, None]).max(dim=1)
            _, topk_inds = max_scores.topk(nms_pre)
            points = points[topk_inds, :]
            bbox_pred = bbox_pred[topk_inds, :]
            scores = scores[topk_inds, :]
            dir_cls_pred = dir_cls_pred[topk_inds, :]
            centerness = centerness[topk_inds]
            dir_cls_score = dir_cls_score[topk_inds]
            attr_score = attr_score[topk_inds]
        # change the offset to actual center predictions
        bbox_pred[:, :2] = points - bbox_pred[:, :2]
        if rescale:
            bbox_pred[:, :2] /= bbox_pred[:, :2].new_tensor(scale_factor)
        pred_center2d = bbox_pred[:, :3].clone()
        bbox_pred[:, :3] = points_img2cam(bbox_pred[:, :3], view)
        mlvl_centers2d.append(pred_center2d)
        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)
        mlvl_attr_scores.append(attr_score)
        mlvl_centerness.append(centerness)

    mlvl_centers2d = torch.cat(mlvl_centers2d)
    mlvl_bboxes = torch.cat(mlvl_bboxes)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores)

    # change local yaw to global yaw for 3D nms
    cam2img = mlvl_centers2d.new_zeros((4, 4))
    cam2img[: view.shape[0], : view.shape[1]] = mlvl_centers2d.new_tensor(view)
    mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d, mlvl_dir_scores, self.dir_offset, cam2img)

    mlvl_bboxes_for_nms = xywhr2xyxyr(
        input_meta['box_type_3d'](mlvl_bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)).bev
    )

    mlvl_scores = torch.cat(mlvl_scores)
    padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
    # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    # BG cat_id: num_class
    mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
    mlvl_attr_scores = torch.cat(mlvl_attr_scores)
    mlvl_centerness = torch.cat(mlvl_centerness)
    # no scale_factors in box3d_multiclass_nms
    # Then we multiply it from outside
    mlvl_nms_scores = mlvl_scores * mlvl_centerness[:, None]
    results = box3d_multiclass_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_nms_scores,
        cfg.score_thr,
        cfg.max_per_img,
        cfg,
        mlvl_dir_scores,
        mlvl_attr_scores,
    )
    bboxes, scores, labels, dir_scores, attrs = results
    attrs = attrs.to(labels.dtype)  # change data type to int
    # bboxes = input_meta['box_type_3d'](bboxes.squeeze(0), box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
    # bboxes = bboxes.squeeze(0)
    # Note that the predictions use origin (0.5, 0.5, 0.5)
    # Due to the ground truth centers2d are the gravity center of objects
    # v0.10.0 fix inplace operation to the input tensor of cam_box3d
    # So here we also need to add origin=(0.5, 0.5, 0.5)
    if not self.pred_attrs:
        attrs = None

    return bboxes, scores, labels, attrs


@FUNCTION_REWRITER.register_rewriter('mmdet3d.models.dense_heads.fcos_mono3d_head.FCOSMono3DHead.get_bboxes')
def fcosmono3dhead__get_bboxes(
    ctx,
    self,
    cls_scores,
    bbox_preds,
    dir_cls_preds,
    attr_preds,
    centernesses,
    cam2img,
    cam2img_inverse,
    img_meta,
    cfg=None,
    rescale=False,
):
    deploy_cfg = ctx.cfg
    is_dynamic_flag = is_dynamic_shape(deploy_cfg)
    assert len(cls_scores) == len(bbox_preds) == len(dir_cls_preds) == len(centernesses) == len(attr_preds)
    num_levels = len(cls_scores)
    batch_size = 1

    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)

    cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    dir_cls_preds = [dir_cls_preds[i].detach() for i in range(num_levels)]
    attr_preds = [attr_preds[i].detach() for i in range(num_levels)]
    centernesses = [centernesses[i].detach() for i in range(num_levels)]

    input_meta = img_meta[0]
    scale_factor = input_meta['scale_factor']
    cfg = self.test_cfg if cfg is None else cfg

    mlvl_centers2d = []
    mlvl_bboxes = []
    mlvl_scores = []
    mlvl_dir_scores = []
    mlvl_attr_scores = []
    mlvl_centerness = []
    for cls_score, bbox_pred, dir_cls_pred, attr_pred, centerness, points in zip(
        cls_scores, bbox_preds, dir_cls_preds, attr_preds, centernesses, mlvl_points
    ):
        assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
        scores = cls_score.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_out_channels).sigmoid()
        dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]
        attr_pred = attr_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, self.num_attrs)
        attr_score = torch.max(attr_pred, dim=-1)[1]
        centerness = centerness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()

        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(batch_size, -1, sum(self.group_reg_dims))
        bbox_pred = bbox_pred[..., : self.bbox_code_size]
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and scores.shape[1] > nms_pre:
            max_scores, _ = (scores * centerness[..., None]).max(dim=2)
            max_scores = max_scores[0]
            _, topk_inds = max_scores.topk(nms_pre)
            points = points[topk_inds, :]
            bbox_pred = bbox_pred[:, topk_inds, :]
            scores = scores[:, topk_inds, :]
            dir_cls_pred = dir_cls_pred[:, topk_inds, :]
            centerness = centerness[:, topk_inds]
            dir_cls_score = dir_cls_score[:, topk_inds]
            attr_score = attr_score[:, topk_inds]
        # change the offset to actual center predictions
        bbox_pred[..., :2] = points.unsqueeze(0) - bbox_pred[..., :2]
        if rescale:
            bbox_pred[..., :2] /= bbox_pred[..., :2].new_tensor(scale_factor)
        pred_center2d = bbox_pred[..., :3].clone()
        bbox_pred[..., :3] = points_img2cam(bbox_pred[..., :3], cam2img_inverse)
        mlvl_centers2d.append(pred_center2d)
        mlvl_bboxes.append(bbox_pred)
        mlvl_scores.append(scores)
        mlvl_dir_scores.append(dir_cls_score)
        mlvl_attr_scores.append(attr_score)
        mlvl_centerness.append(centerness)
    mlvl_centers2d = torch.cat(mlvl_centers2d, dim=1)
    mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
    mlvl_dir_scores = torch.cat(mlvl_dir_scores, dim=1)

    # change local yaw to global yaw for 3D nms
    mlvl_bboxes = self.bbox_coder.decode_yaw(mlvl_bboxes, mlvl_centers2d, mlvl_dir_scores, self.dir_offset, cam2img)

    mlvl_bboxes_for_nms = mlvl_bboxes[...,[0,2,3,5,6]].clone()
    mlvl_bboxes_for_nms[...,-1] = -mlvl_bboxes_for_nms[...,-1]
    # mlvl_bboxes_for_nms = xywhr2xyxyr(mlvl_bboxes_for_nms)

    mlvl_scores = torch.cat(mlvl_scores, dim=1)
    # padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], mlvl_scores.shape[1], 1)
    # # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
    # # BG cat_id: num_class
    # mlvl_scores = torch.cat([mlvl_scores, padding], dim=-1)
    mlvl_attr_scores = torch.cat(mlvl_attr_scores, dim=1)
    mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
    # no scale_factors in box3d_multiclass_nms
    # Then we multiply it from outside
    mlvl_nms_scores = mlvl_scores * mlvl_centerness[..., None]
    return box3d_multiclass_nms(
        mlvl_bboxes,
        mlvl_bboxes_for_nms,
        mlvl_nms_scores,
        cfg.score_thr,
        cfg.nms_thr,
        cfg.max_per_img,
        mlvl_dir_scores,
        mlvl_attr_scores,
    )

    # bboxes = mlvl_bboxes
    # scores, labels = torch.max(mlvl_nms_scores, dim=-1)
    # attrs = mlvl_attr_scores
    # dir_scores = mlvl_dir_scores
    # return bboxes, scores, labels, dir_scores, attrs
    # bboxes, scores, labels, dir_scores, attrs = results
    # attrs = attrs.to(labels.dtype)  # change data type to int
    # bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5))
    # # Note that the predictions use origin (0.5, 0.5, 0.5)
    # # Due to the ground truth centers2d are the gravity center of objects
    # # v0.10.0 fix inplace operation to the input tensor of cam_box3d
    # # So here we also need to add origin=(0.5, 0.5, 0.5)
    # if not self.pred_attrs:
    #     attrs = None

    # return bboxes, scores, labels, attrs
