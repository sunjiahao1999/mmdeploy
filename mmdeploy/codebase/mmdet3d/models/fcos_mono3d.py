# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdeploy.core import FUNCTION_REWRITER, mark
from mmdeploy.utils import is_dynamic_shape




@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.fcos_mono3d.FCOSMono3D.simple_test')
def fcosmono3d__simple_test(ctx,
                           self,
                            img, img_metas, rescale=False):
    """Rewrite this function to run the model directly."""
    x = self.extract_feat(img)
    cls_scores, bbox_preds, dir_cls_preds, attr_preds, centerness = self.bbox_head(x)

    return cls_scores, bbox_preds, dir_cls_preds, attr_preds, centerness
