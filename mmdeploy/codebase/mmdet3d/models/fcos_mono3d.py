# Copyright (c) OpenMMLab. All rights reserved.
from mmdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.fcos_mono3d.FCOSMono3D.simple_test')
def fcosmono3d__simple_test(ctx,
                            self,
                            img,
                            cam2img,
                            cam2img_inverse,
                            img_metas,
                            rescale=False):
    """Rewrite this function to run the model directly."""
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    outs = self.bbox_head.get_bboxes(
        *outs, cam2img, cam2img_inverse, img_metas, rescale=rescale)

    return outs
