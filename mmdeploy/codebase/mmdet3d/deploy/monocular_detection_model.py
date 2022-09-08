# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence, Union

import mmcv
import torch
from mmcv.utils import Registry
from torch.nn import functional as F

from mmdeploy.codebase.base import BaseBackendModel
from mmdeploy.core import RewriterContext
from mmdeploy.utils import (Backend, get_backend, get_codebase_config,
                            get_root_logger, load_config)

import numpy as np
from mmdet3d.core import (box3d_multiclass_nms, limit_period, points_img2cam,
                          xywhr2xyxyr)
from mmcv.runner import force_fp32
from mmdet3d.models import build_head



def __build_backend_monocular_model(cls_name: str, registry: Registry, *args,
                                **kwargs):
    return registry.module_dict[cls_name](*args, **kwargs)


__BACKEND_MODEL = mmcv.utils.Registry(
    'backend_monocular_detectors', build_func=__build_backend_monocular_model)


@__BACKEND_MODEL.register_module('end2end')
class MonocularDetectionModel(BaseBackendModel):
    """End to end model for inference of 3d voxel detection.

    Args:
        backend (Backend): The backend enum, specifying backend type.
        backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
        device (str): A string specifying device type.
        model_cfg (str | mmcv.Config): The model config.
        deploy_cfg (str|mmcv.Config): Deployment config file or loaded Config
            object.
    """

    def __init__(self,
                 backend: Backend,
                 backend_files: Sequence[str],
                 device: str,
                 model_cfg: mmcv.Config,
                 deploy_cfg: Union[str, mmcv.Config] = None):
        super().__init__(deploy_cfg=deploy_cfg)
        self.deploy_cfg = deploy_cfg
        self.model_cfg = model_cfg
        self.device = device
        self.head = build_head(model_cfg.model.bbox_head)
        self._init_wrapper(
            backend=backend, backend_files=backend_files, device=device)

    def _init_wrapper(self, backend: Backend, backend_files: Sequence[str],
                      device: str):
        """Initialize backend wrapper.

        Args:
            backend (Backend): The backend enum, specifying backend type.
            backend_files (Sequence[str]): Paths to all required backend files
                (e.g. '.onnx' for ONNX Runtime, '.param' and '.bin' for ncnn).
            device (str): A string specifying device type.
        """
        output_names = self.output_names
        self.wrapper = BaseBackendModel._build_wrapper(
            backend=backend,
            backend_files=backend_files,
            device=device,
            output_names=output_names,
            deploy_cfg=self.deploy_cfg)

    def forward(self,
                img: Sequence[torch.Tensor],
                img_metas: Sequence[dict],
                return_loss=False,
                rescale=False):
        """Run forward inference.

        Args:
            points (Sequence[torch.Tensor]): A list contains input pcd(s)
                in [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            img_metas (Sequence[dict]): A list of meta info for image(s).
            return_loss (Bool): Consistent with the pytorch model.
                Default = False.

        Returns:
            list: A list contains predictions.
        """
        input_img = img[0].contiguous()
        cam2img = torch.tensor(img_metas[0][0]['cam2img'], device=input_img.device)
        cam2img_inverse = torch.inverse(cam2img)
        outputs = self.wrapper({'img': input_img,'cam2img':cam2img,'cam2img_inverse':cam2img_inverse})
        outputs = self.wrapper.output_to_list(outputs)
        outputs = [x.squeeze(0) for x in outputs]
        outputs[0] = img_metas[0][0]['box_type_3d'](
            outputs[0], 9, origin=(0.5, 0.5, 0.5))
        outputs.pop(3)
        # outputs = [outputs[:5],outputs[5:10],outputs[10:15],outputs[15:20],outputs[20:25]]
        # bbox_outputs = self.head.get_bboxes(
        #     *outputs, img_metas[0], cfg=self.model_cfg.model.test_cfg,rescale=rescale)

        from mmdet3d.core import bbox3d2result

        bbox_img = [bbox3d2result(*outputs)]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        return bbox_list

    
    def show_result(self,
                    data: Dict,
                    result: List,
                    out_dir: str,
                    file_name: str,
                    show=False,
                    snapshot=False,
                    **kwargs):
        from mmcv.parallel import DataContainer as DC
        from mmdet3d.core import show_result
        if isinstance(data['points'][0], DC):
            points = data['points'][0]._data[0][0].numpy()
        elif mmcv.is_list_of(data['points'][0], torch.Tensor):
            points = data['points'][0][0]
        else:
            ValueError(f"Unsupported data type {type(data['points'][0])} "
                       f'for visualization!')
        pred_bboxes = result[0]['boxes_3d']
        pred_labels = result[0]['labels_3d']
        pred_bboxes = pred_bboxes.tensor.cpu().numpy()
        show_result(
            points,
            None,
            pred_bboxes,
            out_dir,
            file_name,
            show=show,
            snapshot=snapshot,
            pred_labels=pred_labels)



def build_monocular_detection_model(model_files: Sequence[str],
                                model_cfg: Union[str, mmcv.Config],
                                deploy_cfg: Union[str,
                                                  mmcv.Config], device: str):
    """Build 3d voxel object detection model for different backends.

    Args:
        model_files (Sequence[str]): Input model file(s).
        model_cfg (str | mmcv.Config): Input model config file or Config
            object.
        deploy_cfg (str | mmcv.Config): Input deployment config file or
            Config object.
        device (str):  Device to input model

    Returns:
        VoxelDetectionModel: Detector for a configured backend.
    """
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    backend = get_backend(deploy_cfg)
    model_type = get_codebase_config(deploy_cfg).get('model_type', 'end2end')

    backend_detector = __BACKEND_MODEL.build(
        model_type,
        backend=backend,
        backend_files=model_files,
        device=device,
        model_cfg=model_cfg,
        deploy_cfg=deploy_cfg)

    return backend_detector
