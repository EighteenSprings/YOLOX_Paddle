#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import paddle
import paddle.nn as nn

class IOUloss(nn.Layer):
    def __init__(self, reduction="none", loss_type="iou"):
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        assert pred.shape[-1] == target.shape[-1] == 4

        pred = pred.flatten(stop_axis=-2)
        target = target.flatten(stop_axis=-2)

        tl = paddle.maximum(
            (pred[:,:2] - pred[:,2:]/2), (target[:,:2] - target[:,2:]/2)
        )
        br = paddle.maximum(
            (pred[:,:2] + pred[:,2:]/2), (target[:, :2] + target[:, 2:]/2)
        )

        area_p = paddle.prod(pred[:, 2:], 1)
        area_g = paddle.prod(target[:, 2:], 1)

        en = (tl < br).all(axis=1).astype(tl.dtype)

        area_i = paddle.prod( br - tl, axis=1) * en
        area_u = area_p + area_g - area_i
        iou = area_i / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss =  1 - iou**2
        elif self.loss_type == "giou":
            c_tl = paddle.minimum(
                (pred[:, :2] - pred[:, 2:]/2), (target[:, :2] -  target[:,2:]/2)
            )
            c_br = paddle.maximum(
                (pred[:, :2] + pred[:, 2:]/2), (target[:, :2] + target[:, 2:]/2)
            )
            area_c = paddle.prod( c_br - c_tl, axis=1) 
            giou = iou -(area_c - area_u)/area_c.clip(1e-16)
            loss = 1 - giou.clip(min=-1., max=1.)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss



        return None