#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from functools import partial
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# for test
from losses import IOUloss
from network_blocks import BaseConv, DWConv
# for package
# from .losses import IOUloss
# from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Layer):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors= 1
        self.num_classes = num_classes
        self.decode_in_inference = True

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()

        self.stems = nn.LayerList()

        Conv = DWConv if depthwise else BaseConv
        Conv_Mid = partial(
            Conv,
            in_channels=int(256*width),
            out_channels=int(256*width),
            ksize=3,
            stride=1,
            act=act
        )

        # construct decoupled head
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i]*width),
                    out_channels=int(256*width),
                    ksize=1,
                    stride=1,
                    act=act
                )
            )

            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv_Mid(),
                        Conv_Mid()
                    ]
                )
            )

            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv_Mid(),
                        Conv_Mid()
                    ]
                )
            )

            self.cls_preds.append(
                nn.Conv2D(
                    in_channels=int(256*width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

            self.reg_preds.append(
                nn.Conv2D(
                    in_channels=int(256*width),
                    out_channels=self.n_anchors*4,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

            self.obj_preds.append(
                nn.Conv2D(
                    in_channels=int(256*width),
                    out_channels=self.n_anchors*1,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides= strides
        self.grids = [paddle.zeros([1])] * len(in_channels)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = paddle.concat([reg_output, obj_output, cls_output], axis=1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype
                )
                x_shifts.append(grid[:,:,0])
                y_shifts.append(grid[:,:,1])
                expanded_strides.append(
                    paddle.full(shape=[1, grid.shape[1]], fill_value=stride_this_level, dtype=xin[0].dtype)
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.reshape(
                        [batch_size, 4, self.n_anchors, hsize, wsize]
                    )
                    reg_output = reg_output.transpose([0, 2, 3, 4, 1]).reshape(
                        [batch_size, -1, 4]
                    )
                    origin_preds.append(reg_output.clone())
            
            else:
                output = paddle.concat(
                    [reg_output, F.sigmoid(obj_output), F.sigmoid(cls_output)], axis=1
                )
        
            outputs.append(output)
        

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                paddle.concat(outputs, axis=1),
                origin_preds,
                dtype=xin[0].dtype
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = paddle.concat(
                [x.flatten(start_axis=2) for x in outputs], axis=2
            ).transpose([0, 2, 1])
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = self.n_anchors*(5 + self.num_classes)
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack([xv, yv], axis=2).reshape([1,1,hsize, wsize, 2]).astype(dtype)
            self.grids[k] = grid
        output = output.reshape([batch_size, n_ch, self.n_anchors, hsize, wsize])
        output = output.transpose([0, 2, 3, 4, 1]).reshape(
            [batch_size, self.n_anchors * hsize * wsize, -1]
        )
        grid = grid.reshape([1, -1, 2])
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = paddle.exp(output[..., 2:4]) * stride
        
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack([xv, yv], axis=2).reshape([1,-1,2])
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(paddle.full(shape=(*shape, 1), fill_value = stride))
        
        grids = paddle.concat(grids, axis=1).astype(dtype)
        strides = paddle.concat(strides, axis=1).astype(dtype)

        outputs[...,:2] = (outputs[..., :2] + grids) * strides
        outputs[...,2:4] = paddle.exp(outputs[...,2:4]) * strides

        return outputs

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt
    ):
        expanded_strides_per_image = expanded_strides[0]
        
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image

        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile([num_gt, 1])
        )
        y_certers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .tile([num_gt, 1])
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .tile([1, total_num_anchors])
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_certers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_certers_per_image

        # print([x.shape for x in [b_l, b_t, b_r, b_b]])
        #
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], axis=2)

        is_in_boxes = bbox_deltas.min(axis=-1) > 0.
        is_in_boxes_all = is_in_boxes.sum(axis=0) > 0

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
                [1, total_num_anchors]
            ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
                [1, total_num_anchors]
            ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
                [1, total_num_anchors]
            ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
                [1, total_num_anchors]
            ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        
        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_certers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_certers_per_image

        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], axis=2)
        
        is_in_centers = center_deltas.min(axis=-1) > 0.
        is_in_centers_all = is_in_centers.sum(axis=0) > 0.

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        print(is_in_boxes_anchor.dtype)

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
        
        



    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs
    ):
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt
        )


    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype
    ):
        bbox_preds = outputs[:,:,:4]
        obj_preds = outputs[:,:,4].unsqueeze(-1)
        cls_preds = outputs[:,:,5:]

        nlabel = (labels.sum(axis=2)>0).sum(axis=1)

        total_num_anchors = outputs.shape[1]
        x_shifts = paddle.concat(x_shifts, 1)
        y_shifts = paddle.concat(y_shifts, 1)
        expanded_strides = paddle.concat(expanded_strides, 1)

        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.
        num_gts = 0.

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            # test
            num_gt = 10
            if num_gt == 0:
                print("aaaa")
                cls_target = paddle.zeros([0, self.num_classes], dtype=outputs.dtype)
                reg_target = paddle.zeros([0, 4], dtype=outputs.dtype)
                l1_target = paddle.zeros([0,4], dtype=outputs.dtype)
                obj_target = paddle.zeros([total_num_anchors, 1], dtype=outputs.dtype)
                fg_mask = paddle.zeros([total_num_anchors]).astype(paddle.bool)
            else:
                print("bbbb")
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                print("bbbb")

                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    batch_idx,
                    num_gt,
                    total_num_anchors,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    imgs
                )
            





            
            
if __name__ == "__main__":
    xin = [paddle.randn([1,256, 80, 80]), paddle.randn([1, 512, 40, 40]), paddle.randn([1, 1024, 20, 20])]
    xin = [x*10 for x in xin]
    labels = paddle.randint(0, 1, shape=[4, 16, 5]).astype('float')
    model = YOLOXHead(num_classes=80)
    # model.eval()
    outs = model(xin, labels)
    print(outs is None)