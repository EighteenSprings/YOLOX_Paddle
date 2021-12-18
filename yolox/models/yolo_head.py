#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .network_blocks import BaseConv, DWConv


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

        
