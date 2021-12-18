#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from functools import partial

def get_activation(name="silu"):
    if name == "silu":
        act =  F.selu
    elif name == "relu":
        act = F.relu
    elif name == "lrelu":
        act =  partial(F.leaky_relu, negative_slope=0.1)
    else:
        raise AttributeError("Unspupported act type: {}".format(name))
    return act

class BaseConv(nn.Layer):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        pad = (ksize - 1)//2
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=bias
        )
        self.bn = nn.BatchNorm2D(out_channels)
        self.act = get_activation(act)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Layer):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels=in_channels,
            out_channels=in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act
        )
        self.pconv = BaseConv(
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=1,
            stride=1,
            groups=1,
            act=act
        )
    
    def forward(self, x):
        return self.pconv(self.dconv(x))

if __name__ =="__main__":
    # test baseconv
    paddle.set_device("cpu")
    x = paddle.randn(shape=(1,3, 224, 224), dtype="float32")
    baseconv = BaseConv(3, 16, 3, 2)
    out = baseconv(x)
    print(f"BaseConv: in.shape = {x.shape}, out.shape = {out.shape}")
    paddle.summary(baseconv, input_size=(1, 3, 224, 224))
    """
    Total params: 496
    Trainable params: 432
    Non-trainable params: 64
    ---------------------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 3.06
    Params size (MB): 0.00
    Estimated Total Size (MB): 3.64
    """
    dwconv = DWConv(3, 16, 3, 2)
    out = dwconv(x)
    print(f"DWConv in.shape = {x.shape}, out.shape = {out.shape}")
    paddle.summary(dwconv, input_size=(1, 3, 224, 224))
    """
    Total params: 151
    Trainable params: 75
    Non-trainable params: 76
    ---------------------------------------------------------------------------
    Input size (MB): 0.57
    Forward/backward pass size (MB): 5.46
    Params size (MB): 0.00
    Estimated Total Size (MB): 6.03
    """