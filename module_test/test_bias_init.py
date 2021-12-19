import paddle
import paddle.nn as nn

def main():
    paddle.set_device('cpu')
    x = paddle.randn(shape=( 1, 3, 224, 224), dtype='float32')
    block = nn.Conv2D( 3, 16, kernel_size=3, stride=1, padding=1, bias_attr=None)
    for k, v in block.named_parameters():
        print(k)
        if k=='bias':
            v.set_value(paddle.ones_like(v))
    print(dict(block.named_parameters())['bias'])

if __name__=="__main__":
    main()
