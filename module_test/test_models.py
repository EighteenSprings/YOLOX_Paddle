import paddle
import paddle.nn as nn

class ModelSeq(nn.Layer):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv2D(3,16,kernel_size=3,padding=1,stride=2),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                nn.Conv2D(16,16,kernel_size=3,padding=1,stride=1),
                )

    def forward(self, x):
        return self.block(x)

class ModelList(nn.Layer):
    def __init__(self):
        super().__init__()
        self.model_list = nn.LayerList(
                [nn.Conv2D(3, 16, kernel_size=3, padding=1, stride=2)]
                )
        for _ in range(6):
            self.model_list.append(nn.Conv2D(16, 16, kernel_size=3, padding=1, stride=1))
    
    def forward(self,x):
        for conv in self.model_list:
            x = conv(x)
        return x

"""
In [4]: modelseq = ModelSeq()

In [5]: modellist = ModelList()

In [6]: %timeit y = modelseq(x)
1.1 ms ± 7.11 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

In [7]: %timeit y = modellist(x)
1.14 ms ± 3.69 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
"""