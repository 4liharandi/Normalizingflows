from FrEIA.modules import InvertibleModule
import torch
import torch.nn as nn

class UpSqueeze(InvertibleModule):

    def __init__(self, dims_in, dims_c=None, factor=2):
        super().__init__(dims_in, dims_c)
        self.factor = factor

    def forward(self, x, c=None, rev=False, jac=True):
        x = x[0]
        batch_size, channels, N1, N2 = x.size()
        if not rev:
            x = torch.reshape(x, shape=[batch_size, channels, N1//self.factor, self.factor, N2//self.factor, self.factor])
            x = x.permute(0, 1, 2, 4, 3, 5)
            x = torch.reshape(x, shape=[batch_size, channels*self.factor*self.factor, N1//self.factor, N2//self.factor])
        else:
            x = torch.reshape(x, shape=[batch_size, channels//self.factor**2, N1, N2, self.factor, self.factor])
            x = x.permute(0, 1, 2, 4, 3, 5)
            x = torch.reshape(x, shape=[batch_size, channels//self.factor**2, N1*self.factor, N2*self.factor])

        return (x,), 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        channels, N1, N2 = input_dims[0]
        return [(channels*self.factor*self.factor, N1//self.factor, N2//self.factor)]

class Print(InvertibleModule):

    def __init__(self, dims_in, dims_c=None, factor=2):
        super().__init__(dims_in, dims_c)
        self.factor = factor

    def forward(self, x, c=None, rev=False, jac=True):
        print(x[0].size())
        x = x[0]
        return (x,), 0.

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims

def sub_conv(ch_hidden, kernel):
    pad = kernel // 2
    return lambda ch_in, ch_out: nn.Sequential(
                                    nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                    nn.ReLU(),
                                    nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

def sub_fc(ch_hidden):
    return lambda ch_in, ch_out: nn.Sequential(
                                    nn.Linear(ch_in, ch_hidden),
                                    nn.ReLU(),
                                    nn.Linear(ch_hidden, ch_out))
