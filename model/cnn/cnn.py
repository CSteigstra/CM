import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels=1, kernel_size=2, stride=2, dropout=0.1):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(num_channels, 20, kernel_size=kernel_size, stride=stride)
        self.activ = nn.ReLU()

        conv_out = lambda x: ((x - kernel_size) // stride + 1)
        out_sz = conv_out(input_size[0]) * conv_out(input_size[1]) * 20
        self.lin2 = nn.Linear(out_sz, out_sz//2)
        self.linear = nn.Linear(out_sz//2, output_size)
        # self.linear = nn.Linear((conv_out(input_size[0]) * conv_out(input_size[1])) * 20, 1)

    def forward(self, inputs):
        """Inputs have to have dimension (B, C, H, W)"""
        # print(inputs.shape)
        # assert 1 == 2
        # print(inputs.shape)
        y1 = self.activ(self.conv(inputs))
        # print(y1.shape)
        y1 = y1.view(y1.size(0), -1)
        # print(y1.shape)
        o = self.linear(self.activ(self.lin2(y1)))
        # print(o.shape)

        # return F.log_softmax(o, dim=1)
        return o

class CNN(nn.Module):
    def __init__(self, input_size, output_size, num_channels=1, kernel_size=2, stride=2, dropout=0.1):
        super(CNN, self).__init__()
        self.lin2 = nn.Linear(out_sz, out_sz//2)
        self.linear = nn.Linear(out_sz//2, output_size)
        # self.linear = nn.Linear((conv_out(input_size[0]) * conv_out(input_size[1])) * 20, 1)

    def forward(self, inputs):
        """Inputs have to have dimension (B, C, H, W)"""
        # print(inputs.shape)
        # assert 1 == 2
        # print(inputs.shape)
        y1 = self.activ(self.conv(inputs))
        # print(y1.shape)
        y1 = y1.view(y1.size(0), -1)
        # print(y1.shape)
        o = self.linear(self.activ(self.lin2(y1)))
        # print(o.shape)

        # return F.log_softmax(o, dim=1)
        return o