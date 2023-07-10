
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim, latent_size):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        out_channels = 384

        if latent_size == 8:
                self.inverse_conv_stack = nn.Sequential(
                        nn.Upsample(size=8, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=15, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=32, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=63, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=127, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=56, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3,
                                stride=1, padding=1)
                        )
        else:
                self.inverse_conv_stack = nn.Sequential(
                        nn.Upsample(size=4, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=8, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=15, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=32, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=63, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=127, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                                padding=2),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.2),
                        nn.Upsample(size=56, mode='bilinear'),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=64, out_channels=384, kernel_size=3,
                                stride=1, padding=1)
                        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 40, 40, 200))
    x = torch.tensor(x).float()

    # test decoder
    decoder = Decoder(40, 128, 3, 64)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out.shape)
