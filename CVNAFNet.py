import torch as T
import complextorch as cT
from complextorch.nn.modules.conv import CVConv2d, CVConvTranpose2d
from complextorch.nn.modules.attention.eca import CVEfficientChannelAttention2d
from complextorch.nn.modules.layernorm import CVLayerNorm
from complextorch.nn.modules.batchnorm import CVBatchNorm2d
from complextorch.nn.modules.dropout import CVDropout
from complextorch.nn import functional as cF

from torch.nn import functional as F

__all__ = ["CVNAFNet", "CVNAFEnc", "CVDownSample"]

class DynamicCVLayerNorm(T.nn.Module):
    def __init__(self):
        super(DynamicCVLayerNorm, self).__init__()

    def forward(self, x):
        # Calculate the shape dynamically based on the input tensor size
        normalized_shape = x.size(-1)
        
        # Apply LayerNorm with dynamically calculated shape
        return CVLayerNorm(normalized_shape)(x)


class SimpleGate(T.nn.Module):
    def forward(self, x):
        x1,x2 = x.chunk(2,dim=1)
        return x1*x2


class CVNAFBlock(T.nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = CVConv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = CVConv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = CVConv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        self.sca = CVEfficientChannelAttention2d(dw_channel)

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = CVConv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = CVConv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = DynamicCVLayerNorm()
        self.norm2 = DynamicCVLayerNorm()

        self.dropout1 = CVDropout(drop_out_rate) if drop_out_rate > 0. else T.nn.Identity()
        self.dropout2 = CVDropout(drop_out_rate) if drop_out_rate > 0. else T.nn.Identity()

        self.beta = T.nn.Parameter(T.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = T.nn.Parameter(T.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class CVNAFNet(T.nn.Module):

    def __init__(self, img_channel=1, width=16, middle_blk_num=1, enc_blk_nums=[1,1], dec_blk_nums=[1,1], scale=1, ):
        super().__init__()
        self.scale = scale

        self.intro = CVConv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = CVConv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.upscale = CVConvTranpose2d(in_channels=width, out_channels=img_channel, kernel_size=scale+2, padding=1, stride=scale, groups=1,
                              bias=True)
                              #explore for in_channels on ending: int(width*(2**len(enc_blk_nums))/(2**len(dec_blk_nums)))

        self.encoders = T.nn.ModuleList()
        self.decoders = T.nn.ModuleList()
        self.middle_blks = T.nn.ModuleList()
        self.ups = T.nn.ModuleList()
        self.downs = T.nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            #print(f"encoders {num}")
            self.encoders.append(
                T.nn.Sequential(
                    *[CVNAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                CVConv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            T.nn.Sequential(
                *[CVNAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                T.nn.Sequential(
                    CVConv2d(chan, chan * 2, 1, bias=False),
                    T.nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                T.nn.Sequential(
                    *[CVNAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, T.nn.Conv2d):
                # Xavier initialization for convolutional layers
                T.nn.init.xavier_uniform_(m.weight)
                # Initialize biases to zeros
                if m.bias is not None:
                    T.nn.init.zeros_(m.bias)

    
    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        # print(f"Before ending x.shape: {x.shape}")
        #tim
        #print(f"before adding to input x:{x.size()}, inp:{inp.size()}")
        #x = self.check_image_size(x) + inp
        x = self.ending(x)#+self.upscale(inp)
        # print(f"After ending x.shape: {x.shape}")

        #x = x + inp
        return x

    def check_image_size(self, x):
        #print(x.shape)
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
        
        
class CVNAFEnc(T.nn.Module):

    def __init__(self, img_channel=1, width=16, enc_blk_nums=[1,1], output_hw=16):
        super().__init__()

        self.intro = CVConv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        

        self.encoders = T.nn.ModuleList()
        self.downs = T.nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            #print(f"encoders {num}")
            self.encoders.append(
                T.nn.Sequential(
                    *[CVNAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                CVConv2d(chan, 2*chan, 4, 2)
            )
            chan = chan * 2
        self.ending = CVConv2d(in_channels=chan, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)


        self.padder_size = 2 ** len(self.encoders)
        self.pool = cT.nn.CVAdaptiveAvgPool2d(output_size=output_hw)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)
        

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.ending(x)
        x = self.pool(x)  # Pooling with kernel size to create 16x16 output
        #x = T.abs(x) #calculates radius of complex values converting from complex to real

        #x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        # print(x.shape)
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class CVDownSample(T.nn.Module):
    def __init__(self, input_size, downsample_rate=1):
        super().__init__()
        self.down = downsample_rate
        self.pool = cT.nn.CVAdaptiveAvgPool2d(input_size//downsample_rate)

    def forward(self, x):
        return self.pool(x)