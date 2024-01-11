from torch import nn
import torch

class ConvBlock(nn.Module):
    """Block of two downsample 3D convolution layers
    
    Attributes:
        in_channels:
        out_channels:
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,
                 use_batch_norm=True, is_bottleneck=False, pool_kernel_size=2, pool_strid=2):
        """Initialized the block"""
        super(ConvBlock, self).__init__()
        mid_channels = out_channels//2
        self.use_batch_norm = use_batch_norm
        self.is_bottleneck = is_bottleneck

        self.conv_1 = nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        if use_batch_norm: self.bn_1 = nn.BatchNorm3d(mid_channels)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding)
        if use_batch_norm: self.bn_2 = nn.BatchNorm3d(out_channels)
        self.relu_2 = nn.ReLU()
        print(f'in and out: {in_channels, out_channels}')
        if not is_bottleneck: self.pooling = nn.MaxPool3d(pool_kernel_size, stride=pool_strid)
    
    def forward(self, x):
        if self.use_batch_norm:
            res = self.relu_1(self.bn_1(self.conv_1(x)))
            res = self.relu_2(self.bn_2(self.conv_2(res)))
        else:
            res = self.relu_1(self.conv_1(x))
            res = self.relu_2(self.conv_2(res))
        
        if not self.is_bottleneck:
            out = self.pooling(res)
        else:
            out = res

        return out, res


class UpsampleBlock(nn.Module):
    """Block of two 3D upsample layers

    Attributes:
        in_channels:
        res_channels:
    """

    def __init__(self, in_channels, res_channels, up_kernel_size=2, up_stride_size=2,
                 kernel_size=3, padding=1, is_output=False, num_classes=2):
        super(UpsampleBlock, self).__init__()
        self.res_channels = res_channels
        mid_channels = in_channels//2
        self.is_output = is_output

        self.conv_trans = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=up_kernel_size, stride=up_stride_size)
        self.conv_1 = nn.Conv3d(in_channels+res_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.bn_1 = nn.BatchNorm3d(mid_channels)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding)
        self.bn_2 = nn.BatchNorm3d(mid_channels)
        self.relu_2 = nn.ReLU()
        print(f'in, res and out: {in_channels, res_channels, mid_channels}')
        if is_output:
            self.conv_3 = nn.Conv3d(mid_channels, num_classes, kernel_size=1)
            print(f'output numbner of classes: {num_classes}')
        
    def forward(self, x, res):
        #assert res.size()[0] == self.res_channels, "residual input channels not equal to res_channels!"
        out = self.conv_trans(x)
        if res is not None: 
            out = torch.cat((out, res), 1)
        out = self.relu_1(self.bn_1(self.conv_1(out)))
        out = self.relu_2(self.bn_2(self.conv_2(out)))
        if self.is_output: out = self.conv_3(out)
        return out
    

class UNet3D(nn.Module):
    """3D U-Net model
    
    Dynamic 3D U-Net model for semantic segmentation
    will auto-adjust depth and size given different block_channels.
    
    Attributes:
        in_channels: number of channels for input data.
        num_classes: number of classes to indentify, default 1
        block_channels: list or tuple, numbers of channels during downsampleing
          numbers of channels during upsampling are reversed of this list/tuple
          default [64, 128, 256, 512]
    """
    
    def __init__(self, in_channels, num_classes=1, block_channels=[64, 128, 256, 512]):
        super(UNet3D, self).__init__()
        
        # add conv blocks
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ConvBlock(in_channels, block_channels[0])) # first layer
        for i in range(len(block_channels)-2):
            self.conv_blocks.append(ConvBlock(block_channels[i], block_channels[i+1]))
        self.conv_blocks.append(ConvBlock(block_channels[-2], block_channels[-1], is_bottleneck=True)) # bottlenect block, no pooling
        
        # add upsample blocks
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(block_channels)-1, 1, -1):
            self.upsample_blocks.append(UpsampleBlock(block_channels[i], block_channels[i-1]))
        self.upsample_blocks.append(UpsampleBlock(block_channels[1], block_channels[0], is_output=True, num_classes=num_classes)) # output block

    def forward(self, input):
        out = input
        res_list = []
        for block in self.conv_blocks[:-1]:
            out, res = block(out)
            res_list.append(res)
        out = self.conv_blocks[-1](out)[0] # bottleneck block, no maxpool, res is out     
        assert len(self.upsample_blocks) == len(res_list), "number of upsample blocks and number of residuals don't match!"
        for block, res in zip(self.upsample_blocks, reversed(res_list)):
            out = block(out, res)
        
        return out

if __name__ == '__main__':
    import torchinfo
    import sys
    model = UNet3D(in_channels=1, num_classes=1)
    # summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device='cpu')
    # input_size = eval(sys.argv[1])
    torchinfo.summary(model, input_size=(1, 1, 128, 128, 128), device='cuda')
    print('-' * 10)
    # print(model)
    