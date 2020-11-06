import torch
import torch.nn as nn
import torch.nn.functional as F
from lambda_networks import LambdaLayer


class DecoderNormal(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderNormal, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )
        self._initialize_weights()

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block(x)
        return x
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DecoderSC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, attention=True):
        super(DecoderSC, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.attention = attention

        if self.attention:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.channel_gate = nn.Sequential(
                nn.Conv2d(out_channels, out_channels//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU(),
                nn.Conv2d(out_channels//2, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )
        self._initialize_weights()

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block(x)

        if self.attention:
            g2 = self.channel_gate(F.avg_pool2d(x, x.size()[2:]))

            g1 = self.spatial_gate(x)
            x = g2*x + g1*x

        return x
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class DecoderAtt(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, spatial=False, channel=False):
        super(DecoderAtt, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.spatial = spatial
        self.channel = channel

        if self.spatial:
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(out_channels, 1, kernel_size=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        if self.channel:
            self.channel_gate = nn.Sequential(
                nn.Conv2d(out_channels, out_channels//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels//2),
                nn.ReLU(),
                nn.Conv2d(out_channels//2, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid()
            )

        self._initialize_weights()

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block(x)

        if self.spatial:
            g1 = self.spatial_gate(x)
            x = g1*x
            
        if self.channel:
            g2 = self.channel_gate(F.avg_pool2d(x, x.size()[2:]))
            x = g2*x

        return x
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class DecoderLambda(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderLambda, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

        self.lambda_layer =LambdaLayer(
            dim = out_channels,
            dim_out = out_channels,
            r = 7,         # the receptive field for relative positional encoding (23 x 23)
            dim_k = 16,
            heads = 1,
            dim_u = 1
        )

        self._initialize_weights()

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block(x)
        x = self.lambda_layer(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class CAA(nn.Module):
    def __init__(self, in_channels, out_channels, num_class=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, num_class, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        b, c1, h, w = x1.shape
        x1 = torch.reshape(x1, (b, c1, h*w))

        x2 = self.conv2(x)
        b, c2, h, w = x2.shape
        x2 = torch.reshape(x2, (b, c2, h*w))

        x3 = torch.transpose(x2, 1, 2)
        x1 = F.softmax(torch.matmul(x1, x3), dim=-1)

        x2 = torch.matmul(x1, x2)
        x2 = torch.reshape(x2, (b, c1, h, w))

        return torch.cat([x, x2], dim=1)


class DecoderClass(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderClass, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(middle_channels + out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

        self.caa = CAA(middle_channels, out_channels)
        self._initialize_weights()

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.block(x)
        x = self.caa(x)
        x = self.block2(x)
        return x
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DecoderSCSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

if __name__=='__main__':
    
    caa = CAA(5, 5)

    x = torch.ones((32, 5, 16, 16))

    y = caa(x)
    print(x.shape)