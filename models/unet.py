from base import BaseModel
import torch
import torch.nn as nn
from models import resnet
import torchvision.models as models
import torch.nn.functional as F
from itertools import chain

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.down_conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled

class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_copy, x, interpolate=True):
        x = self.up(x)
        if interpolate:
            # Iterpolating instead of padding gives better results
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode="bilinear", align_corners=True)
        else:
            # Padding in case the incomping volumes are of different sizes
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x

class DecoderRes(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderRes, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
            # nn.PixleShuffle(upscale_factor=2),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # 最近邻（nearest），线性插值（linear），双线性插值（bilinear），三次线性插值（trilinear），默认是最近邻（nearest）
            
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, *args):
        if len(args) > 1:
            x = torch.cat(args, 1)
        else:
            x = args[0]
        x = self.block(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block2(x)



class UNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(UNet, self).__init__()
        self.down1 = encoder(in_channels, 64)
        self.down2 = encoder(64, 128)
        self.down3 = encoder(128, 256)
        self.down4 = encoder(256, 512)
        self.middle_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.up1 = decoder(1024, 512)
        self.up2 = decoder(512, 256)
        self.up3 = decoder(256, 128)
        self.up4 = decoder(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self._initialize_weights()
        if freeze_bn: self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)
        x = self.middle_conv(x)
        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class ResUnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ResUnet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, "resnet50")(True, norm_layer=norm_layer, deep_base=True, dilated=False)
        
        self.down1 = nn.Sequential(model.conv1, 
                                    model.bn1,
                                    model.relu) # 128
        self.down2 = nn.Sequential(model.maxpool,
                                    model.layer1) # 256
        self.down3 = model.layer2 # 512
        self.down4 = model.layer3 # 1024
        self.middle_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DecoderRes(1024, 512)
        )
        self.up1 = DecoderRes(512+1024, 512)
        self.up2 = DecoderRes(512+512, 256)
        self.up3 = DecoderRes(256+256, 128)
        self.up4 = DecoderRes(128+128, 64)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64+128+256+512, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
        # self._initialize_weights()
        # if freeze_bn: self.freeze_bn()


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        c = self.middle_conv(x4)
        d4 = self.up1(x4, c)
        d3 = self.up2(x3, d4)
        d2 = self.up3(x2, d3)
        d1 = self.up4(x1, d2)

        u4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d = torch.cat((d1, u2, u3, u4), 1)
        x = self.final_conv(d)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class MUnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(MUnet, self).__init__()
        model = models.mobilenet_v2(pretrained=True)

        self.down1 = nn.Sequential(model.features[0],
                                    model.features[1]) # 16 - 128
        self.down2 = nn.Sequential(model.features[2],
                                    model.features[3]) # 24 - 64
        self.down3 = nn.Sequential(model.features[4],
                                    model.features[5],
                                    model.features[6]) # 32 - 32
        self.down4 = nn.Sequential(model.features[7],
                                    model.features[8],
                                    model.features[9],
                                    model.features[10],
                                    model.features[11],
                                    model.features[12],
                                    model.features[13]) # 96 - 16
        self.middle_conv = nn.Sequential(
            model.features[14],
            model.features[15],
            model.features[16],
            model.features[17], # 320 - 8
            DecoderRes(320, 160)
        )
        self.up1 = DecoderRes(160+96, 96)
        self.up2 = DecoderRes(96+32, 32)
        self.up3 = DecoderRes(32+24, 24)
        self.up4 = DecoderRes(24+16, 16)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16+24+32+96, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )


    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        c = self.middle_conv(x4)
        d4 = self.up1(x4, c)
        d3 = self.up2(x3, d4)
        d2 = self.up3(x2, d3)
        d1 = self.up4(x1, d2)

        u4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d = torch.cat((d1, u2, u3, u4), 1)
        x = self.final_conv(d)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()
