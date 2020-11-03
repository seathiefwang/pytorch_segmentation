from base import BaseModel
import torch
import torch.nn as nn
from models import resnet
import torchvision.models as models
import torch.nn.functional as F
from itertools import chain
from .resnest import resnest50
from .decoder import DecoderNormal, DecoderSC, DecoderClass

Decoder = DecoderClass

class ASKCFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super().__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d()
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(),
            nn.Conv2d(inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(),
            nn.ReLU(),
            nn.Conv2d(channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d()
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_add(xa)
        xg = self.global_att(xa)
        xlg = torch.add(xl,  xg)
        wei = F.sigmoid(xlg)
        xo = 2 * torch.mul(x, wei) + 2 * torch.mul(residual, 1-wei)
        return xo

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
            DecoderSC(1024, 512, 64)
        )
        self.up1 = DecoderSC(64+1024, 512, 64)
        self.up2 = DecoderSC(64+512, 256, 64)
        self.up3 = DecoderSC(64+256, 128, 64)
        self.up4 = DecoderSC(64+128, 96, 64)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64+64+64+64, 64, kernel_size=1, padding=0),
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
            Decoder(320, 160, 80)
        )
        self.up1 = Decoder(80+96, 128, 64)
        self.up2 = Decoder(64+32, 64, 32)
        self.up3 = Decoder(32+24, 32, 16)
        self.up4 = DecoderSC(16+16, 16, 8)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(8+16+32+64, 64, kernel_size=1, padding=0),
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

class MSCUnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(MSCUnet, self).__init__()
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
            DecoderSC(320, 160, 160)
        )
        self.up1 = DecoderSC(160+96, 128, 128)
        self.up2 = DecoderSC(128+32, 64, 64)
        self.up3 = DecoderSC(64+24, 32, 32)
        self.up4 = DecoderSC(32+16, 16, 16)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16+32+64+128, 64, kernel_size=1, padding=0),
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


    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(VGGUnet, self).__init__()
        model = models.vgg13_bn(pretrained=True)

        self.down1 = nn.Sequential(model.features[0:7]) # 64 - 128
        self.down2 = nn.Sequential(model.features[7:14]) # 128 - 64
        self.down3 = nn.Sequential(model.features[14:21]) # 256 - 32
        self.down4 = nn.Sequential(model.features[21:28]) # 512 - 16
        self.middle_conv = nn.Sequential(
            model.features[28:35], # 512 - 8
            DecoderSC(512, 256, 64)
        )
        self.up1 = DecoderSC(64+512, 256, 64)
        self.up2 = DecoderSC(64+256, 128, 64)
        self.up3 = DecoderSC(64+128, 96, 64)
        self.up4 = DecoderSC(64+64, 64, 32)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32+64+64+64, 64, kernel_size=1, padding=0),
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



class ResNeStUnet(BaseModel):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, **_):
        super(ResNeStUnet, self).__init__()
        model = resnest50(True)
        self.down1 = nn.Sequential(model.conv1, 
                                    model.bn1,
                                    model.relu) # 64
        self.down2 = nn.Sequential(model.maxpool,
                                    model.layer1) # 256
        self.down3 = model.layer2 # 512
        self.down4 = model.layer3 # 1024
        self.middle_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            DecoderSC(1024, 512, 128)
        )
        self.up1 = DecoderSC(128+1024, 512, 64)
        self.up2 = DecoderSC(64+512, 256, 64)
        self.up3 = DecoderSC(64+64+256, 128, 64)
        self.up4 = DecoderSC(64+64+64+64, 128, 64, attention=False)
        # self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64+64+64+64, 64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )


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

        # up2 interpolate
        up1_1 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False)
        d3_x = torch.cat((d3, up1_1), 1)

        d2 = self.up3(x2, d3_x)

        # up3 interpolate
        up3_1 = F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=False)
        up3_2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d2_x = torch.cat((d2, up3_2, up3_1), 1)
        
        d1 = self.up4(x1, d2_x)

        u4 = F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=False)
        u3 = F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=False)
        u2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d = torch.cat((d1, u2, u3, u4), 1)
        x = self.final_conv(d)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return chain(self.down1.parameters(), self.down2.parameters(), 
                    self.down3.parameters(), self.down4.parameters())

    def get_decoder_params(self):
        return chain(self.up1.parameters(), self.up2.parameters(), self.up3.parameters(), self.up4.parameters(), 
            self.middle_conv.parameters(), self.final_conv.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


