import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.affga.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, upSize, angle_cls):
        """
        :param num_classes:
        :param backbone:
        :param BatchNorm:
        :param upSize: 320
        """
        super(Decoder, self).__init__()

        self.upSize = upSize
        self.angleLabel = angle_cls


        # 抓取置信度预测
        self.able_conv = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),

                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),

                                       nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                       nn.ReLU(),

                                       nn.Conv2d(128, 1, kernel_size=1, stride=1))

        # 角度预测
        self.angle_conv = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, self.angleLabel, kernel_size=1, stride=1))
        # 种类预测
        self.class_conv = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(256, 26, kernel_size=1, stride=1))

        # 抓取宽度预测
        self.width_conv = nn.Sequential(nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),

                                        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(128),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),

                                        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
                                        nn.ReLU(),

                                        nn.Conv2d(128, 1, kernel_size=1, stride=1))

        self._init_weight()


    def forward(self, outs):
        """
        :param feat_low: Res_1 的输出特征            (-1, 256, 80, 80)
        :param hasp_small: rate = {1, 6}            (-1, 256, 20, 20)
        :param hasp_big: rate = {12, 18}            (-1, 256, 20, 20)
        :param hasp_all: rate = {1, 6, 12, 18}      (-1, 256, 20, 20)
        """
        outs[2] = F.interpolate(outs[2], size=outs[0].size()[2:], mode='bilinear', align_corners=True)
        outs[1] = F.interpolate(outs[1], size=outs[0].size()[2:], mode='bilinear', align_corners=True)

        hasp_small = torch.cat((outs[1], outs[2]), dim=1)
        hasp_big = torch.cat((outs[0], hasp_small), dim=1)


        # 预测
        able_pred = self.able_conv(hasp_big)
        angle_pred = self.angle_conv(hasp_big)
        width_pred = self.width_conv(hasp_big)

        class_pred = self.class_conv(hasp_big)

        return able_pred, angle_pred, width_pred, class_pred

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm, upSize, angle_cls):
    return Decoder(num_classes, backbone, BatchNorm, upSize, angle_cls)
