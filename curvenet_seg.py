"""
@Author: Tiange Xiang
@Contact: txia7609@uni.sydney.edu.au
@File: curvenet_seg.py
@Time: 2021/01/21 3:10 PM
"""

import torch.nn as nn
import torch.nn.functional as F
from curvenet_util import *


curve_config = {
        'default': [[100, 5], [100, 5], None, None, None]
    }

class CurveNet(nn.Module):
    def __init__(self, num_classes=50, category=16, k=32, setting='default'):
        super(CurveNet, self).__init__()

        assert setting in curve_config

        additional_channel = 32
        self.lpfa = LPFA(9, additional_channel, k=k, mlp_num=1, initial=True)

        # encoder
        self.cic11 = CIC(npoint=16000, radius=0.2, k=k, in_channels=additional_channel, output_channels=64, bottleneck_ratio=2, curve_config=curve_config[setting][0])
        self.cic12 = CIC(npoint=16000, radius=0.2, k=k, in_channels=64, output_channels=64, bottleneck_ratio=4, curve_config=curve_config[setting][0])

        self.cic21 = CIC(npoint=4000, radius=0.4, k=k, in_channels=64, output_channels=128, bottleneck_ratio=2, curve_config=curve_config[setting][1])
        self.cic22 = CIC(npoint=4000, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4, curve_config=curve_config[setting][1])

        self.cic31 = CIC(npoint=1000, radius=0.8, k=k, in_channels=128, output_channels=256, bottleneck_ratio=2, curve_config=curve_config[setting][2])
        self.cic32 = CIC(npoint=1000, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4, curve_config=curve_config[setting][2])

        self.cic41 = CIC(npoint=250, radius=1.2, k=31, in_channels=256, output_channels=512, bottleneck_ratio=2, curve_config=curve_config[setting][3])
        self.cic42 = CIC(npoint=250, radius=1.2, k=31, in_channels=512, output_channels=512, bottleneck_ratio=4, curve_config=curve_config[setting][3])



        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=1000, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=4000, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation(in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32])
        self.up_cic2 = CIC(npoint=16000, radius=0.2, k=k, in_channels=128+64+64+3, output_channels=256, bottleneck_ratio=4)
        self.up_cic1 = CIC(npoint=16000, radius=0.2, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)
        

        self.global_conv2 = nn.Sequential(
            nn.Conv1d(512, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2))
        self.global_conv1 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Conv1d(256, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, num_classes, 1)
        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                nn.Conv1d(256, 256//8, 1, bias=False),
                                nn.BatchNorm1d(256//8),
                                nn.LeakyReLU(negative_slope=0.2),
                                nn.Conv1d(256//8, 256, 1, bias=False),
                                nn.Sigmoid())
                                
    def forward(self, xyz_data, l=None):
        batch_size = xyz_data.size(0)
        #xyz = xyz[:, 9:12, :].clone().detach()
        xyz = xyz_data[:, 9:12, :].contiguous()
        #xyz = xyz_data
        #print('xyz size: ', xyz.size())
        #raise ValueError("Exit!")

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        print('1 l1_xyz: {}, l1_points: {}'.format(l1_xyz.size(), l1_points.size()))
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
        print('2 l1_xyz: {}, l1_points: {}'.format(l1_xyz.size(), l1_points.size()))

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        print('3 l2_xyz: {}, l2_points: {}'.format(l2_xyz.size(), l2_points.size()))
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)
        print('4 l2_xyz: {}, l2_points: {}'.format(l2_xyz.size(), l2_points.size()))

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        print('5 l3_xyz: {}, l3_points: {}'.format(l3_xyz.size(), l3_points.size()))
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
        print('6 l3_xyz: {}, l3_points: {}'.format(l3_xyz.size(), l3_points.size()))
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        print('7 l4_xyz: {}, l4_points: {}'.format(l4_xyz.size(), l4_points.size()))
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)
        print('7 l4_xyz: {}, l4_points: {}'.format(l4_xyz.size(), l4_points.size()))

        #l5_xyz, l5_points = self.cic51(l4_xyz, l4_points)
        #l5_xyz, l5_points = self.cic52(l5_xyz, l5_points)
        #l5_xyz, l5_points = self.cic53(l5_xyz, l5_points)

        # global features
        #emb1 = self.global_conv1(l4_points)
        emb1 = self.global_conv1(l3_points)
        emb1 = emb1.max(dim=-1, keepdim=True)[0] # bs, 64, 1
        #emb2 = self.global_conv2(l5_points)
        emb2 = self.global_conv2(l4_points)
        emb2 = emb2.max(dim=-1, keepdim=True)[0] # bs, 128, 1

        # Feature Propagation layers
        #l4_points = self.fp4(l4_xyz, l5_xyz, l4_points, l5_points)
        #l4_xyz, l4_points = self.up_cic5(l4_xyz, l4_points)

        l3_points = self.fp3(l3_xyz, l4_xyz, l3_points, l4_points)
        l3_xyz, l3_points = self.up_cic4(l3_xyz, l3_points)

        l2_points = self.fp2(l2_xyz, l3_xyz, l2_points, l3_points)
        l2_xyz, l2_points = self.up_cic3(l2_xyz, l2_points)

        l1_points = self.fp1(l1_xyz, l2_xyz, l1_points, l2_points)

        if l is not None:
            l = l.view(batch_size, -1, 1)
            emb = torch.cat((emb1, emb2, l), dim=1) # bs, 128 + 16, 1
        else:
            emb = torch.cat((emb1, emb2), dim=1) # bs, 128 + 16
        l = emb.expand(-1,-1, xyz.size(-1))
        #print('l size: ', l.size())
        #print('l1_xyz size: ', l1_xyz.size())
        #print('l1_points size: ', l1_points.size())
        x = torch.cat((l1_xyz, l1_points, l), dim=1)

        xyz, x = self.up_cic2(l1_xyz, x)
        xyz, x = self.up_cic1(xyz, x)

        x =  F.leaky_relu(self.bn1(self.conv1(x)), 0.2, inplace=True)
        se = self.se(x)
        x = x * se
        x = self.drop1(x)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    #data = torch.rand(2, 24, 2048).cuda()
    #data = torch.rand(2, 3, 2048).cuda()
    #data = torch.rand(1, 3, 2048).cuda()
    #data = torch.rand(2, 3, 16000).cuda()
    data = torch.rand(2, 24, 16000).cuda()
    #data = data.permute(0, 2, 1)
    #norm = torch.rand(2, 3, 2048)
    #cls_label = torch.rand([2, 16]).cuda()
    cls_label = torch.rand([2, 16]).cuda()
    print("===> testing modelD ...")
    #model = pointMLP(50)
    #model = pointMLP_seg()
    model = CurveNet().cuda()
    #out = model(data, label_one_hot)
    #out = model(data, cls_label)
    #out = model(data, cls_label)  # [2,2048,50]
    #out = model(data, norm, cls_label)  # [2,2048,50]
    out = model(data)  # [2,2048,50]
    print(out.shape)
