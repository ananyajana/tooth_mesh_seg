
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat
from curvenet_util import *

curve_config = {
        'default': [[100, 5], [100, 5], None, None, None]
    }

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            #idx = knn(x[:, 6:], k=k)
            #idx = knn(x[:, 9:12], k=k)
            idx = knn(x[:, :3], k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation.lower() == 'leakyrelu0.2':
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx


class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="anchor", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]

        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()
        #fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize =="center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]	
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points-mean)/(std + 1e-5)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points


class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)  # [b, d, k]
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x


class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):  # [b, d, g]
        return self.operation(x)


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = ConvBNReLU1D(in_channel, out_channel, 1, bias=bias)
        self.extraction = PosExtraction(out_channel, blocks, groups=groups,
                                        res_expansion=res_expansion, bias=bias, activation=activation)


    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        new_points = self.fuse(new_points)
        new_points = self.extraction(new_points)
        return new_points




class PointMLP(nn.Module):
    #def __init__(self, num_classes=50,points=2048, embed_dim=64, groups=1, res_expansion=1.0,
    def __init__(self, num_classes=50,points=16000, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],
                 #gmp_dim=64,cls_dim=64, num_channels=24, **kwargs):
                 gmp_dim=64,cls_dim=64, num_channels=3, 
                 num_channels2=6, k2=32, emb_dims2=1024, dropout2=0.5, k=32, setting='default', **kwargs):
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(6, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.classifier2 = nn.Sequential(
            nn.Conv1d(gmp_dim+de_dims[-1]+256, 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

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



        self.fp3 = PointNetFeaturePropagation2(in_channel=512 + 256, mlp=[256, 256], att=[512, 256, 128])
        self.up_cic4 = CIC(npoint=1000, radius=0.8, k=k, in_channels=256, output_channels=256, bottleneck_ratio=4)

        self.fp2 = PointNetFeaturePropagation2(in_channel=256 + 128, mlp=[128, 128], att=[256, 128, 64])
        self.up_cic3 = CIC(npoint=4000, radius=0.4, k=k, in_channels=128, output_channels=128, bottleneck_ratio=4)

        self.fp1 = PointNetFeaturePropagation2(in_channel=128 + 64, mlp=[64, 64], att=[128, 64, 32])
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
    #def forward(self, x, norm_plt, cls_label):
    def forward(self, x1, l=None):
        xyz = x1[:, :3, :].permute(0, 2, 1)
        x = torch.cat((x1[:, :3, :], x1[:, 21:24, :]), dim=1)
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]
        #print('xyz size: {}, x size: {}'.format(xyz.size(), x.size()))

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            #print('xyz size: {}, x size: {}'.format(xyz.size(), x.size()))
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)
            #print('xyz size: {}, x size: {}'.format(xyz.size(), x.size()))

        #raise ValueError("Exit!")
        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]])], dim=1)
        #print('x size after cat: ', x.size())
        #raise ValueError("Exit!")
        #x2 = x.clone()
        x2 = x
        '''
        x = self.classifier(x)
        print('x size after classifier: ', x.size())
        x = F.log_softmax(x, dim=1)
        #print('x size after log_softmax: ', x.size())
        x = x.permute(0, 2, 1)
        '''
        ##### curvenet
        batch_size = x1.size(0)
        #xyz = x1[:, 9:12, :].contiguous()
        xyz = x1[:, :3, :].contiguous()
        #print('xyz size: ', xyz.size())
        #raise ValueError("Exit!")

        l0_points = self.lpfa(xyz, xyz)

        l1_xyz, l1_points = self.cic11(xyz, l0_points)
        #print('1 l1_xyz: {}, l1_points: {}'.format(l1_xyz.size(), l1_points.size()))
        l1_xyz, l1_points = self.cic12(l1_xyz, l1_points)
        #print('2 l1_xyz: {}, l1_points: {}'.format(l1_xyz.size(), l1_points.size()))

        l2_xyz, l2_points = self.cic21(l1_xyz, l1_points)
        #print('3 l2_xyz: {}, l2_points: {}'.format(l2_xyz.size(), l2_points.size()))
        l2_xyz, l2_points = self.cic22(l2_xyz, l2_points)
        #print('4 l2_xyz: {}, l2_points: {}'.format(l2_xyz.size(), l2_points.size()))

        l3_xyz, l3_points = self.cic31(l2_xyz, l2_points)
        #print('5 l3_xyz: {}, l3_points: {}'.format(l3_xyz.size(), l3_points.size()))
        l3_xyz, l3_points = self.cic32(l3_xyz, l3_points)
        #print('6 l3_xyz: {}, l3_points: {}'.format(l3_xyz.size(), l3_points.size()))
 
        l4_xyz, l4_points = self.cic41(l3_xyz, l3_points)
        #print('7 l4_xyz: {}, l4_points: {}'.format(l4_xyz.size(), l4_points.size()))
        l4_xyz, l4_points = self.cic42(l4_xyz, l4_points)
        #print('8 l4_xyz: {}, l4_points: {}'.format(l4_xyz.size(), l4_points.size()))


        # global features
        emb1 = self.global_conv1(l3_points)
        emb1 = emb1.max(dim=-1, keepdim=True)[0] # bs, 64, 1
        emb2 = self.global_conv2(l4_points)
        emb2 = emb2.max(dim=-1, keepdim=True)[0] # bs, 128, 1

        # Feature Propagation layers

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
        #print('after self.drop1 x size: ', x.size())


        x = torch.cat((x, x2), dim=1)
        #print('x size after pointnet and curvenet feature cat:', x.size())
        x = self.classifier2(x)
        #print('x size after classifier2: ', x.size())
        x = F.log_softmax(x, dim=1)
        #print('x size after log_softmax: ', x.size())
        x = x.permute(0, 2, 1)

        #x = self.conv2(x)
        #x = F.log_softmax(x, dim=1)
        #x = x.permute(0, 2, 1)
        #raise ValueError("Exit!")
        return x


def pointMLP_seg(num_classes=8, num_channels=3, **kwargs) -> PointMLP:
    return PointMLP(num_classes=num_classes, points=16000, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, num_channels=num_channels,
                 num_channels2=6, k2=32, emb_dims2=1024, dropout2=0.5, k=32, setting='default', **kwargs)


if __name__ == '__main__':
    data = torch.rand(2, 24, 16000)
    print("===> testing modelD ...")
    data = data.cuda()
    model = pointMLP_seg()
    model = model.cuda()
    out = model(data)  # [2,2048,50]
    print(out.shape)
