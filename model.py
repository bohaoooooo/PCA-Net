import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from einops import rearrange
from torch.hub import load_state_dict_from_url

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class Cross_transformer_backbone(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer_backbone, self).__init__()
        
        self.to_key = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.to_value = nn.Linear(in_channels * 2, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.cam_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.cam_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.cam_layer2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input_feature, features):
        Query_features = input_feature
        Query_features = self.cam_layer0(Query_features)       
        key_features = self.cam_layer1(features)
        value_features = self.cam_layer2(features)
        
        QK = torch.einsum("nlhd,nshd->nlsh", Query_features, key_features)
        softmax_temp = 1. / Query_features.size(3)**.5
        A = torch.softmax(softmax_temp * QK, dim=2)
        queried_values = torch.einsum("nlsh,nshd->nlhd", A, value_features).contiguous()
        message = self.mlp(torch.cat([input_feature, queried_values], dim=1))
        
        return input_feature + message

class Cross_transformer(nn.Module):
    def __init__(self, in_channels = 48):
        super(Cross_transformer, self).__init__()
        self.faLinears = nn.ModuleList([nn.Linear(in_channels , in_channels, bias=False) for i in range(9)])
        self.fbLinears = nn.ModuleList([nn.Linear(in_channels , in_channels, bias=False) for i in range(9)])
        self.fcLinears = nn.ModuleList([nn.Linear(in_channels , in_channels, bias=False) for i in range(9)])
        self.fdLinears = nn.ModuleList([nn.Linear(in_channels , in_channels, bias=False) for i in range(9)])
        self.fa_all = nn.Linear(in_channels , in_channels, bias=False)
        self.fb_all = nn.Linear(in_channels, in_channels, bias=False)
        self.fc_all = nn.Linear(in_channels , in_channels, bias=False)
        self.fd_all = nn.Linear(in_channels, in_channels, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.to_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma_cam_lay3 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_0 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_1 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_2 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_3 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_4 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_5 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_6 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_7 = nn.Parameter(torch.zeros(1))
        self.gamma_cam_lay3_8 = nn.Parameter(torch.zeros(1))

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse0 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse1 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse4 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse5 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse6 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse7 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.fuse8 = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def attention_layer(self, q, k, v, m_batchsize, C, height, width):
        q = q.permute(0, 2, 1)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        out = torch.matmul(v, attention)
        out = out.view(m_batchsize, C, height, width)
        
        return out

    def forward(self, input_feature, features):
        m_batchsize, C, height, width = input_feature.size()
        
        # full- abcd
        fa_all = input_feature
        fb_all = features[0]
        fc_all = features[1]
        fd_all = features[2]
        
        fa_all = self.fa_all(fa_all.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fb_all = self.fb_all(fb_all.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fc_all = self.fc_all(fc_all.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        fd_all = self.fd_all(fd_all.view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        
        qkv_1_all = self.attention_layer(fa_all, fa_all, fa_all, m_batchsize, C, height, width)
        qkv_2_all = self.attention_layer(fa_all, fb_all, fb_all, m_batchsize, C, height, width)  
        qkv_3_all = self.attention_layer(fa_all, fc_all, fc_all, m_batchsize, C, height, width)
        qkv_4_all = self.attention_layer(fa_all, fd_all, fd_all, m_batchsize, C, height, width)
        
        atten_all = self.fuse(torch.cat((qkv_1_all, qkv_2_all, qkv_3_all, qkv_4_all), dim = 1))
        
         # 9 patches
        fa = [input_feature[:, :, 0:height//2, 0:width//2], input_feature[:, :, 0:height//2, width//4:width//4*3], input_feature[:, :, 0:height//2, width//2:],
              input_feature[:, :, height//4:height//4*3, 0:width//2], input_feature[:, :, height//4:height//4*3, width//4:width//4*3], input_feature[:, :, height//4:height//4*3, width//2:],
              input_feature[:, :, height//2:, 0:width//2], input_feature[:, :, height//2:,  width//4:width//4*3], input_feature[:, :, height//2:, width//2:]] 
        
        fb = [features[0][:, :, 0:height//2, 0:width//2], features[0][:, :, 0:height//2, width//4:width//4*3], features[0][:, :, 0:height//2, width//2:],
              features[0][:, :, height//4:height//4*3, 0:width//2], features[0][:, :, height//4:height//4*3, width//4:width//4*3], features[0][:, :, height//4:height//4*3, width//2:],
              features[0][:, :, height//2:, 0:width//2], features[0][:, :, height//2:, width//4:width//4*3], features[0][:, :, height//2:, width//2:]]
        
        fc = [features[1][:, :, 0:height//2, 0:width//2], features[1][:, :, 0:height//2, width//4:width//4*3], features[1][:, :, 0:height//2, width//2:],
              features[1][:, :, height//4:height//4*3, 0:width//2], features[1][:, :, height//4:height//4*3, width//4:width//4*3], features[1][:, :, height//4:height//4*3, width//2:],
              features[1][:, :, height//2:, 0:width//2], features[1][:, :, height//2:, width//4:width//4*3], features[1][:, :, height//2:, width//2:]]
        
        fd = [features[2][:, :, 0:height//2, 0:width//2], features[2][:, :, 0:height//2, width//4:width//4*3], features[2][:, :, 0:height//2, width//2:],
              features[2][:, :, height//4:height//4*3, 0:width//2], features[2][:, :, height//4:height//4*3, width//4:width//4*3], features[2][:, :, height//4:height//4*3, width//2:],
              features[2][:, :, height//2:, 0:width//2], features[2][:, :, height//2:, width//4:width//4*3], features[2][:, :, height//2:, width//2:]]

        
        # (B, C, H, W) -> (B, C, H*W)
        for i, faLinear, fbLinear, fcLinear, fdLinear in zip(range(9), self.faLinears, self.fbLinears, self.fcLinears, self.fdLinears) :
            fa[i] = faLinear(fa[i].contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
            fb[i] = fbLinear(fb[i].contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
            fc[i] = fcLinear(fc[i].contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
            fd[i] = fdLinear(fd[i].contiguous().view(m_batchsize, C, -1).permute(0, 2, 1)).permute(0, 2, 1)
        
        atten_map = []
        for a, b, c, d in zip(fa, fb, fc, fd) :
            qkv_1 = self.attention_layer(a, a, a, m_batchsize, C, height//2, width//2)
            qkv_2 = self.attention_layer(a, b, b, m_batchsize, C, height//2, width//2)  
            qkv_3 = self.attention_layer(a, c, c, m_batchsize, C, height//2, width//2)
            qkv_4 = self.attention_layer(a, d, d, m_batchsize, C, height//2, width//2)
        
            atten = torch.cat((qkv_1, qkv_2, qkv_3, qkv_4), dim = 1)
            atten_map.append(atten)
        
        # out = input_feature + self.gamma_cam_lay3 * atten_all
        out = input_feature + self.gamma_cam_lay3 * atten_all
        out[:, :, 0:height//2, 0:width//2] += self.gamma_cam_lay3_0 * self.fuse0(atten_map[0])
        out[:, :, 0:height//2, width//4:width//4*3] += self.gamma_cam_lay3_1 * self.fuse1(atten_map[1])
        out[:, :, 0:height//2, width//2:] += self.gamma_cam_lay3_2 * self.fuse2(atten_map[2])
        out[:, :, height//4:height//4*3, 0:width//2] += self.gamma_cam_lay3_3 * self.fuse3(atten_map[3])
        out[:, :, height//4:height//4*3, width//4:width//4*3] += self.gamma_cam_lay3_4 * self.fuse4(atten_map[4])
        out[:, :, height//4:height//4*3, width//2:] += self.gamma_cam_lay3_5 * self.fuse5(atten_map[5])
        out[:, :, height//2:, 0:width//2] += self.gamma_cam_lay3_6 * self.fuse6(atten_map[6])
        out[:, :, height//2:, width//4:width//4*3] += self.gamma_cam_lay3_7 * self.fuse7(atten_map[7])
        out[:, :, height//2:, width//2:] += self.gamma_cam_lay3_8 * self.fuse8(atten_map[8])
        
        out = self.to_out(out)

        return out
        
class Scale_aware(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(Scale_aware, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()
        
    def forward(self, scene_feature_1x1, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]

        return content_feats
    
class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]

        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


    
class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self.channe1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )            
        self.channe2 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 
        self.channe3 = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 
        self.channe4 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        ) 

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)  # H,W 480->240
        x = self.bn1(x)
        x0 = self.relu(x)
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3)
        x1 = self.channe1(x1)
        x2 = self.channe2(x2)
        x3 = self.channe3(x3)
        x4 = self.channe4(x4)
        return [x1, x2, x3, x4] 

    def forward(self, x):
        return self._forward_impl(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            dilation = 1

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out    
    
    
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=False)
    return model    
    
def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)    
    

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

class Change_detection(nn.Module):
    # Implementing only the object path
    def __init__(self, num_classes=2, use_aux=True, fpn_out=48, **_):
        super(Change_detection, self).__init__()
        
        num_channels = [64, 128, 256, 512]
        self.resnet = resnet18(pretrained=True, replace_stride_with_dilation=[False,True,True])
        
        self.PPN = PSPModule(num_channels[-1])

         # scale-aware
        self.gap = GlobalAvgPool2D()

        self.sr0 = Scale_aware(in_channels = num_channels[0], channel_list = num_channels, out_channels = num_channels[0], scale_aware_proj=True)
        self.sr1 = Scale_aware(in_channels = num_channels[1], channel_list = num_channels, out_channels = num_channels[1], scale_aware_proj=True)
        self.sr2 = Scale_aware(in_channels = num_channels[2], channel_list = num_channels, out_channels = num_channels[2], scale_aware_proj=True)
        self.sr3 = Scale_aware(in_channels = num_channels[3], channel_list = num_channels, out_channels = num_channels[3], scale_aware_proj=True)

        # Cross transformer
        self.Cross_transformer0 = Cross_transformer(in_channels = num_channels[0])
        self.Cross_transformer1 = Cross_transformer(in_channels = num_channels[1])
        self.Cross_transformer2 = Cross_transformer(in_channels = num_channels[2])
        self.Cross_transformer3 = Cross_transformer(in_channels = num_channels[3])
        

        # Generate change map
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(960, fpn_out, kernel_size=3, padding=1, bias=False), # 960=64+128+256+512
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        self.output_fill = nn.Sequential(
            nn.ConvTranspose2d(fpn_out , fpn_out, kernel_size=2, stride = 2, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        )
        

    def forward(self, x):
        input_size = (x.size()[-2], x.size()[-1])

        # Backbone
        features = self.resnet(x)

        features[-1] = self.PPN(features[-1])

        H, W = features[0].size(2), features[0].size(3)
        
        # sclae-aware
        c0 = self.gap(features[0])
        c1 = self.gap(features[1])
        c2 = self.gap(features[2])
        c3 = self.gap(features[3])

        features0, features1, features2, features3 = [], [], [], []
        # #features0[:] = [F.interpolate(feature, size=(240, 240), mode='nearest') for feature in features[:]]
        # features0[:] = [F.interpolate(feature, size=(80, 80), mode='nearest') for feature in features[:]]
        # list0 = self.sr0(c0, features0[0], features0)
        # fe0 = self.Cross_transformer0(list0[0], [list0[1], list0[2], list0[3]]) 
        fe0 = features[0]

        features1[:] = [F.interpolate(feature, size=(80, 80), mode='nearest') for feature in features[:]]
        list1 = self.sr1(c1, features1[1], features1)
        fe1 = self.Cross_transformer1(list1[1], [list1[0], list1[2], list1[3]]) 

        features2[:] = [F.interpolate(feature, size=(40, 40), mode='nearest') for feature in features[:]]
        list2 = self.sr2(c2, features2[2], features2)
        fe2 = self.Cross_transformer2(list2[2], [list2[0], list2[1], list2[3]]) 

        features3[:] = [F.interpolate(feature, size=(40, 40), mode='nearest') for feature in features[:]]
        list3 = self.sr3(c3, features3[3], features3)
        fe3 = self.Cross_transformer3(list3[3], [list3[0], list3[1], list3[2]]) 
        
        refined_fpn_feat_list = [fe0, fe1, fe2, fe3]
    
        # Upsampling 
        refined_fpn_feat_list[0] = F.interpolate(refined_fpn_feat_list[0], scale_factor=1, mode='nearest')
        refined_fpn_feat_list[1] = F.interpolate(refined_fpn_feat_list[1], scale_factor=1, mode='nearest')
        refined_fpn_feat_list[2] = F.interpolate(refined_fpn_feat_list[2], scale_factor=2, mode='nearest')
        refined_fpn_feat_list[3] = F.interpolate(refined_fpn_feat_list[3], scale_factor=2, mode='nearest')

        # Generate crack map
        output = self.conv_fusion(torch.cat((refined_fpn_feat_list), dim=1))  
        output = self.output_fill(output)

        return output