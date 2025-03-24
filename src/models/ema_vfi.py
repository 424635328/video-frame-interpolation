import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 辅助函数，用于创建卷积块
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', act=nn.ReLU()):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode),
        act
    )

# 可变形卷积 (Deformable Convolution)
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    print("请安装 torchvision 以使用 DeformConv2d.")
    DeformConv2d = None

class ModulatedDeformConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ModulatedDeformConvPack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels # 修改这里！
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.offset_conv = nn.Conv2d(self.in_channels,
                                     self.groups * 3 * self.kernel_size * self.kernel_size,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.dcn_v2 = DeformConv2d(self.in_channels,  # 修改这里！
                                     self.out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=self.bias)

    def forward(self, x):
        # print(f"ModulatedDeformConvPack input shape: {x.shape}") # 打印输入形状
        offset = self.offset_conv(x)
        # print(f"Offset shape: {offset.shape}") # 打印 offset 形状
        offset_static, mask, offset_dynamic = torch.chunk(offset, 3, dim=1)
        offset = torch.cat((offset_static, offset_dynamic), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn_v2(x, offset, mask)

# EMA-VFI 模型
class EMA_VFI(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, num_blocks=3): #移除deformable_groups参数，不再控制
        super(EMA_VFI, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.deformable_groups = 8

        # 1. Feature Extraction (更深层的特征提取)
        self.feat_ext_conv1 = conv_block(self.in_channels * 2, self.mid_channels, kernel_size=3, padding=1)  # 输入是两帧拼接
        self.feat_ext_blocks = nn.Sequential(OrderedDict([
            (f'conv_block_{i}', conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)) for i in range(self.num_blocks)
        ]))

        # 2. Context Encoding (上下文编码)
        self.context_encoding = nn.Sequential(
            conv_block(self.mid_channels, self.mid_channels * 2, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 2, self.mid_channels * 4, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 4, self.mid_channels * 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1), # 全局平均池化
            nn.Flatten(),
            nn.Linear(self.mid_channels * 4, self.mid_channels) # 线性层，用于压缩特征
        )

        # 3. Motion Estimation (使用更复杂的光流估计网络)
        self.motion_estimation = nn.Sequential(
            conv_block(self.mid_channels * 2, self.mid_channels, kernel_size=3, padding=1), # 拼接特征
            conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            conv(self.mid_channels, 2, kernel_size=3, padding=1) # 输出光流
        )

         # 4. Multi-Attention Fusion (多头注意力融合)
        self.attention_blocks = nn.ModuleList([
            ModulatedDeformConvPack(self.mid_channels+3, self.mid_channels + 3, kernel_size=3, padding=1, groups=1) #取消了和deformable_groups的绑定
            for _ in range(self.num_blocks)
        ])

         # 5. Reconstruction (重建)
        self.reconstruction = nn.Sequential(
            conv_block(self.mid_channels + 3, self.mid_channels, kernel_size=3, padding=1),  # 拼接 warped feature  # 输入128->64
            conv_block(self.mid_channels, self.mid_channels // 2, kernel_size=3, padding=1), # 64 -> 32
            conv(self.mid_channels // 2, self.in_channels, kernel_size=3, padding=1),  # 32 -> 3
            nn.Tanh() # 输出 tanh 激活，将像素值缩放到 [-1, 1]
        )


    def forward(self, frame1, frame2):
        # 1. Feature Extraction
        feat_input = torch.cat([frame1, frame2], dim=1)  # 沿通道拼接
        # print(f"feat_input shape: {feat_input.shape}") # 打印 feat_input 形状
        feat = self.feat_ext_conv1(feat_input)
        # print(f"feat shape: {feat.shape}") # 打印 feat 形状
        feat = self.feat_ext_blocks(feat)
        # print(f"feat shape after feat_ext_blocks: {feat.shape}") # 打印 feat 在 feature extraction 后的形状

        # 2. Context Encoding
        context = self.context_encoding(feat)
        # print(f"context shape: {context.shape}") # 打印 context 形状

        # 3. Motion Estimation
        flow_input = torch.cat([feat, context[:, :, None, None].repeat(1, 1, feat.size(2), feat.size(3))], dim=1) # 拼接特征和上下文信息
        # print(f"flow_input shape: {flow_input.shape}") # 打印 flow_input 形状
        flow = self.motion_estimation(flow_input)
        # print(f"flow shape: {flow.shape}") # 打印 flow 形状

        # 4. Warping
        warped_feat2 = self.warp(frame2, feat, flow)
        # print(f"warped_feat2 shape: {warped_feat2.shape}") # 打印 warped_feat2 形状

        # 5. Multi-Attention Fusion
        fused_feat = torch.cat([feat, warped_feat2], dim=1)
        # print(f"fused_feat shape: {fused_feat.shape}") # 打印 fused_feat 形状
        for i, block in enumerate(self.attention_blocks):
            # print(f"Attention block {i+1} input shape: {fused_feat.shape}") # 打印每个 attention block 的输入形状
            fused_feat = block(fused_feat)
            # print(f"Attention block {i+1} output shape: {fused_feat.shape}") # 打印每个 attention block 的输出形状



        # 6. Reconstruction
        output = self.reconstruction(fused_feat)
        # print(f"reconstruction output shape: {output.shape}") # 打印 reconstruction 的输出形状
        output = (output + 1) / 2 # 将像素值从 [-1, 1] 缩放到 [0, 1]
        return output

    def warp(self, frame2, feature, flow):
        """ 使用光流扭曲 (Warp) 图像"""
        B, C, H, W = frame2.size() # 获取图像的尺寸，而不是特征的尺寸
        # 创建网格
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if feature.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flow

        # 归一化网格
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(frame2, vgrid, align_corners=True)  # 使用原始图像进行warp

        return output