import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import imageio  # For reading .flo files
from tqdm import tqdm  # For progress bars
import torch.optim as optim
import yaml  # For the training config


# 辅助函数，用于创建卷积块
def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
         padding_mode='zeros'):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True,
               padding_mode='zeros', act=nn.ReLU()):
    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
             groups=groups, bias=bias, padding_mode=padding_mode),
        act
    )


# 可变形卷积 (Deformable Convolution)
try:
    from torchvision.ops import DeformConv2d
except ImportError:
    print("请安装 torchvision 以使用 DeformConv2d.")
    DeformConv2d = None


class ModulatedDeformConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True):
        super(ModulatedDeformConvPack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels  # 修改这里！
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
        # print(f"ModulatedDeformConvPack input shape: {x.shape}")  # 打印输入形状
        offset = self.offset_conv(x)
        # print(f"Offset shape: {offset.shape}")  # 打印 offset 形状
        offset_static, mask, offset_dynamic = torch.chunk(offset, 3, dim=1)
        offset = torch.cat((offset_static, offset_dynamic), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn_v2(x, offset, mask)


# EMA-VFI 模型
class EMA_VFI(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, num_blocks=3):  # 移除deformable_groups参数，不再控制
        super(EMA_VFI, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.deformable_groups = 8

        # 1. Feature Extraction (更深层的特征提取)
        self.feat_ext_conv1 = conv_block(self.in_channels * 2, self.mid_channels,
                                         kernel_size=3, padding=1)  # 输入是两帧拼接
        self.feat_ext_blocks = nn.Sequential(OrderedDict([
            (f'conv_block_{i}', conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)) for i in
            range(self.num_blocks)
        ]))

        # 2. Context Encoding (上下文编码)
        self.context_encoding = nn.Sequential(
            conv_block(self.mid_channels, self.mid_channels * 2, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 2, self.mid_channels * 4, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 4, self.mid_channels * 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(self.mid_channels * 4, self.mid_channels)  # 线性层，用于压缩特征
        )

        # 3. Motion Estimation (使用更复杂的光流估计网络)
        self.motion_estimation = nn.Sequential(
            conv_block(self.mid_channels * 2, self.mid_channels, kernel_size=3, padding=1),  # 拼接特征
            conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            conv(self.mid_channels, 2, kernel_size=3, padding=1)  # 输出光流
        )

        # 4. Multi-Attention Fusion (多头注意力融合)
        self.attention_blocks = nn.ModuleList([
            ModulatedDeformConvPack(self.mid_channels + 3, self.mid_channels + 3, kernel_size=3, padding=1,
                                    groups=1)  # 取消了和deformable_groups的绑定
            for _ in range(self.num_blocks)
        ])

        # 5. Reconstruction (重建)
        self.reconstruction = nn.Sequential(
            conv_block(self.mid_channels + 3, self.mid_channels, kernel_size=3, padding=1),  # 拼接 warped feature  # 输入128->64
            conv_block(self.mid_channels, self.mid_channels // 2, kernel_size=3, padding=1),  # 64 -> 32
            conv(self.mid_channels // 2, self.in_channels, kernel_size=3, padding=1),  # 32 -> 3
            nn.Tanh()  # 输出 tanh 激活，将像素值缩放到 [-1, 1]
        )

    def forward(self, frame1, frame2):
        # 1. Feature Extraction
        feat_input = torch.cat([frame1, frame2], dim=1)  # 沿通道拼接
        # print(f"feat_input shape: {feat_input.shape}")  # 打印 feat_input 形状
        feat = self.feat_ext_conv1(feat_input)
        # print(f"feat shape: {feat.shape}")  # 打印 feat 形状
        feat = self.feat_ext_blocks(feat)
        # print(f"feat shape after feat_ext_blocks: {feat.shape}")  # 打印 feat 在 feature extraction 后的形状

        # 2. Context Encoding
        context = self.context_encoding(feat)
        # print(f"context shape: {context.shape}")  # 打印 context 形状

        # 3. Motion Estimation
        flow_input = torch.cat([feat, context[:, :, None, None].repeat(1, 1, feat.size(2), feat.size(3))], dim=1)  # 拼接特征和上下文信息
        # print(f"flow_input shape: {flow_input.shape}")  # 打印 flow_input 形状
        flow = self.motion_estimation(flow_input)
        # print(f"flow shape: {flow.shape}")  # 打印 flow 形状

        # 4. Warping
        warped_feat2 = self.warp(frame2, feat, flow)
        # print(f"warped_feat2 shape: {warped_feat2.shape}")  # 打印 warped_feat2 形状

        # 5. Multi-Attention Fusion
        fused_feat = torch.cat([feat, warped_feat2], dim=1)
        # print(f"fused_feat shape: {fused_feat.shape}")  # 打印 fused_feat 形状
        for i, block in enumerate(self.attention_blocks):
            # print(f"Attention block {i+1} input shape: {fused_feat.shape}")  # 打印每个 attention block 的输入形状
            fused_feat = block(fused_feat)
            # print(f"Attention block {i+1} output shape: {fused_feat.shape}")  # 打印每个 attention block 的输出形状

        # 6. Reconstruction
        output = self.reconstruction(fused_feat)
        # print(f"reconstruction output shape: {output.shape}")  # 打印 reconstruction 的输出形状
        output = (output + 1) / 2  # 将像素值从 [-1, 1] 缩放到 [0, 1]
        return output

    def warp(self, frame2, feature, flow):
        """使用光流扭曲 (Warp) 图像"""
        B, C, H, W = frame2.size()  # 获取图像的尺寸，而不是特征的尺寸
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


# Helper function to read .flo files
def read_flo(file):
    """
    Read .flo file format:
    E.g., read_flo('flow10.flo')
    """
    with open(file, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            data = np.fromfile(f, np.float32, count=2 * w * h)
            flow = np.resize(data, (h, w, 2))
            return flow


def warp(x, flo):
    """
    扭曲图像。

    Args:
        x (torch.Tensor): 图像，形状为 (B, C, H, W)。
        flo (torch.Tensor): 光流，形状为 (B, 2, H, W)。

    Returns:
        torch.Tensor: 扭曲后的图像，形状为 (B, C, H, W)。
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    # 使用 nearest 模式，减少插值带来的artifact
    output = F.grid_sample(x, vgrid, mode='nearest', align_corners=True)
    return output


class FloDataset(Dataset):
    def __init__(self, data_dir, transform=None, frame_interval=1,
                 crop_size=(256, 256),
                 random_rotation=True,
                 horizontal_flip=True,
                 color_jitter=None,
                 random_grayscale=0.1):
        self.data_dir = data_dir
        self.transform = transform
        self.frame_interval = frame_interval
        self.image_pairs = []
        self.crop_size = crop_size
        self.random_rotation = random_rotation
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale

        # 遍历每个视频序列 (例如 "Beanbags")
        for video_dir in os.listdir(self.data_dir):
            video_path = os.path.join(self.data_dir, video_dir)
            if os.path.isdir(video_path):
                # 获取所有帧，并按名称排序
                flows = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.flo'))])

                #  生成 (frame0, frame_t, frame1) 对
                #  Assuming flow10.flo represents flow from frame0 to frame1
                #  We need frame0, frame1, and potentially generated frame_t as the 'target'

                if len(flows) > 0:
                    for flo_path in flows:
                        # Construct pseudo image names.  The image data is not read here.  It is only used as a reference and
                        # needs to be adjusted to fit the training process.
                        pseudo_frame0_path = os.path.join(video_path, 'frame0.png')
                        pseudo_frame1_path = os.path.join(video_path, 'frame1.png')
                        pseudo_frame_t_path = os.path.join(video_path, 'frame_t.png')

                        self.image_pairs.append(
                            (pseudo_frame0_path, pseudo_frame_t_path, pseudo_frame1_path, flo_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        frame0_path, frame_t_path, frame1_path, flo_path = self.image_pairs[idx]

        # We do not read the pseudo image because this data is not provided.
        # Instead we create a dummy images

        # Generate random image of the proper size.

        flow = read_flo(flo_path)
        H, W, _ = flow.shape

        # Create random images for frame0 and frame1
        frame0 = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        frame1 = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        frame0 = Image.fromarray(frame0)
        frame1 = Image.fromarray(frame1)

        # Generate the target frame (frame_t) by warping frame0 using the flow
        frame0_tensor = transforms.ToTensor()(frame0).unsqueeze(0).to(
            torch.float32)  # Add batch dimension
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(
            0)  # Add batch dimension and change to C x H x W
        flow_tensor = flow_tensor.float()

        # Warp frame0 using the flow to generate frame_t
        frame_t_tensor = warp(frame0_tensor, flow_tensor.to(frame0_tensor.device))

        # Convert the generated frame_t back to a PIL Image
        frame_t_tensor = frame_t_tensor.squeeze(0).clamp(0, 1)  # Remove batch dimension and clamp values between 0 and 1
        frame_t = transforms.ToPILImage()(frame_t_tensor.cpu())

        # 数据增强
        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                frame0, output_size=self.crop_size
            )
            frame0 = transforms.functional.crop(frame0, i, j, h, w)
            frame_t = transforms.functional.crop(frame_t, i, j, h, w)
            frame1 = transforms.functional.crop(frame1, i, j, h, w)

        if self.random_rotation:
            angle = transforms.RandomRotation.get_params([-180, 180])  # 随机角度
            frame0 = transforms.functional.rotate(frame0, angle)
            frame_t = transforms.functional.rotate(frame_t, angle)
            frame1 = transforms.functional.rotate(frame1, angle)

        if self.horizontal_flip:
            if random.random() > 0.5:  # 使用 random.random()
                frame0 = transforms.functional.hflip(frame0)
                frame_t = transforms.functional.hflip(frame_t)
                frame1 = transforms.functional.hflip(frame1)

        # 应用色彩抖动 (如果在 __init__ 中定义了)
        if self.color_jitter:
            color_jitter = transforms.ColorJitter(**self.color_jitter)  # 使用解包运算符
            frame0 = color_jitter(frame0)
            frame_t = color_jitter(frame_t)
            frame1 = color_jitter(frame1)

        # 随机灰度转换
        if random.random() < self.random_grayscale:
            frame0 = transforms.functional.to_grayscale(frame0, num_output_channels=3)  # 转换为3通道灰度图
            frame_t = transforms.functional.to_grayscale(frame_t, num_output_channels=3)
            frame1 = transforms.functional.to_grayscale(frame1, num_output_channels=3)

        # 定义默认转换
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.transform:
            frame0 = self.transform(frame0)
            frame_t = self.transform(frame_t)
            frame1 = self.transform(frame1)
        else:
            frame0 = default_transform(frame0)
            frame_t = default_transform(frame_t)
            frame1 = default_transform(frame1)

        return frame0, frame_t, frame1


# Define Loss Function (Example: L1 Loss)
def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


if __name__ == '__main__':
    # Configuration
    config_path = 'config/train_config.yaml'  # Path to your YAML config file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:  # Specify encoding
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Could not find config file at {config_path}.  Make sure this file exists.")
        exit()
    except yaml.YAMLError as e:
        print(f"Error: Could not parse YAML file: {e}")
        exit()

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Directories
    train_data_dir = 'data/processed/other-gt-flow'  # Your .flo training dataset directory
    val_data_dir = config.get('val_data_dir',
                               'data/val')  # Your image-based validation dataset directory, default to 'data/val'

    # Hyperparameters
    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 1e-4)
    num_epochs = config.get('num_epochs', 10)

    # Model, Optimizer and Datasets
    model = EMA_VFI().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Datasets and DataLoaders

    # Create training dataset, using the .flo dataset
    train_dataset = FloDataset(data_dir=train_data_dir, transform=transform, crop_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Validation
    try:
        from src.utils.data_utils import VideoDataset
    except ImportError:
        print("Error: Could not import VideoDataset from src.utils.data_utils.  "
              "Make sure the 'src' directory is in your Python path and the data_utils.py file exists.")
        exit()
    val_dataset = VideoDataset(data_dir=val_data_dir, transform=transform, crop_size=(256, 256))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training Loop
    model.train()  # Set the model to training mode

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as t:  # Iterate through the data loader with tqdm
            for i, (frame0, frame_t, frame1) in enumerate(t):
                frame0, frame_t, frame1 = frame0.to(device), frame_t.to(device), frame1.to(
                    device)  # Move data to device

                optimizer.zero_grad()

                # Pass the frames through the model
                output = model(frame0, frame1)  # Get the model output
                # Calculate the loss.

                loss = l1_loss(output, frame_t)

                # Backpropogation
                loss.backward()

                # Optimization
                optimizer.step()

                # Print the loss
                total_loss += loss.item()

                t.set_postfix({"loss": loss.item()})  # Add loss to tqdm

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}")
    print("Training finished!")