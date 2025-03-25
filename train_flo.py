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
import torchvision
import imageio
from tqdm import tqdm
import torch.optim as optim
import yaml


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
        self.out_channels = in_channels
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

        self.dcn_v2 = DeformConv2d(self.in_channels,
                                     self.out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=self.bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        offset_static, mask, offset_dynamic = torch.chunk(offset, 3, dim=1)
        offset = torch.cat((offset_static, offset_dynamic), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn_v2(x, offset, mask)


class EMA_VFI(nn.Module):
    def __init__(self, in_channels=3, mid_channels=64, num_blocks=3):
        super(EMA_VFI, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.deformable_groups = 8

        self.feat_ext_conv1 = conv_block(self.in_channels * 2, self.mid_channels,
                                         kernel_size=3, padding=1)
        self.feat_ext_blocks = nn.Sequential(OrderedDict([
            (f'conv_block_{i}', conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1)) for i in
            range(self.num_blocks)
        ]))

        self.context_encoding = nn.Sequential(
            conv_block(self.mid_channels, self.mid_channels * 2, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 2, self.mid_channels * 4, kernel_size=3, stride=2, padding=1),
            conv_block(self.mid_channels * 4, self.mid_channels * 4, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.mid_channels * 4, self.mid_channels)
        )

        self.motion_estimation = nn.Sequential(
            conv_block(self.mid_channels * 2, self.mid_channels, kernel_size=3, padding=1),
            conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1),
            conv(self.mid_channels, 2, kernel_size=3, padding=1)
        )

        self.attention_blocks = nn.ModuleList([
            ModulatedDeformConvPack(self.mid_channels + 3, self.mid_channels + 3, kernel_size=3, padding=1,
                                    groups=1)
            for _ in range(self.num_blocks)
        ])

        self.reconstruction = nn.Sequential(
            conv_block(self.mid_channels + 3, self.mid_channels, kernel_size=3, padding=1),
            conv_block(self.mid_channels, self.mid_channels // 2, kernel_size=3, padding=1),
            conv(self.mid_channels // 2, self.in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, frame1, frame2):
        feat_input = torch.cat([frame1, frame2], dim=1)
        feat = self.feat_ext_conv1(feat_input)
        feat = self.feat_ext_blocks(feat)

        context = self.context_encoding(feat)

        flow_input = torch.cat([feat, context[:, :, None, None].repeat(1, 1, feat.size(2), feat.size(3))], dim=1)
        flow = self.motion_estimation(flow_input)

        warped_feat2 = self.warp(frame2, feat, flow)

        fused_feat = torch.cat([feat, warped_feat2], dim=1)
        for i, block in enumerate(self.attention_blocks):
            fused_feat = block(fused_feat)

        output = self.reconstruction(fused_feat)
        output = (output + 1) / 2
        return output

    def warp(self, frame2, feature, flow):
        B, C, H, W = frame2.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if feature.is_cuda:
            grid = grid.cuda()

        vgrid = grid + flow

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(frame2, vgrid, align_corners=True)

        return output


def read_flo(file):
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
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)

    vgrid = grid + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
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

        for video_dir in os.listdir(self.data_dir):
            video_path = os.path.join(self.data_dir, video_dir)
            if os.path.isdir(video_path):
                flows = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.flo'))])

                if len(flows) > 0:
                    for flo_path in flows:
                        pseudo_frame0_path = os.path.join(video_path, 'frame0.png')
                        pseudo_frame1_path = os.path.join(video_path, 'frame1.png')
                        pseudo_frame_t_path = os.path.join(video_path, 'frame_t.png')

                        self.image_pairs.append(
                            (pseudo_frame0_path, pseudo_frame_t_path, pseudo_frame1_path, flo_path))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        frame0_path, frame_t_path, frame1_path, flo_path = self.image_pairs[idx]

        flow = read_flo(flo_path)
        H, W, _ = flow.shape

        frame0 = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        frame1 = np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
        frame0 = Image.fromarray(frame0)
        frame1 = Image.fromarray(frame1)

        frame0_tensor = transforms.ToTensor()(frame0).unsqueeze(0).to(
            torch.float32)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(
            0)
        flow_tensor = flow_tensor.float()

        frame_t_tensor = warp(frame0_tensor, flow_tensor.to(frame0_tensor.device))

        frame_t_tensor = frame_t_tensor.squeeze(0).clamp(0, 1)
        frame_t = transforms.ToPILImage()(frame_t_tensor.cpu())

        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                frame0, output_size=self.crop_size
            )
            frame0 = transforms.functional.crop(frame0, i, j, h, w)
            frame_t = transforms.functional.crop(frame_t, i, j, h, w)
            frame1 = transforms.functional.crop(frame1, i, j, h, w)

        if self.random_rotation:
            angle = transforms.RandomRotation.get_params([-180, 180])
            frame0 = transforms.functional.rotate(frame0, angle)
            frame_t = transforms.functional.rotate(frame_t, angle)
            frame1 = transforms.functional.rotate(frame1, angle)

        if self.horizontal_flip:
            if random.random() > 0.5:
                frame0 = transforms.functional.hflip(frame0)
                frame_t = transforms.functional.hflip(frame_t)
                frame1 = transforms.functional.hflip(frame1)

        if self.color_jitter:
            color_jitter = transforms.ColorJitter(**self.color_jitter)
            frame0 = color_jitter(frame0)
            frame_t = color_jitter(frame_t)
            frame1 = color_jitter(frame1)

        if random.random() < self.random_grayscale:
            frame0 = transforms.functional.to_grayscale(frame0, num_output_channels=3)
            frame_t = transforms.functional.to_grayscale(frame_t, num_output_channels=3)
            frame1 = transforms.functional.to_grayscale(frame1, num_output_channels=3)

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


def l1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for frame0, frame_t, frame1 in dataloader:
            frame0, frame_t, frame1 = frame0.to(device), frame_t.to(device), frame1.to(device)
            output = model(frame0, frame1)
            loss = l1_loss(output, frame_t)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    model.train()
    return avg_loss


if __name__ == '__main__':
    config_path = 'config/train_config.yaml'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {config_path}。请确认文件是否存在。")
        exit()
    except yaml.YAMLError as e:
        print(f"错误：无法解析 YAML 文件: {e}")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data_dir = 'data/processed/other-gt-flow'
    val_data_dir = config.get('val_data_dir','data/processed/val')

    batch_size = config.get('batch_size', 4)
    learning_rate = config.get('learning_rate', 1e-4)
    num_epochs = config.get('num_epochs', 10)
    output_dir = config.get('output_dir', 'output_flo')
    
    print(f"config: {config}")

    os.makedirs(output_dir, exist_ok=True)

    model = EMA_VFI().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FloDataset(data_dir=train_data_dir, transform=transform, crop_size=(256, 256))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    try:
        from src.utils.data_utils import VideoDataset
    except ImportError:
        print("错误：无法从 src.utils.data_utils 导入 VideoDataset。 "
              "请确保 'src' 目录在你的 Python 路径中，并且存在 data_utils.py 文件。")
        exit()
    val_dataset = VideoDataset(data_dir=val_data_dir, transform=transform, crop_size=(256, 256))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as t:
            for i, (frame0, frame_t, frame1) in enumerate(t):
                frame0, frame_t, frame1 = frame0.to(device), frame_t.to(device), frame1.to(
                    device)

                optimizer.zero_grad()

                output = model(frame0, frame1)
                loss = l1_loss(output, frame_t)

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

                t.set_postfix({"loss": loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_train_loss}")

        val_loss = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

        scheduler.step(val_loss)
        print(f"Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")

        with torch.no_grad():
          model.eval()
          val_frame0, val_frame_t, val_frame1 = next(iter(val_loader))
          val_frame0, val_frame_t, val_frame1 = val_frame0.to(device), val_frame_t.to(device), val_frame1.to(device)
          val_output = model(val_frame0, val_frame1)
          val_output = val_output.cpu().clamp(0, 1)

          output_image_path = os.path.join(output_dir, f"result_epoch_{epoch+1}.png")
          torchvision.utils.save_image(val_output[0], output_image_path)

        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"模型在第 {epoch + 1} 轮达到最佳验证损失，已保存模型到: {best_model_path}")

    print("训练完成！")
    print(f"最佳模型保存在: {best_model_path}")