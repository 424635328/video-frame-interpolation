import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn

def l1_loss(pred, target):
    """
    计算 L1 损失（平均绝对误差）。
    """
    return torch.mean(torch.abs(pred - target))

def l2_loss(pred, target):
    """
    计算 L2 损失（均方误差）。
    """
    return torch.mean((pred - target)**2)

def charbonnier_loss(pred, target, epsilon=1e-3):
    """
    计算 Charbonnier 损失，一种平滑的 L1 损失。
    """
    return torch.mean(torch.sqrt((pred - target)**2 + epsilon**2))

class VGGPerceptualLoss(torch.nn.Module):
    """
    使用预训练的 VGG16 网络计算感知损失。
    """
    def __init__(self, resize=True, normalize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize
        self.normalize = normalize

    def forward(self, input, target):
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        if self.normalize:
            input = (input-self.mean) / self.std
            target = (target-self.mean) / self.std
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

# 梯度损失
def gradient_loss(pred, target):
    """
    计算梯度损失，鼓励生成图像具有更清晰的边缘。
    """
    def sobel(x):
        """
        使用 Sobel 算子计算图像梯度。
        """
        sobel_filter_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).to(x.device).view(1, 1, 3, 3)
        sobel_filter_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(x.device).view(1, 1, 3, 3)

        # 初始化存储梯度的张量
        Gx = torch.zeros_like(x)
        Gy = torch.zeros_like(x)

        # 遍历每个通道
        for c in range(x.shape[1]):
            # 对每个通道应用 Sobel 算子
            Gx[:, c:c+1] = F.conv2d(x[:, c:c+1], sobel_filter_x, padding=1)
            Gy[:, c:c+1] = F.conv2d(x[:, c:c+1], sobel_filter_y, padding=1)

        return Gx, Gy

    pred_Gx, pred_Gy = sobel(pred)
    target_Gx, target_Gy = sobel(target)
    return torch.mean(torch.abs(pred_Gx - target_Gx) + torch.abs(pred_Gy - target_Gy))

def color_histogram_loss(pred, target, num_bins=256):
    """
    计算色彩直方图损失。

    Args:
        pred (torch.Tensor): 预测图像，形状为 (B, C, H, W)，其中 C=3 (RGB)。
        target (torch.Tensor): 目标图像，形状为 (B, C, H, W)，其中 C=3 (RGB)。
        num_bins (int): 直方图的 bin 数量。

    Returns:
        torch.Tensor: 色彩直方图损失。
    """
    batch_size, channels, height, width = pred.size()

    # 初始化损失
    loss = torch.tensor(0.0, device=pred.device)

    # 遍历每个图像
    for i in range(batch_size):
        # 初始化每个通道的直方图
        pred_hist = torch.zeros((channels, num_bins), device=pred.device)
        target_hist = torch.zeros((channels, num_bins), device=pred.device)

        # 遍历每个通道（R、G、B）
        for c in range(channels):
            # 计算直方图
            pred_channel = pred[i, c, :, :]
            target_channel = target[i, c, :, :]

            # 将像素值缩放到 [0, num_bins-1] 范围内，并确保是非负整数
            pred_inds = torch.floor(pred_channel * (num_bins - 1))
            pred_inds = torch.clamp(pred_inds, min=0, max=num_bins - 1).long()

            target_inds = torch.floor(target_channel * (num_bins - 1))
            target_inds = torch.clamp(target_inds, min=0, max=num_bins - 1).long()

            # 统计每个 bin 的像素数量
            pred_hist[c] = torch.bincount(pred_inds.view(-1), minlength=num_bins).float()
            target_hist[c] = torch.bincount(target_inds.view(-1), minlength=num_bins).float()

            # 归一化直方图
            pred_hist[c] /= (height * width)
            target_hist[c] /= (height * width)

            # 计算 L1 距离
            loss += torch.sum(torch.abs(pred_hist[c] - target_hist[c]))

    # 平均每个图像的损失
    loss /= batch_size
    return loss

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
    output = F.grid_sample(x, vgrid, align_corners=True)
    return output

def temporal_consistency_loss(pred, frame0, frame1, flow_estimator):
    """
    计算时间一致性损失。

    Args:
        pred (torch.Tensor): 预测的中间帧，形状为 (B, C, H, W)。
        frame0 (torch.Tensor): 第一帧，形状为 (B, C, H, W)。
        frame1 (torch.Tensor): 第二帧，形状为 (B, C, H, W)。
        flow_estimator: 光流估计网络。

    Returns:
        torch.Tensor: 时间一致性损失。
    """
    # 估计从 frame0 到 pred 的光流
    flow01 = flow_estimator(frame0, pred)

    # 估计从 pred 到 frame1 的光流
    flow12 = flow_estimator(pred, frame1)

    # 扭曲 frame0 和 frame1
    frame0_warped = warp(frame0, flow01)
    frame1_warped = warp(frame1, flow12)

    # 计算像素一致性损失
    loss = torch.mean(torch.abs(frame0_warped - pred)) + torch.mean(torch.abs(frame1_warped - pred))
    return loss

def total_loss(pred, target, vgg_loss_fn, charbonnier_weight=1.0, vgg_weight=0.05, color_weight=0.0, gradient_weight=0.0, temporal_weight=0.0, flow_estimator=None, frame0=None, frame1=None):
    """
    计算总损失，包括 Charbonnier 损失、VGG 感知损失、色彩直方图损失和梯度损失。
    """
    charbonnier = charbonnier_loss(pred, target)  # 计算 charbonnier 损失
    vgg = vgg_loss_fn(pred, target)
    color_loss = color_histogram_loss(pred, target)  # 计算色彩损失
    gradient = gradient_loss(pred, target) #计算梯度损失

    temporal_loss = 0 # 初始化时间一致性损失
    if temporal_weight > 0 and flow_estimator is not None and frame0 is not None and frame1 is not None:
        temporal_loss = temporal_consistency_loss(pred, frame0, frame1, flow_estimator)

    return charbonnier_weight * charbonnier + vgg_weight * vgg + color_weight * color_loss + gradient_weight * gradient + temporal_weight * temporal_loss  # 添加梯度损失和时间一致性损失