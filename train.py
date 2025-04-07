import torch
from torch.utils.data import DataLoader
from src.models.ema_vfi import EMA_VFI
from src.utils.loss_functions import total_loss, VGGPerceptualLoss, color_histogram_loss, charbonnier_loss, gradient_loss, temporal_consistency_loss, warp # 导入所有损失函数
from src.utils.data_utils import VideoDataset  # 确保路径正确
import torch.optim as optim
import yaml
import os
from tqdm import tqdm  # 导入 tqdm
import cv2  # 使用cv2保存图像
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

if __name__ == '__main__':
    # 加载配置文件
    with open('config/train_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("config:", config)

    # 定义训练参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    num_epochs = config['num_epochs']
    train_data_dir = config['train_data_dir']
    val_data_dir = config['val_data_dir']
    checkpoint_path = config.get('checkpoint_path', 'checkpoints')  # 从配置文件中获取checkpoint路径，如果没有则使用默认值 checkpoints
    best_model_path = config.get('best_model_path', 'best_ema_vfi.pth')  # 用于存储最佳模型的文件名，并尝试从配置文件中读取，默认为根目录下的best_ema_vfi.pth
    charbonnier_weight = config.get('charbonnier_weight', 1.0)  # 获取 charbonnier 权重
    vgg_weight = config.get('vgg_weight', 0.05) # 获取 vgg 权重
    color_weight = config.get('color_weight', 0.1) # 获取色彩损失权重，默认为0.0, 调整为0.1
    gradient_weight = config.get('gradient_weight', 0.0) # 获取梯度损失权重
    temporal_weight = config.get('temporal_weight', 0.0)  # 获取时间一致性损失权重
    output_image_path = config.get('output_image_path', 'output_images') # 获取图像的输出路径
    color_jitter_params = config.get('color_jitter', None)  # 获取色彩抖动参数
    random_grayscale = config.get('random_grayscale', 0.1)  # 获取随机灰度转换概率

    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VideoDataset(data_dir=train_data_dir,
                                 transform=transform,
                                 color_jitter=color_jitter_params,
                                 crop_size=config.get('crop_size', (256, 256)),
                                 random_rotation=config.get('random_rotation', True),
                                 horizontal_flip=config.get('horizontal_flip', True),
                                 random_grayscale=random_grayscale)

    val_dataset = VideoDataset(data_dir=val_data_dir,
                              transform=transform,
                              color_jitter=color_jitter_params,
                              crop_size=config.get('crop_size', (256, 256)),
                              random_rotation=config.get('random_rotation', True),
                              horizontal_flip=config.get('horizontal_flip', True),
                              random_grayscale=random_grayscale)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # 初始化模型、优化器和损失函数
    model = EMA_VFI().to(device)

    # **你需要初始化你的光流估计网络并移动到正确的设备**
    # **你需要替换下面的`YourFlowEstimator()` 为你实际的光流估计模型**
    try:
        try:
            from src.models.sepconv_enhanced import YourFlowEstimator  # 确保路径正确
        except ImportError:
            raise ImportError("Module 'src.models.sepconv_enhanced' not found. Ensure the module exists and is in the Python path.")
        flow_estimator = YourFlowEstimator().to(device)  # 实例化你的光流估计器
    except ImportError:
        print("请确保你的光流估计器存在(如果没有请忽略这个错误)")
        flow_estimator = None

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4) # 替换为 ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True) #factor 每次降低的比例，patience可以容忍的轮数
    vgg_loss_fn = VGGPerceptualLoss(layer='relu2_2').to(device) # 修改 VGG loss 的参数
    print("模型初始化完成!")

    # 创建 checkpoint 目录 (如果不存在)
    os.makedirs(checkpoint_path, exist_ok=True)

    # 创建生成图像保存目录
    output_image_path = config.get('output_image_path', 'output_images')
    os.makedirs(output_image_path, exist_ok=True)

    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')  # 初始化最佳验证损失为正无穷大

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0  # 累加每个epoch的损失
        # 使用 tqdm 包裹 train_dataloader, 添加描述和动态更新
        with tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch") as tepoch:
            for i, (frame0, frame_t, frame1) in enumerate(tepoch):  # data loader 直接返回三个帧
                frame0 = frame0.to(device)
                frame_t = frame_t.to(device)
                frame1 = frame1.to(device)

                # 前向传播
                pred_frame_t = model(frame0, frame1)

                # 计算损失
                loss = total_loss(pred_frame_t, frame_t, vgg_loss_fn,
                                  charbonnier_weight=charbonnier_weight,  # 传递 charbonnier 权重
                                  vgg_weight=vgg_weight, # 传递 vgg 权重
                                  color_weight=color_weight, # 传递 color_weight
                                  gradient_weight=gradient_weight, # 传递梯度损失权重
                                  temporal_weight=temporal_weight, # 传递时间一致性权重
                                  flow_estimator=flow_estimator,  # 传递光流估计器
                                  frame0=frame0, # 传递 frame0
                                  frame1=frame1) # 传递 frame1

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # 调整梯度裁剪

                optimizer.step()

                running_loss += loss.item()  # 累加损失

                # 更新 tqdm 的描述信息，显示当前 loss
                tepoch.set_postfix(loss=loss.item())

                # 保存生成图像 (每个 epoch 只保存一次)
                if i == 0:
                    # 保存生成图像 (将数据缩放到 [0, 1] 范围)
                    img = pred_frame_t[0].cpu().detach()

                    # 反归一化
                    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
                    img = img * std[:, None, None] + mean[:, None, None]  # 反归一化
                    img = torch.clamp(img, 0, 1)  # 确保范围在 0 到 1 之间

                    # 转换成numpy数组，并调整维度顺序
                    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                    # 使用cv2保存图像
                    cv2.imwrite(os.path.join(output_image_path, f"epoch_{epoch+1}_generated.png"), img)


        # 验证
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for frame0, frame_t, frame1 in val_dataloader:
                frame0 = frame0.to(device)
                frame_t = frame_t.to(device)
                frame1 = frame1.to(device)

                pred_frame_t = model(frame0, frame1)
                loss = total_loss(pred_frame_t, frame_t, vgg_loss_fn,
                                  charbonnier_weight=charbonnier_weight,  # 传递 charbonnier 权重
                                  vgg_weight=vgg_weight, # 传递 vgg 权重
                                  color_weight=color_weight, # 传递 color_weight
                                  gradient_weight=gradient_weight, # 传递梯度损失权重
                                  temporal_weight=temporal_weight, # 传递时间一致性权重
                                  flow_estimator=flow_estimator, # 传递光流估计器
                                  frame0=frame0,  # 传递 frame0
                                  frame1=frame1)  # 传递 frame1
                val_loss += loss.item()

            val_loss /= len(val_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

            # 保存最佳模型
            best_val_loss = val_loss
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...")
            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型到指定路径

        scheduler.step(val_loss) # 如果使用 ReduceLROnPlateau
        # 在scheduler.step之前获取学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1}/{num_epochs}], Current Learning Rate: {current_lr:.6f}')

        # 保存每个 epoch 的模型
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'ema_vfi_epoch_{epoch + 1}.pth'))


    print("训练完成!")
    print(f"Best model saved to: {best_model_path}")