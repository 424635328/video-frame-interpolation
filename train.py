# train.py (修改后的训练脚本)
import torch
from torch.utils.data import DataLoader
from src.models.ema_vfi import EMA_VFI
from src.utils.loss_functions import total_loss, VGGPerceptualLoss
from src.utils.data_utils import VideoDataset  # 确保路径正确
import torch.optim as optim
import yaml
import os  # 导入 os 模块
from tqdm import tqdm  # 导入 tqdm

# 加载配置文件
with open('config/train_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("config:", config)

# 定义训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['num_epochs']
checkpoint_path = config.get('checkpoint_path', 'checkpoints')  # 从配置文件中获取checkpoint路径，如果没有则使用默认值 checkpoints
best_model_path = config.get('best_model_path', 'best_ema_vfi.pth')  # 用于存储最佳模型的文件名，并尝试从配置文件中读取，默认为根目录下的best_ema_vfi.pth

# 准备数据集
train_dataset = VideoDataset(data_dir=config['train_data_dir'],  # 修改为你的训练集目录
                             transform=None)  # 可以添加数据增强
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = VideoDataset(data_dir=config['val_data_dir'],  # 修改为你的验证集目录
                           transform=None)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

# 初始化模型、优化器和损失函数
model = EMA_VFI().to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
vgg_loss_fn = VGGPerceptualLoss().to(device)
print("模型初始化完成!")

# 创建 checkpoint 目录 (如果不存在)
os.makedirs(checkpoint_path, exist_ok=True)

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
            loss = total_loss(pred_frame_t, frame_t, vgg_loss_fn)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # 累加损失

            # 更新 tqdm 的描述信息，显示当前 loss
            tepoch.set_postfix(loss=loss.item())
            
    # 学习率更新
    scheduler.step()

    # 验证
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for frame0, frame_t, frame1 in val_dataloader:
            frame0 = frame0.to(device)
            frame_t = frame_t.to(device)
            frame1 = frame1.to(device)

            pred_frame_t = model(frame0, frame1)
            loss = total_loss(pred_frame_t, frame_t, vgg_loss_fn)
            val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...")
            torch.save(model.state_dict(), best_model_path)  # 保存最佳模型到指定路径

    # 保存每个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(checkpoint_path, f'ema_vfi_epoch_{epoch + 1}.pth'))


print("训练完成!")
print(f"Best model saved to: {best_model_path}")