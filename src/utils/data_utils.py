import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None, frame_interval=1):
        self.data_dir = data_dir
        self.transform = transform
        self.frame_interval = frame_interval
        self.image_pairs = []

        # 遍历每个视频序列 (例如 "Beanbags")
        for video_dir in os.listdir(self.data_dir):
            video_path = os.path.join(self.data_dir, video_dir)
            if os.path.isdir(video_path):
                # 获取所有帧，并按名称排序
                frames = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

                #  生成 (frame0, frame_t, frame1) 对
                for i in range(len(frames) - 2 * self.frame_interval):
                    frame0_path = frames[i]
                    frame_t_path = frames[i + self.frame_interval]
                    frame1_path = frames[i + 2 * self.frame_interval]
                    self.image_pairs.append((frame0_path, frame_t_path, frame1_path))


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        frame0_path, frame_t_path, frame1_path = self.image_pairs[idx]

        frame0 = Image.open(frame0_path).convert('RGB')
        frame_t = Image.open(frame_t_path).convert('RGB')
        frame1 = Image.open(frame1_path).convert('RGB')

        if self.transform:
            frame0 = self.transform(frame0)
            frame_t = self.transform(frame_t)
            frame1 = self.transform(frame1)
        else:
             # 默认转换
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            frame0 = transform(frame0)
            frame_t = transform(frame_t)
            frame1 = transform(frame1)
        return frame0, frame_t, frame1

# 测试代码
if __name__ == '__main__':
    # 创建一个虚拟数据集目录结构（仅用于测试）
    os.makedirs('data/processed/train/test_video', exist_ok=True)
    for i in range(7):
        dummy_image = Image.new('RGB', (64, 48), color=(i*30, i*30, i*30)) # 创建一些虚拟图像
        dummy_image.save(f'data/processed/train/test_video/frame{i:02d}.png')

    dataset = VideoDataset(data_dir='data/processed/train', frame_interval=1) # data_dir指向'train'目录
    print(f"Dataset size: {len(dataset)}") # 应该输出Dataset size: 5
    frame0, frame_t, frame1 = dataset[0]
    print(f"Frame shape: {frame0.shape}") # 应该输出Frame shape: torch.Size([3, 48, 64])