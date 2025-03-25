import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None, frame_interval=1,
                  crop_size=(256, 256),
                 random_rotation=True,
                 horizontal_flip=True,
                 color_jitter=None,
                 random_grayscale=0.1): # 添加随机灰度转换
        self.data_dir = data_dir
        self.transform = transform
        self.frame_interval = frame_interval
        self.image_pairs = []
        self.crop_size = crop_size
        self.random_rotation = random_rotation
        self.horizontal_flip = horizontal_flip
        self.color_jitter = color_jitter
        self.random_grayscale = random_grayscale # 随机灰度转换的概率

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

        # 数据增强
        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(
                frame0, output_size=self.crop_size
            )
            frame0 = transforms.functional.crop(frame0, i, j, h, w)
            frame_t = transforms.functional.crop(frame_t, i, j, h, w)
            frame1 = transforms.functional.crop(frame1, i, j, h, w)

        if self.random_rotation:
            angle = transforms.RandomRotation.get_params([-180, 180]) # 随机角度
            frame0 = transforms.functional.rotate(frame0, angle)
            frame_t = transforms.functional.rotate(frame_t, angle)
            frame1 = transforms.functional.rotate(frame1, angle)

        if self.horizontal_flip:
            if random.random() > 0.5: # 使用 random.random()
                frame0 = transforms.functional.hflip(frame0)
                frame_t = transforms.functional.hflip(frame_t)
                frame1 = transforms.functional.hflip(frame1)

        # 应用色彩抖动
        if self.color_jitter:
            color_jitter = transforms.ColorJitter(**self.color_jitter)  # 使用解包运算符
            frame0 = color_jitter(frame0)
            frame_t = color_jitter(frame_t)
            frame1 = color_jitter(frame1)

        # 随机灰度转换
        if random.random() < self.random_grayscale:
            frame0 = transforms.functional.to_grayscale(frame0, num_output_channels=3) # 转换为3通道灰度图
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