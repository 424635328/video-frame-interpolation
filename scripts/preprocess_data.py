import os
from PIL import Image

def resize_images(data_dir, target_size=(640, 480)):
    """调整指定目录下所有图像的尺寸"""
    for video_dir in os.listdir(data_dir):
        video_path = os.path.join(data_dir, video_dir)
        if os.path.isdir(video_path):
            for filename in os.listdir(video_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(video_path, filename)
                    try:
                        img = Image.open(image_path)
                        img = img.resize(target_size, Image.Resampling.LANCZOS)
                        img.save(image_path)  # 覆盖原始文件
                        print(f"调整图像尺寸: {image_path} -> {target_size}")
                    except Exception as e:
                        print(f"调整图像尺寸失败: {image_path}: {e}")

# 使用示例
data_dirs = ["data/processed/train", "data/processed/val"]
for data_dir in data_dirs:
    resize_images(data_dir) # 调整所有图像尺寸