
import cv2
import torch
import numpy as np
from src.models.ema_vfi import EMA_VFI
from torchvision import transforms
import argparse
from torch.cuda.amp import autocast

# 定义参数解析器
parser = argparse.ArgumentParser(description="使用 EMA-VFI 模型进行视频插帧")
parser.add_argument("--input_video", type=str, default="input.mp4", help="输入视频路径")
parser.add_argument("--output_video", type=str, default="output.mp4", help="输出视频路径")
parser.add_argument("--model_path", type=str, default="best_ema_vfi.pth", help="模型权重文件路径")
parser.add_argument("--target_fps", type=float, default=30.0, help="目标帧率") #降低默认帧率
parser.add_argument("--interpolation_factor", type=int, default=1, help="每两帧之间插入的帧数")
parser.add_argument("--frame_interval", type=int, default=1, help="抽帧间隔")
parser.add_argument("--device", type=str, default="cuda", help="使用 cuda 或 cpu")
args = parser.parse_args()

# 图像预处理全局定义，避免重复创建
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def interpolate_video(input_video_path, output_video_path, model_path, target_fps, interpolation_factor, frame_interval, device):
    """
    对视频进行插帧，并保存到指定路径。
    """
    # 1. 加载模型
    model = EMA_VFI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    print(f"使用设备：{device}")
    print("模型加载完成!")

    # 2. 读取视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{input_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 降低分辨率
    new_width = width // 2
    new_height = height // 2
    print(f"调整后的视频尺寸：{new_width}x{new_height}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"原始帧率: {fps}, 视频尺寸: {width}x{height}, 总帧数: {frame_count}")

    # 3. 视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (new_width, new_height)) # 使用新的分辨率
    print(f"目标帧率：{target_fps}， 输出视频编码器：{fourcc}")

    # 5. 插帧
    frame_num = 0
    success, frame1 = cap.read()

    if success:  # 调整第一帧的大小
        frame1 = cv2.resize(frame1, (new_width, new_height))

    with torch.no_grad():  # 禁用梯度计算
        with autocast(): #启用混合精度
            while success:
                frame_num += 1

                if frame_num % frame_interval == 0:
                    success, frame2 = cap.read()
                    if not success:
                        print("视频读取结束!")
                        out.write(frame1)
                        break

                    # 调整帧的大小
                    frame2 = cv2.resize(frame2, (new_width, new_height))

                    # 转换为 PyTorch Tensor，并移动到 GPU
                    frame1_tensor = transform(frame1).unsqueeze(0).to(device, non_blocking=True) #非阻塞传输
                    frame2_tensor = transform(frame2).unsqueeze(0).to(device, non_blocking=True) #非阻塞传输

                    # 插入帧
                    for i in range(1, interpolation_factor + 1):
                        alpha = i / (interpolation_factor + 1)

                        # 使用模型预测中间帧
                        pred_frame_tensor = model(frame1_tensor, frame2_tensor)

                        # 将预测的帧转换为 numpy 数组
                        pred_frame = pred_frame_tensor.squeeze(0).cpu().float().numpy()  # float() 确保精度
                        pred_frame = np.transpose(pred_frame, (1, 2, 0))

                        # 还原归一化
                        pred_frame = (pred_frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                        pred_frame = np.clip(pred_frame, 0, 1)
                        pred_frame = (pred_frame * 255).astype(np.uint8)

                        # 写入视频
                        out.write(pred_frame)
                        del pred_frame_tensor #释放内存

                    # 写入原始帧
                    out.write(frame1)
                    del frame1_tensor, frame2_tensor #释放内存

                    # 更新 frame1
                    frame1 = frame2
                else:
                    success, frame2 = cap.read()
                    if success:
                        frame2 = cv2.resize(frame2, (new_width, new_height))
                        frame1 = frame2
                    else:
                        out.write(frame1)
                        break

    # 6. 释放资源
    cap.release()
    out.release()
    print("视频插帧完成!")


if __name__ == "__main__":

    #  在这里可以覆盖从命令行获取的参数。
    args.input_video = "input.mp4"
    args.output_video = "output.mp4"
    args.model_path = "best_ema_vfi.pth"
    args.target_fps = 165
    args.interpolation_factor = 1
    args.frame_interval = 1
    args.device = "cuda"

    input_video = args.input_video
    output_video = args.output_video
    model_path = args.model_path
    target_fps = args.target_fps
    interpolation_factor = args.interpolation_factor
    frame_interval = args.frame_interval
    device = args.device

    interpolate_video(input_video, output_video, model_path, target_fps, interpolation_factor, frame_interval, device)

# 使用方法：
# python inference.py --input_video my_input.mp4 --output_video my_output.mp4 --model_path my_model.pth --target_fps 120

# 或者不提供参数，使用默认值：
# python inference.py