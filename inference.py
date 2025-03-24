import cv2
import torch
import numpy as np
from src.models.ema_vfi import EMA_VFI  # 确保路径正确
from torchvision import transforms
import argparse # 导入 argparse

# 定义参数解析器
parser = argparse.ArgumentParser(description="使用 EMA-VFI 模型进行视频插帧")
parser.add_argument("--input_video", type=str, default="input.mp4", help="输入视频路径")
parser.add_argument("--output_video", type=str, default="output.mp4", help="输出视频路径")
parser.add_argument("--model_path", type=str, default="best_ema_vfi.pth", help="模型权重文件路径")
parser.add_argument("--target_fps", type=float, default=60.0, help="目标帧率")
parser.add_argument("--interpolation_factor", type=int, default=1, help="每两帧之间插入的帧数（可以不用，如果目标帧率控制插帧数量）")
parser.add_argument("--frame_interval", type=int, default=1, help="抽帧")
parser.add_argument("--device", type=str, default="cuda", help="使用cuda或者cpu")
args = parser.parse_args()

# 定义插帧函数
def interpolate_video(input_video_path, output_video_path, model_path, target_fps, interpolation_factor, frame_interval, device):
    """
    对视频进行插帧，并保存到指定路径。

    Args:
        input_video_path (str): 输入视频路径。
        output_video_path (str): 输出视频路径。
        model_path (str): 训练好的模型权重文件路径。
        target_fps (float): 目标帧率。
        interpolation_factor (int): 每两帧之间插入的帧数，插帧数量 = interpolation_factor。
        frame_interval (int): 每隔多少帧抽一帧
        device (str): 使用 cuda 或者 cpu
    """
    # 1. 加载模型
    model = EMA_VFI().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device))) # 指定设备加载
    model.eval()
    print(f"使用设备：{device}")
    print("模型加载完成!")

    # 2. 读取视频
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"原始帧率: {fps}, 视频尺寸: {width}x{height}, 总帧数: {frame_count}")

    # 3. 视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或 'XVID'，取决于你的系统
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (width, height))
    print(f"目标帧率：{target_fps}， 输出视频编码器：{fourcc}")

    # 4. 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(), # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化
    ])

    # 5. 插帧
    frame_num = 0
    success, frame1 = cap.read()  # 读取第一帧

    while success:
        frame_num += 1

        if frame_num % frame_interval == 0:
            success, frame2 = cap.read() # 读取下一帧
            if not success:
                print("视频读取结束!")
                out.write(frame1)  # 写入最后一帧
                break

            # 转换为 PyTorch Tensor，并移动到 GPU
            frame1_tensor = transform(frame1).unsqueeze(0).to(device)
            frame2_tensor = transform(frame2).unsqueeze(0).to(device)

            # 插入帧
            with torch.no_grad():
                for i in range(1, interpolation_factor + 1):
                    alpha = i / (interpolation_factor + 1)  # 计算插值权重

                    # 使用模型预测中间帧
                    pred_frame_tensor = model(frame1_tensor, frame2_tensor)

                    # 将预测的帧转换为 numpy 数组
                    pred_frame = pred_frame_tensor.squeeze(0).cpu().numpy()
                    pred_frame = np.transpose(pred_frame, (1, 2, 0)) # CHW -> HWC

                    # 还原归一化
                    pred_frame = (pred_frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                    pred_frame = np.clip(pred_frame, 0, 1) # 裁剪到 [0, 255]
                    pred_frame = (pred_frame * 255).astype(np.uint8)

                    # 写入视频
                    out.write(pred_frame)

            # 写入原始帧
            out.write(frame1)

            # 更新 frame1
            frame1 = frame2
        else:
            success, frame2 = cap.read()
            if success:
                frame1 = frame2
            else:
                out.write(frame1)
                break

    # 6. 释放资源
    cap.release()
    out.release()
    print("视频插帧完成!")

#  主函数，用于调用插帧函数
if __name__ == "__main__":

    #  在这里可以覆盖从命令行获取的参数。

    # args.input_video = "input.mp4"
    # args.output_video = "output.mp4"
    # args.model_path = "best_ema_vfi.pth"
    # args.target_fps = 120
    # args.interpolation_factor = 1
    # args.frame_interval = 1
    # args.device = "cuda"

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