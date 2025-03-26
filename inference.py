import cv2
import torch
import numpy as np
from src.models.ema_vfi import EMA_VFI
from torchvision import transforms
import argparse
import logging
import sys
import os
from tqdm import tqdm # 导入tqdm
from torch.cuda.amp import autocast

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # 将日志输出到控制台
    ]
)


# 定义参数解析器
parser = argparse.ArgumentParser(description="使用 EMA-VFI 模型进行视频插帧")
parser.add_argument("--input_video", type=str, default="input.mp4", help="输入视频路径")
parser.add_argument("--output_video", type=str, default="output.mp4", help="输出视频路径")
parser.add_argument("--model_path", type=str, default="best_ema_vfi.pth", help="模型权重文件路径")
parser.add_argument("--target_fps", type=float, default=30.0, help="目标帧率")
parser.add_argument("--interpolation_factor", type=int, default=1, help="每两帧之间插入的帧数")
parser.add_argument("--frame_interval", type=int, default=1, help="抽帧间隔")
parser.add_argument("--device", type=str, default="cuda", help="使用 cuda 或 cpu")
args = parser.parse_args()

# 图像预处理全局定义，避免重复创建
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_frame(frame, new_width, new_height, device, transform):
    """处理单帧图像，包括调整大小、转换为Tensor和移动到设备。"""
    frame = cv2.resize(frame, (new_width, new_height))
    frame_tensor = transform(frame).unsqueeze(0).to(device, non_blocking=True)
    return frame, frame_tensor

def denormalize_frame(frame_tensor):
    """将归一化的帧Tensor转换为可显示的图像"""
    frame = frame_tensor.squeeze(0).cpu().float().numpy()  # float() 确保精度
    frame = np.transpose(frame, (1, 2, 0))

    # 还原归一化
    frame = (frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame


def interpolate_video(input_video_path, output_video_path, model_path, target_fps, interpolation_factor, frame_interval, device):
    """对视频进行插帧，并保存到指定路径。"""

    logging.info(f"开始插帧：输入视频={input_video_path}, 输出视频={output_video_path}, 模型={model_path}, 目标帧率={target_fps}, 插帧因子={interpolation_factor}, 抽帧间隔={frame_interval}, 设备={device}")

    # 1. 加载模型
    try:
        model = EMA_VFI().to(device)
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        model.eval()
        logging.info(f"模型加载完成，使用设备：{device}")
    except FileNotFoundError:
        logging.error(f"模型文件未找到：{model_path}")
        return
    except Exception as e:
        logging.error(f"加载模型时发生错误：{e}")
        return

    # 2. 读取视频
    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logging.info(f"原始帧率: {fps}, 视频尺寸: {width}x{height}, 总帧数: {frame_count}")

        # 降低分辨率
        new_width = width // 2
        new_height = height // 2
        logging.info(f"调整后的视频尺寸：{new_width}x{new_height}")

    except Exception as e:
        logging.error(f"读取视频文件时发生错误：{e}")
        return

    # 目标帧率校验
    max_possible_fps = fps * (interpolation_factor + 1)
    if target_fps > max_possible_fps:
        logging.warning(f"目标帧率 {target_fps} 大于原始帧率插帧所能达到的最高值 {max_possible_fps}。可能无法达到目标帧率。")

    # 3. 视频编码器
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (new_width, new_height))
        logging.info(f"目标帧率：{target_fps}， 输出视频编码器：{fourcc}")
    except Exception as e:
        logging.error(f"创建视频编码器时发生错误：{e}")
        cap.release()
        return


    # 4. 插帧
    frame1 = None
    frame1_tensor = None
    frame_num = 0
    success = True # 用于保证第一次循环可以执行

    try:
        success, frame = cap.read() #read outside the loop

        if success:
            frame1, frame1_tensor = process_frame(frame, new_width, new_height, device, transform)
        else:
            logging.warning("视频为空，或者读取第一帧失败!")
            cap.release()
            out.release()
            return

        # 使用tqdm创建进度条
        with tqdm(total=frame_count, desc="插帧进度") as pbar:
            with torch.no_grad():  # 禁用梯度计算
                with autocast(): #启用混合精度
                    while success:
                        frame_num += 1

                        if frame_num % frame_interval == 0:
                            success, frame2 = cap.read()
                            if not success:
                                logging.info("视频读取结束!")
                                out.write(frame1)  # 写最后一帧
                                break  # 确保结束循环

                            frame2, frame2_tensor = process_frame(frame2, new_width, new_height, device, transform)

                            # 插入帧
                            for i in range(1, interpolation_factor + 1):
                                alpha = i / (interpolation_factor + 1)

                                # 使用模型预测中间帧
                                pred_frame_tensor = model(frame1_tensor, frame2_tensor)

                                # 将预测的帧转换为 numpy 数组
                                pred_frame = denormalize_frame(pred_frame_tensor)

                                # 写入视频
                                out.write(pred_frame)
                                del pred_frame_tensor #释放内存

                            # 写入原始帧
                            frame1_numpy = denormalize_frame(frame1_tensor)
                            out.write(frame1_numpy)

                            # 更新 frame1，并准备下一次迭代
                            frame1 = frame2
                            frame1_tensor = frame2_tensor

                        else:
                            success, frame2 = cap.read()
                            if success:
                                 frame2, frame2_tensor = process_frame(frame2, new_width, new_height, device, transform)
                                 frame1 = frame2
                                 frame1_tensor = frame2_tensor
                            else:
                                frame1_numpy = denormalize_frame(frame1_tensor)
                                out.write(frame1_numpy)
                                break

                        pbar.update(1) # 更新进度条

    except Exception as e:
        logging.error(f"处理帧时发生错误：{e}")
    finally:
        # 5. 释放资源
        cap.release()
        out.release()
        logging.info("视频插帧完成!")
        torch.cuda.empty_cache()  # 清理 CUDA 缓存


if __name__ == "__main__":

    # 在这里可以覆盖从命令行获取的参数。
    args.input_video = "input.mp4"
    args.output_video = "output.mp4"
    args.model_path = "2025.03.26.pth"
    args.target_fps = 180
    args.interpolation_factor = 2
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