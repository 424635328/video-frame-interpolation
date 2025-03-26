import cv2
import torch
import numpy as np
from src.models.ema_vfi import EMA_VFI  # 确保路径正确
from torchvision import transforms
import argparse
import logging
import sys
import os
from tqdm import tqdm
from torch.cuda.amp import autocast

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 定义参数解析器
parser = argparse.ArgumentParser(description="使用 EMA-VFI 模型进行视频插帧")
parser.add_argument("--input_video", type=str, default="input.mp4", help="输入视频路径")
parser.add_argument("--output_video", type=str, default="output.mp4", help="输出视频路径")
parser.add_argument("--model_path", type=str, default="2025.03.26.pth", help="模型权重文件路径")
parser.add_argument("--target_fps", type=float, default=None, help="目标帧率 (如果为None，则自动匹配插帧因子)")
parser.add_argument("--max_interpolation_factor", type=int, default=4, help="最大的插帧因子 (自动匹配时使用)")
parser.add_argument("--frame_interval", type=int, default=1, help="抽帧间隔")
parser.add_argument("--device", type=str, default="cuda", help="使用 cuda 或 cpu")
parser.add_argument("--codec", type=str, default="mp4v", help="输出视频编码器 (例如：mp4v, XVID)")
parser.add_argument("--bitrate", type=str, default="5M", help="输出视频比特率 (例如：5M, 10M)")
parser.add_argument("--scale", type=float, default=0.5, help="缩放比例, 0.5表示缩小一半")

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
    frame = frame_tensor.squeeze(0).cpu().float().numpy()
    frame = np.transpose(frame, (1, 2, 0))
    frame = (frame * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    return frame


def interpolate_video(input_video_path, output_video_path, model_path, target_fps, max_interpolation_factor, frame_interval, device, codec, bitrate, scale):
    """对视频进行插帧，并保存到指定路径。"""

    logging.info(f"开始插帧：输入视频={input_video_path}, 输出视频={output_video_path}, 模型={model_path}, 目标帧率={target_fps}, 最大插帧因子={max_interpolation_factor}, 抽帧间隔={frame_interval}, 设备={device}, 编码器={codec}, 比特率={bitrate}, 缩放={scale}")

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
        new_width = int(width * scale)
        new_height = int(height * scale)
        logging.info(f"调整后的视频尺寸：{new_width}x{new_height}")

    except Exception as e:
        logging.error(f"读取视频文件时发生错误：{e}")
        return

    # 3. 自动匹配插帧因子
    if target_fps is None:
        # Find the best interpolation factor to achieve a target FPS close to the desired FPS.
        best_interpolation_factor = 0
        min_fps_diff = float('inf')
        for interpolation_factor in range(1, max_interpolation_factor + 1):
            possible_fps = fps * (interpolation_factor + 1)
            fps_diff = abs(possible_fps - 60) # Assume desired fps is 60
            if fps_diff < min_fps_diff:
                min_fps_diff = fps_diff
                best_interpolation_factor = interpolation_factor
        target_fps = fps * (best_interpolation_factor + 1)
        interpolation_factor = best_interpolation_factor
        logging.info(f"自动匹配插帧因子：插帧因子 = {interpolation_factor}, 目标帧率 = {target_fps}")
    else:
        #如果指定了目标帧率，则计算所需的插帧因子。
        interpolation_factor = round(target_fps/fps -1)
        logging.info(f"手动指定目标帧率, 插帧因子 = {interpolation_factor}, 目标帧率 = {target_fps}")

    # 目标帧率校验
    max_possible_fps = fps * (interpolation_factor + 1)
    if target_fps > max_possible_fps:
        logging.warning(f"目标帧率 {target_fps} 大于原始帧率插帧所能达到的最高值 {max_possible_fps}。将使用最大可能帧率。")
        target_fps = max_possible_fps

    # 4. 视频编码器
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (new_width, new_height))
        logging.info(f"目标帧率：{target_fps}， 输出视频编码器：{codec}")

        # 尝试设置比特率 (这可能不适用于所有编码器)
        # out.set(cv2.VIDEOWRITER_PROP_BITRATE, int(bitrate[:-1]) * 100000)  # 例如，5M -> 5000000
    except Exception as e:
        logging.error(f"创建视频编码器时发生错误：{e}")
        cap.release()
        return

    # 5. 插帧
    frame1 = None
    frame1_tensor = None
    frame_num = 0
    success = True

    try:
        success, frame = cap.read()  # read outside the loop

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
                with autocast():  # 启用混合精度
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
                                del pred_frame_tensor  # 释放内存

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

                        pbar.update(1)  # 更新进度条

    except Exception as e:
        logging.error(f"处理帧时发生错误：{e}")
    finally:
        # 6. 释放资源
        cap.release()
        out.release()
        logging.info("视频插帧完成!")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    input_video = args.input_video
    output_video = args.output_video
    model_path = args.model_path
    target_fps = args.target_fps
    max_interpolation_factor = args.max_interpolation_factor
    frame_interval = args.frame_interval
    device = args.device
    codec = args.codec
    bitrate = args.bitrate
    scale = args.scale

    interpolate_video(input_video, output_video, model_path, target_fps, max_interpolation_factor, frame_interval, device, codec, bitrate, scale)