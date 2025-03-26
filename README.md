[English](README_en.md) | [简体中文](README.md)

# 基于 EMA-VFI 的视频帧插值

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/424635328/video-frame-interpolation?style=social)](https://github.com/424635328/video-frame-interpolation)

本项目实现了一个基于指数移动平均 (EMA) 和可变形卷积的视频帧插值 (VFI) 模型。它可以生成两帧之间的过渡帧，从而提高视频的流畅度。

**遇到显存不足？使用 AutoDL 云算力！**

训练和推理 VFI 模型通常需要大量的显存。 如果你的 GPU 显存不够，我们强烈推荐使用 AutoDL AI 算力云 (https://www.autodl.com/)。 它提供各种高性能 GPU 实例，能显著加速训练，避免显存溢出问题。

**如何在 AutoDL 上运行本项目：**

1.  **创建实例：** 在 AutoDL 上选择合适的 GPU 型号 (例如 RTX 4090, A100)，推荐选择预装 PyTorch 的镜像 (例如 `yolov5-master v1`)，节省环境配置时间。
2.  **上传代码：** 将项目代码克隆到 AutoDL 实例：`git clone https://github.com/424635328/video-frame-interpolation`。 如果有预处理的数据集，也一并上传。
3.  **安装依赖：** 运行 `pip install -r requirements.txt` 安装项目所需依赖。
4.  **配置训练：** 修改 `config/train_config.yaml`，根据你选择的 GPU 型号调整 `batch_size` 等参数。 显存越大，`batch_size` 可以设置得越大，从而提高训练效率。
5.  **运行脚本：** 运行 `python train.py` 或 `python inference.py` 开始训练或推理。
6.  **下载结果：** 训练完成后，将模型或推理结果下载到本地。

使用 AutoDL 能大幅简化 VFI 模型的训练流程，并获得更好的性能。

## 模型架构概览

EMA-VFI 模型主要包含以下几个核心模块：

*   **特征提取：** 从输入帧提取深层特征，为后续处理提供信息基础。
*   **上下文编码：** 学习全局场景信息，帮助模型理解整体画面。
*   **运动估计：** 估算帧与帧之间的光流，即像素的运动方向和速度，是插值的关键。
*   **扭曲 (Warping)：** 根据光流将一帧的像素“移动”到另一帧的位置，实现视角转换。
*   **多注意力融合：** 使用可变形卷积，自适应地融合原始特征和扭曲后的特征，突出重要信息。
*   **重建：** 将融合后的特征转换为最终的插值帧。

## 数据集准备

本项目默认使用 [Middlebury 光流数据集](https://vision.middlebury.edu/flow/data/) 进行训练和评估。 该数据集包含校准后的立体图像对和真实光流数据，非常适合评估插值效果。

**重要提示：**

*   **下载数据集：** 从 Middlebury 网站下载所需的数据。
*   **预处理：** 将数据集整理成如下目录结构：

    ```
    data/
    └── processed/
        ├── train/
        │   ├── scene1/
        │   │   ├── frame0001.png
        │   │   ├── frame0002.png
        │   │   ├── frame0003.png
        │   │   └── ...
        │   ├── scene2/
        │   │   ├── ...
        │   └── ...
        └── val/
            ├── scene1/
            │   ├── ...
            └── ...
    ```

    *   `data/processed/train/`：训练集场景。
    *   `data/processed/val/`：验证集场景。
    *   每个场景目录下包含一系列按顺序排列的帧，例如 `frame0001.png`, `frame0002.png`。

*   **预处理脚本：** 你可以修改 `src/utils/data_utils.py` 中的 `VideoDataset` 类，使其能够加载 Middlebury 数据集。 你可能需要编写自定义脚本来转换 Middlebury 数据格式。

**数据集处理注意事项：**

*   **文件命名：** 确保图像文件名连续且格式一致，例如 `frame_00001.png, frame_00002.png,...`。
*   **数据增强 (关键！)：** 由于 Middlebury 数据集较小，务必使用数据增强来提高模型的泛化能力，避免过拟合。 常用增强方法包括：
    *   **随机裁剪：** `crop_size` 参数。
    *   **随机旋转：** `random_rotation` 参数。
    *   **水平翻转：** `horizontal_flip` 参数。
    *   **颜色抖动：** `color_jitter` 参数 (调整亮度、对比度、饱和度、色调)。
    *   **随机灰度转换：** `random_grayscale` 参数。
*   **文件格式：** 使用 OpenCV 兼容的图像格式 (例如 PNG, JPG)。 推荐使用无损压缩的 PNG 格式。
*   **分辨率：** 根据模型要求调整图像大小，保持长宽比通常很重要。
*   **`.flo` 文件：** Middlebury 数据集包含光流信息 `.flo` 文件，如果需要使用光流信息，你需要将它们转换为合适的格式 (例如图像或 NumPy 数组)。
*   **数据校验：** 编写脚本检查预处理后的数据，确保图像加载正确、尺寸一致、文件数量正确。

**数据准备常见问题：**

*   **文件不存在：** 检查数据集路径和文件名是否正确。
*   **图像解码错误：** 使用 `try-except` 捕获图像解码错误，并记录错误文件名。
*   **数据类型错误：** 验证数据类型是否与模型期望的类型匹配。

## 依赖项安装

*   Python 3.7+ (推荐 3.8 或更高版本)
*   PyTorch 1.10+ (推荐 1.12+，并启用 CUDA 支持)
*   Torchvision 0.11+ (需要 `DeformConv2d`，建议最新版本)
*   YAML
*   tqdm
*   OpenCV (cv2)
*   (以及 `requirements.txt` 中列出的其他依赖项)

运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

**依赖安装常见问题：**

*   **CUDA 错误：** 确保 PyTorch 支持 CUDA，并且 CUDA 驱动版本与 PyTorch 兼容。
*   **依赖冲突：** 使用虚拟环境 (virtualenv 或 conda) 隔离项目依赖。 推荐使用 Anaconda 创建虚拟环境：`conda create -n vfi python=3.8`。
*   **版本不匹配：** 仔细检查 `requirements.txt` 中的依赖版本，尤其是 `torch` 和 `torchvision` 的版本要对应。

## 模型训练

1.  **配置训练参数：**

    修改 `config/train_config.yaml` 文件以调整训练参数。 **以下参数已针对小数据集进行了优化，降低了学习率以提高训练的稳定性。**

    ```yaml
    batch_size: 2       # 批处理大小，小的 batch_size 有助于缓解小数据集上的过拟合
    learning_rate: 0.00005 # 学习率，降低学习率有助于稳定训练
    num_epochs: 100       # 训练轮数， 增加训练轮数以获得更好的性能
    train_data_dir: "data/processed/train" # 训练数据路径
    val_data_dir: "data/processed/val"   # 验证数据路径

    checkpoint_path: "checkpoints"    # 模型保存路径
    best_model_path: "best_ema_vfi.pth" # 最优模型路径

    # 损失函数权重， 可以根据训练情况进行调整
    charbonnier_weight: 0.6
    vgg_weight: 0.05
    color_weight: 0.25
    gradient_weight: 0.05
    temporal_weight: 0.05

    gradient_order: 1  # 梯度损失阶数
    temporal_alpha: 1.0 # 时间一致性loss权重

    output_image_path: "output_images"

    # 数据增强参数
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1

    crop_size: [256, 256]
    random_rotation: True
    horizontal_flip: True
    random_grayscale: 0.2
    ```

    *   `batch_size`:  较小的批次大小可以减少 GPU 内存的使用，并且可能有助于提高小数据集上的泛化能力。
    *   `learning_rate`: **对于小数据集，通常需要使用更小的学习率。** 5e-5 是一个合理的起点， 你也可以尝试更小的值，例如 1e-5。
    *   `num_epochs`: **增加训练轮数可以使模型更充分地学习数据集的特征。** 100 轮是一个合理的起点， 但你可以根据验证集的性能进行调整。
    *   `train_data_dir`: 训练数据目录的路径。
    *   `val_data_dir`: 验证数据目录的路径。
    *   `checkpoint_path`: 存储模型检查点的路径。
    *   `best_model_path`: 用于保存最佳模型的文件名。
    *   `charbonnier_weight`, `vgg_weight`, `color_weight`, `gradient_weight`, `temporal_weight`:  **建议谨慎调整这些权重，避免过度依赖某些损失函数导致模型不稳定。**
    *   `color_jitter`, `crop_size`, `random_rotation`, `horizontal_flip`, `random_grayscale`: **合理设置这些参数可以有效提高模型的泛化能力。**

    **小数据集训练技巧：**

    *   **降低学习率：** 使用较小的学习率，例如 5e-5 或 1e-5 (已完成)。
    *   **增加训练轮数：** 训练更多的 epoch，例如 100 轮或更多 (已完成)。
    *   **使用更强的数据增强：** 参考数据准备部分。
    *   **使用权重衰减 (L2 正则化)：** 尝试增加 `weight_decay`，例如 1e-4 或 1e-3。
    *   **使用 Dropout：** 在模型中添加 Dropout 层，以随机丢弃一些神经元。
    *   **使用 Early Stopping：** 监控验证集性能，并在验证损失停止下降时提前停止训练。
    *   **梯度裁剪：** 使用梯度裁剪可以防止梯度爆炸。

2.  **启动训练：**

    运行 `train.py` 脚本:

    ```bash
    python train.py
    ```

    训练进度将显示在控制台中。 最佳模型将保存到 `best_model_path`。 推荐使用 TensorBoard 监控训练进度。

**训练常见问题：**

*   **CUDA 内存不足 (OOM) 错误：**
    *   减小 `batch_size`。
    *   使用更小的图像尺寸。
    *   尝试使用 `torch.utils.checkpoint`。
*   **梯度爆炸/消失：**
    *   使用梯度裁剪 (`clip_grad_norm`)。
    *   使用适当的权重初始化方法。
    *   降低学习率。
*   **训练不收敛：**
    *   检查学习率是否合适。
    *   检查数据预处理步骤是否正确。
    *   检查损失函数是否正确。
*   **过拟合：**
    *   使用更强的数据增强。
    *   增加权重衰减。
    *   使用 Dropout。
    *   使用 Early Stopping。

## 模型评估

目前没有专门的评估脚本。 你可以通过修改 `train.py` 脚本，在训练后对验证集或测试集运行推理进行评估。

1.  加载训练好的模型。
2.  准备测试数据集 (使用与训练数据相同的预处理步骤)。
3.  迭代测试数据集，将图像对输入模型。
4.  计算插值帧和真实帧之间的评估指标 (例如 PSNR, SSIM, LPIPS)。 如果没有真实数据，则需要进行定性评估。
5.  报告评估结果。

未来可以考虑添加一个独立的 `eval.py` 脚本。

**评估指标：**

*   **PSNR (峰值信噪比)：** 衡量重建图像的质量，值越高越好。
*   **SSIM (结构相似性指数)：** 衡量重建图像与真实图像的结构相似性，值越接近 1 越好。
*   **LPIPS (感知损失)：** 衡量重建图像与真实图像的感知差异，值越低越好。
*   **用户研究：** 评估插值视频的视觉质量。

## 模型推理

使用 `inference.py` 脚本对视频进行帧插值：

```bash
python inference.py --input_video input.mp4 --output_video output.mp4 --model_path best_ema_vfi.pth --target_fps 60 --interpolation_factor 1 --frame_interval 1 --device cuda
```

或者，使用脚本中定义的默认值：

```bash
python inference.py
```

**参数说明：**

*   `--input_video`: 输入视频文件的路径 (例如 MP4, AVI)。
*   `--output_video`: 输出视频文件的路径。
*   `--model_path`: 训练好的模型权重文件的路径。
*   `--target_fps`: 输出视频的目标帧率。
*   `--interpolation_factor`: 在每对输入帧之间插入的帧数。 `interpolation_factor = n` 将在每对输入帧之间插入 `n` 帧。
*   `--frame_interval`: 处理来自输入视频的每 n 帧。 默认为 1 (处理所有帧)。
*   `--device`:  用于推理的设备 ('cuda' 或 'cpu')。

**确保已正确安装和配置 OpenCV (cv2) 以进行视频读取和写入。**

**推理常见问题：**

*   **视频读取错误：**
    *   检查输入视频文件是否存在且有效。
    *   确保 OpenCV 支持你使用的视频格式。
    *   尝试安装或更新相关的视频编解码器。
*   **模型加载错误：**
    *   确保模型文件存在且与当前模型架构兼容。
    *   检查模型文件是否损坏。
    *   模型文件可能与当前 PyTorch 版本不兼容。
*   **视频写入错误：**
    *   检查输出路径是否具有写入权限。
    *   确保 OpenCV 可以写入指定的视频格式。
    *   检查磁盘空间是否足够。
*   **设备不可用：**
    *   确保指定的设备可用。
    *   如果使用 CUDA，请确保已正确安装 CUDA 驱动程序。

## 项目结构

```
.
├── config/
│   └── train_config.yaml         # 训练配置文件
├── src/
│   ├── models/
│   │   └── ema_vfi.py             # EMA-VFI 模型定义
│   ├── utils/
│   │   └── data_utils.py           # 数据集和数据加载实用程序
│   │   └── loss_functions.py      # 损失函数定义
│   │   └── ...
├── train.py                       # 训练脚本
├── inference.py                   # 推理脚本
├── README.md                      # 此文件
├── requirements.txt               # 项目依赖项
└── LICENSE                        # 许可证文件
```

## 致谢

*   本项目受到 [相关研究论文和开源实现](https://arxiv.org/abs/1904.00842) 的启发。
*   感谢 Middlebury 数据集对光流和视频帧插值研究的贡献。

## 许可证

本项目使用 [MIT 许可证](LICENSE)。

## 贡献

欢迎贡献代码！ 请提交包含错误修复、改进或新功能的 Pull Request。

## 联系方式

MiracleHcat@gmail.com
