[English](README_en.md) | [简体中文](README.md)

# 基于 EMA-VFI 的视频帧插值

本项目实现了一个基于指数移动平均 (EMA) 和可变形卷积的视频帧插值 (VFI) 模型。 它的目标是生成两个给定帧之间的中间帧，从而提高帧率并改善视频的流畅度。

## 模型架构

该模型架构包含以下关键组件：

*   **特征提取:** 使用卷积块从输入帧中提取深度特征。 这是模型理解图像内容的第一步。
*   **上下文编码:** 编码全局上下文信息以理解场景。  这有助于模型理解整个场景，而不仅仅是局部像素。
*   **运动估计:** 估计输入帧之间的光流。  光流表示图像中每个像素的运动矢量，是插值的关键信息。
*   **扭曲 (Warping):** 使用估计的光流将一帧扭曲到另一帧的视角。  这使得可以将一帧的信息“移动”到另一帧的位置。
*   **多注意力融合:** 使用调制的可变形卷积融合原始特征和扭曲特征。  可变形卷积允许模型自适应地关注图像中的相关区域，并更好地融合信息。
*   **重建:** 从融合的特征中重建插值帧。  最后一步，将融合的特征转换回图像像素。

## 数据集

本项目使用（或设计为训练于）[Middlebury 光流数据集](https://vision.middlebury.edu/flow/data/) 进行训练和评估。 该数据集提供了一组校准的立体图像对，以及真实的光流场，这对于评估插值结果至关重要。

**注意:** 在训练之前，您需要下载并预处理 Middlebury 数据集。 有关更多详细信息，请参阅 [数据准备](#数据准备) 部分。 如果您计划使用自己的数据集，请确保对其进行适当的预处理和格式化，以使其与模型和训练脚本兼容。 确保数据集的图像质量和多样性，以获得更好的模型泛化能力。

## 依赖项

*   Python 3.7+ (推荐使用 3.8 或更高版本)
*   PyTorch 1.10+ (推荐使用 1.12 或更高版本，并启用 CUDA 支持)
*   Torchvision 0.11+ (需要 `DeformConv2d`，建议使用最新版本)
*   YAML (用于读取配置文件)
*   tqdm (用于显示训练进度条)
*   OpenCV (cv2) (用于视频读取和写入)
*   （`requirements.txt` 中列出的其他依赖项）

要安装依赖项，请运行：

```bash
pip install -r requirements.txt
```

**错误处理:**

*   **CUDA 错误:** 如果您遇到 CUDA 相关错误，请确保您的 PyTorch 安装支持 CUDA，并且您的 CUDA 驱动程序与 PyTorch 版本兼容。
*   **依赖项冲突:**  使用虚拟环境 (virtualenv 或 conda) 来隔离项目依赖项，避免与其他 Python 项目的冲突。
*   **版本不匹配:** 仔细检查 `requirements.txt` 中的依赖项版本，并确保它们与您的环境兼容。

## 数据准备

1.  **下载 Middlebury 数据集:**
    *   从 [https://vision.middlebury.edu/flow/data/](https://vision.middlebury.edu/flow/data/) 下载所需的立体图像对。
    *   选择适当的数据集版本（例如，“其他数据集”）。  根据您的需求选择数据集的大小和复杂性。

2.  **预处理数据集:**

    本项目要求数据以特定的目录结构组织：

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

    其中:

    *   `data/processed/train/` 包含训练场景。
    *   `data/processed/val/` 包含验证场景。
    *   每个场景目录包含一系列帧。
    *   文件名应为顺序的（例如，`frame0001.png`，`frame0002.png`，...）。  这对于按正确的顺序加载帧至关重要。

    您可以使用提供的 `src/utils/data_utils.py` 并对其进行修改，以根据所需的目录结构加载和预处理 Middlebury 数据集。 调整 `VideoDataset` 类以从 Middlebury 数据集加载图像对。 您可能需要编写自定义脚本以将 Middlebury 数据格式转换为 `VideoDataset` 期望的格式。

    **重要注意事项:**

    *   **数据增强:** 考虑使用数据增强技术（例如，随机裁剪、翻转、旋转、颜色抖动）以提高模型的泛化能力。 在您的数据加载管道中实现这些增强。  数据增强可以帮助模型更好地适应不同的场景和光照条件。
    *   **文件格式:** 确保图像文件采用与 OpenCV 兼容的格式（例如，PNG，JPG）。  不同的图像格式可能需要不同的解码器。
    *   **分辨率:** 模型可能具有特定的输入分辨率要求。 在预处理期间根据需要调整图像大小。  调整大小可能会影响图像质量和模型性能。 保持长宽比通常是重要的。
    *   **`.flo` 文件转换:** Middlebury 数据集包含 `.flo` 文件中的光流信息。 如果您的训练管道使用此信息，则需要将这些文件转换为合适的格式（例如，图像或 NumPy 数组）。 有现有的工具和库可用于读取 `.flo` 文件（在线搜索“python read .flo file”）。  确保使用正确的库来读取和解析 `.flo` 文件。
    *   **数据校验:**  编写脚本来校验预处理后的数据，例如检查图像是否已正确加载，以及光流数据是否有效。
    *   **内存管理:**  对于大型数据集，请考虑使用数据加载器的批处理和预取功能来有效地管理内存。

**错误处理:**

*   **文件不存在:**  确保数据集路径和文件名在配置文件中正确指定，并且文件实际存在。
*   **图像解码错误:**  使用 try-except 块来捕获图像解码错误，并记录错误的文件名。
*   **数据类型错误:**  验证数据类型是否与模型期望的类型匹配（例如，浮点数，整数）。

## 训练

1.  **配置训练:**

    修改 `config/train_config.yaml` 文件以调整训练参数(注意本项目对GPU内存要求较高，建议使用至少8GB显存的GPU)：

    ```yaml
    batch_size: 2
    learning_rate: 0.0003
    num_epochs: 20
    train_data_dir: 'data/processed/train'
    val_data_dir: 'data/processed/val'
    checkpoint_path: 'checkpoints'
    best_model_path: 'best_ema_vfi.pth'
    device: 'cuda'  # 或 'cpu'
    weight_decay: 0.0001 # L2 正则化
    clip_grad_norm: 1.0 # 梯度裁剪，防止梯度爆炸
    scheduler: "ReduceLROnPlateau" # 学习率调整策略，可以是 "StepLR", "MultiStepLR", "ReduceLROnPlateau"
    # StepLR 配置
    step_size: 10 # 每隔多少个 epoch 降低学习率
    gamma: 0.1 # 学习率降低的倍数
    # MultiStepLR 配置
    milestones: [10, 15] # 在哪些 epoch 降低学习率
    # ReduceLROnPlateau 配置
    patience: 3 # 验证损失多少个 epoch 没有改善后降低学习率
    factor: 0.1 # 学习率降低的倍数
    ```

    *   `batch_size`: 训练的批次大小。 增加批次大小通常可以提高训练速度，但需要更多的 GPU 内存。
    *   `learning_rate`: AdamW 优化器的学习率。  选择合适的学习率对训练至关重要。 可以尝试不同的学习率，并使用学习率调整策略。
    *   `num_epochs`: 训练 epoch 的数量。  训练更多的 epoch 通常可以提高模型性能，但也会增加训练时间。
    *   `train_data_dir`: 训练数据目录的路径。
    *   `val_data_dir`: 验证数据目录的路径。
    *   `checkpoint_path`: 存储模型检查点的路径。  定期保存检查点可以防止训练中断时丢失进度。
    *   `best_model_path`: 用于保存最佳模型的文件名。  基于验证集性能选择最佳模型。
    *   `device`: 指定用于训练的设备（'cuda' 或 'cpu'）。  如果您的机器没有 GPU，请使用 'cpu'。
    *   `weight_decay`: L2 正则化系数，防止过拟合。
    *   `clip_grad_norm`: 梯度裁剪的最大范数，防止梯度爆炸。
    *   `scheduler`: 学习率调整策略，可以是 "StepLR", "MultiStepLR", "ReduceLROnPlateau"。
        *   `StepLR`: 每隔 `step_size` 个 epoch 将学习率乘以 `gamma`。
        *   `MultiStepLR`: 在指定的 `milestones` 处将学习率乘以 `gamma`。
        *   `ReduceLROnPlateau`: 当验证损失在 `patience` 个 epoch 内没有改善时，将学习率乘以 `factor`。

2.  **开始训练:**

    运行 `train.py` 脚本:

    ```bash
    python train.py --config config/train_config.yaml
    ```

    训练进度，包括损失和验证损失，将显示在控制台中。 最佳模型将保存到指定的 `best_model_path`。 强烈建议使用 TensorBoard 集成来监控训练进度，例如损失曲线、学习率变化等。

**错误处理:**

*   **内存不足 (OOM) 错误:**  如果遇到 CUDA 内存不足错误，请减小 `batch_size`，减少模型大小，或使用梯度累积。
*   **梯度爆炸/消失:**  使用梯度裁剪 (`clip_grad_norm`) 和适当的权重初始化来缓解梯度问题。
*   **训练不收敛:**  调整学习率、批次大小、优化器参数或模型架构。 确保数据已正确预处理。
*   **过拟合:**  使用数据增强、权重衰减或 dropout 来减少过拟合。
*   **配置文件错误:**  使用 YAML 解析器验证配置文件的格式是否正确。

## 评估

目前，没有专门的评估脚本。 可以通过修改 `train.py` 脚本在训练后对单独的验证或测试数据集运行推理来进行评估。 这将涉及：

1.  加载训练好的模型。
2.  使用与训练数据相同的预处理步骤准备测试数据集。
3.  迭代测试数据集，将图像对馈送到模型。
4.  计算插值帧和真实帧之间的适当评估指标（例如，PSNR，SSIM）（如果可用）。 如果没有真实数据，则需要进行定性评估。
5.  报告评估结果。

为此目的实现一个单独的 `eval.py` 脚本是建议的未来改进。

**建议的评估指标:**

*   **PSNR (峰值信噪比):**  衡量重建图像的质量。  较高的 PSNR 值表示更好的质量。
*   **SSIM (结构相似性指数):**  衡量重建图像与真实图像的结构相似性。  SSIM 值范围为 -1 到 1，值越接近 1 表示相似度越高。
*   **LPIPS (感知损失):**  衡量重建图像与真实图像的感知差异。  LPIPS 值越低表示感知相似度越高。
*   **用户研究:**  如果可能，进行用户研究以评估插值视频的视觉质量。

## 推理

要使用训练好的模型执行视频帧插值，请运行 `inference.py` 脚本（或您的等效脚本）：

```bash
python inference.py --input_video input.mp4 --output_video output.mp4 --model_path best_ema_vfi.pth --target_fps 60 --interpolation_factor 1 --frame_interval 1 --device cuda
```

或者，使用脚本中定义的默认值：

```bash
python inference.py
```

命令行参数：

*   `--input_video`: 输入视频文件的路径。  支持常见的视频格式，例如 MP4, AVI 等。
*   `--output_video`: 输出视频文件的路径。  确保输出路径具有写入权限。
*   `--model_path`: 训练好的模型权重文件的路径。  确保模型权重文件存在且有效。
*   `--target_fps`: 输出视频的目标帧率。  增加目标帧率可以提高视频的流畅度。
*   `--interpolation_factor`: 在每对输入帧之间插入的帧数。 此参数直接控制输出帧率。  `interpolation_factor = n` 将在每对输入帧之间插入 `n` 帧。
*   `--frame_interval`: 处理来自输入视频的每 n 帧。 默认为 1（处理所有帧）。 大于 1 的值允许跳过帧以进行测试或性能原因。 例如，`frame_interval = 2` 将只处理视频中的每隔一帧。
*   `--device`: 用于推理的设备（'cuda' 或 'cpu'）。

**重要:** 确保已正确安装和配置 OpenCV (cv2) 以进行视频读取和写入。

**错误处理:**

*   **视频读取错误:** 检查输入视频文件是否存在且有效。  OpenCV 可能不支持某些视频编解码器。
*   **模型加载错误:** 确保模型文件存在且与当前模型架构兼容。
*   **视频写入错误:** 检查输出路径是否具有写入权限，并且 OpenCV 可以写入指定的视频格式。
*   **设备不可用:** 确保指定的设备（'cuda' 或 'cpu'）可用。

## 项目结构

```
.
├── config/
│   └── train_config.yaml         # 训练配置文件
├── src/
│   ├── models/
│   │   └── ema_vfi.py             # EMA-VFI 模型定义
│   ├── utils/
│   │   ├── data_utils.py           # 数据集和数据加载实用程序
│   │   ├── loss_functions.py      # 损失函数定义
│   │   └── ...
├── train.py                       # 训练脚本
├── inference.py                   # 推理脚本
├── README.md                      # 此文件
├── requirements.txt               # 项目依赖项
└── LICENSE                        # 许可证文件
```

## 致谢

*   本项目受到 [相关研究论文和开源实现](https://arxiv.org/abs/1904.00842) 的启发。 具体来说，EMA-VFI 架构建立在指数移动平均以实现时间一致性和可变形卷积以实现自适应特征融合的原理之上。
*   Middlebury 数据集是光流和视频帧插值研究的宝贵资源。

## 许可证

本项目已获得 [MIT 许可证](LICENSE) 的许可。

## 贡献

欢迎贡献！ 请随时提交包含错误修复、改进或新功能的拉取请求。  请遵循代码风格指南，并提供清晰的提交信息。

## 联系方式

MiracleHcat@gmail.com
