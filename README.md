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

*   **CUDA 错误:** 如果您遇到 CUDA 相关错误，请确保您的 PyTorch 安装支持 CUDA，并且您的 CUDA 驱动程序与 PyTorch 版本兼容。 常见的 CUDA 相关错误包括：`CUDA out of memory`、`CUDNN_STATUS_INTERNAL_ERROR` 等。 请检查 CUDA 版本是否与 PyTorch 版本匹配，并确保 CUDA 驱动程序已正确安装。
*   **依赖项冲突:**  使用虚拟环境 (virtualenv 或 conda) 来隔离项目依赖项，避免与其他 Python 项目的冲突。 建议使用 Anaconda 创建虚拟环境：`conda create -n vfi python=3.8`。
*   **版本不匹配:** 仔细检查 `requirements.txt` 中的依赖项版本，并确保它们与您的环境兼容。 尤其是 `torch` 和 `torchvision` 的版本要对应。

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

    *   **文件命名规范:** 务必确保图像文件名是连续的且格式一致，例如 `frame_00001.png, frame_00002.png,...`。  可以使用脚本批量重命名文件。
    *   **数据增强 (至关重要):** 由于数据集较小，强烈建议使用数据增强技术来提高模型的泛化能力，避免过拟合。 尝试以下增强方法，并根据实际效果调整参数：
        *   **随机裁剪:**  从图像中随机裁剪一块区域 (`crop_size` 参数)。
        *   **随机旋转:**  随机旋转图像 (`random_rotation` 参数)。
        *   **水平翻转:**  随机水平翻转图像 (`horizontal_flip` 参数)。
        *   **颜色抖动:**  随机调整图像的亮度、对比度、饱和度和色调 (`color_jitter` 参数)。  请注意，过度的颜色抖动可能会导致颜色失真。
        *   **随机灰度转换:**  将图像转换为灰度图像 (`random_grayscale` 参数)。
    *   **文件格式:** 确保图像文件采用与 OpenCV 兼容的格式（例如，PNG，JPG）。  不同的图像格式可能需要不同的解码器。 推荐使用 PNG 格式，因为它是无损压缩。
    *   **分辨率:** 模型可能具有特定的输入分辨率要求。 在预处理期间根据需要调整图像大小。  调整大小可能会影响图像质量和模型性能。 保持长宽比通常是重要的， 可以考虑使用 `cv2.resize` 函数进行图像缩放。
    *   **`.flo` 文件转换:** Middlebury 数据集包含 `.flo` 文件中的光流信息。 如果您的训练管道使用此信息，则需要将这些文件转换为合适的格式（例如，图像或 NumPy 数组）。 有现有的工具和库可用于读取 `.flo` 文件（在线搜索“python read .flo file”）。  确保使用正确的库来读取和解析 `.flo` 文件。
    *   **数据校验:**  编写脚本来校验预处理后的数据，例如检查图像是否已正确加载，图像尺寸是否一致，以及文件数量是否正确。 可以使用 `PIL.Image.verify()` 检查图像是否损坏。

**错误处理:**

*   **文件不存在:**  确保数据集路径和文件名在配置文件中正确指定，并且文件实际存在。
*   **图像解码错误:**  使用 try-except 块来捕获图像解码错误，并记录错误的文件名。
*   **数据类型错误:**  验证数据类型是否与模型期望的类型匹配（例如，浮点数，整数）。

## 训练

1.  **配置训练:**

    修改 `config/train_config.yaml` 文件以调整训练参数。 **以下参数针对小数据集训练进行了优化，并降低了学习率以提高训练的稳定性。**

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

    *   `batch_size`: 训练的批次大小。  较小的批次大小可以减少 GPU 内存的使用，并且可能有助于提高小数据集上的泛化能力。
    *   `learning_rate`: AdamW 优化器的学习率。  **对于小数据集，通常需要使用更小的学习率。** 5e-5 是一个合理的起点， 您也可以尝试更小的值，例如 1e-5。
    *   `num_epochs`: 训练 epoch 的数量。  **增加训练轮数可以使模型更充分地学习数据集的特征。** 100 轮是一个合理的起点， 但您可以根据验证集的性能进行调整。
    *   `train_data_dir`: 训练数据目录的路径。
    *   `val_data_dir`: 验证数据目录的路径。
    *   `checkpoint_path`: 存储模型检查点的路径。  定期保存检查点可以防止训练中断时丢失进度。
    *   `best_model_path`: 用于保存最佳模型的文件名。  基于验证集性能选择最佳模型。
    *   `device`: 指定用于训练的设备（'cuda' 或 'cpu'）。  如果您的机器没有 GPU，请使用 'cpu'。
    *   `charbonnier_weight`, `vgg_weight`, `color_weight`, `gradient_weight`, `temporal_weight`: 这些是不同损失函数的权重。 由于是小数据集， **建议谨慎调整这些权重，避免过度依赖某些损失函数导致模型不稳定。**
    *   `color_jitter`, `crop_size`, `random_rotation`, `horizontal_flip`, `random_grayscale`: 这些是数据增强相关的参数。 **合理设置这些参数可以有效提高模型的泛化能力。**

    **重要:**  针对小数据集的训练，重点在于防止过拟合和稳定训练过程。  可以尝试以下策略：

    *   **降低学习率 (已完成):**  使用较小的学习率，例如 5e-5 或 1e-5。
    *   **增加训练轮数 (已完成):**  训练更多的 epoch，例如 100 轮或更多。
    *   **使用更强的数据增强 (参考数据准备部分):**  合理设置数据增强参数。
    *   **使用权重衰减 (L2 正则化):**  尝试增加 `weight_decay`，例如 1e-4 或 1e-3。
    *   **使用 Dropout:**  在模型中添加 Dropout 层，以随机丢弃一些神经元，防止模型过度依赖某些特征。
    *   **使用 Early Stopping:**  监控验证集的性能，并在验证损失停止下降时提前停止训练。  这可以避免模型在训练集上过拟合。
    *   **梯度裁剪:**  使用梯度裁剪可以防止梯度爆炸，使训练更加稳定。

2.  **开始训练:**

    运行 `train.py` 脚本:

    ```bash
    python train.py
    ```

    训练进度，包括损失和验证损失，将显示在控制台中。 最佳模型将保存到指定的 `best_model_path`。 强烈建议使用 TensorBoard 集成来监控训练进度，例如损失曲线、学习率变化等。

**错误处理:**

*   **CUDA 内存不足 (OOM) 错误:**
    *   减小 `batch_size`。
    *   使用更小的图像尺寸。
    *   尝试使用 `torch.utils.checkpoint` 来减少内存使用。
*   **梯度爆炸/消失:**
    *   使用梯度裁剪 (`clip_grad_norm`)。
    *   使用适当的权重初始化方法。
    *   降低学习率。
*   **训练不收敛:**
    *   检查学习率是否合适，调整学习率调整策略。
    *   检查数据预处理步骤是否正确，确保数据集质量。
    *   检查损失函数是否正确，以及是否使用了正确的损失权重。
*   **过拟合:**
    *   使用更强的数据增强。
    *   增加权重衰减。
    *   使用 Dropout。
    *   使用 Early Stopping。

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
*   `--model_path`: 训练好的模型权重文件的路径。  确保模型权重文件存在且与当前模型架构兼容。
*   `--target_fps`: 输出视频的目标帧率。  增加目标帧率可以提高视频的流畅度。
*   `--interpolation_factor`: 在每对输入帧之间插入的帧数。 此参数直接控制输出帧率。  `interpolation_factor = n` 将在每对输入帧之间插入 `n` 帧。
*   `--frame_interval`: 处理来自输入视频的每 n 帧。 默认为 1（处理所有帧）。 大于 1 的值允许跳过帧以进行测试或性能原因。 例如，`frame_interval = 2` 将只处理视频中的每隔一帧。
*   `--device`: 用于推理的设备（'cuda' 或 'cpu'）。

**重要:** 确保已正确安装和配置 OpenCV (cv2) 以进行视频读取和写入。

**错误处理:**

*   **视频读取错误:**
    *   检查输入视频文件是否存在且有效。
    *   确保 OpenCV 支持您使用的视频格式。
    *   尝试安装或更新相关的视频编解码器。
*   **模型加载错误:**
    *   确保模型文件存在且与当前模型架构兼容。
    *   检查模型文件是否损坏。
    *   如果模型文件是在不同的 PyTorch 版本下训练的，可能会出现兼容性问题。  建议在相同的 PyTorch 版本下进行推理。
*   **视频写入错误:**
    *   检查输出路径是否具有写入权限。
    *   确保 OpenCV 可以写入指定的视频格式。
    *   检查磁盘空间是否足够。
*   **设备不可用:**
    *   确保指定的设备（'cuda' 或 'cpu'）可用。
    *   如果使用 CUDA，请确保已正确安装 CUDA 驱动程序，并且 PyTorch 可以访问 GPU。

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

*   本项目受到 [相关研究论文和开源实现](https://arxiv.org/abs/1904.00842) 的启发。 具体来说，EMA-VFI 架构建立在指数移动平均以实现时间一致性和可变形卷积以实现自适应特征融合的原理之上。
*   Middlebury 数据集是光流和视频帧插值研究的宝贵资源。

## 许可证

本项目已获得 [MIT 许可证](LICENSE) 的许可。

## 贡献

欢迎贡献！ 请随时提交包含错误修复、改进或新功能的拉取请求。  请遵循代码风格指南，并提供清晰的提交信息。

## 联系方式

MiracleHcat@gmail.com