[English](README.md) | [简体中文](README_zh.md)

# 基于 EMA-VFI 的视频帧插值

本项目实现了一个基于指数移动平均 (EMA) 和可变形卷积的视频帧插值 (VFI) 模型。 它的目标是生成两个给定帧之间的中间帧，从而提高帧率并改善视频的流畅度。

## 模型架构

该模型架构包含以下关键组件：

*   **特征提取:** 使用卷积块从输入帧中提取深度特征。
*   **上下文编码:** 编码全局上下文信息以理解场景。
*   **运动估计:** 估计输入帧之间的光流。
*   **扭曲 (Warping):** 使用估计的光流将一帧扭曲到另一帧的视角。
*   **多注意力融合:** 使用调制的可变形卷积融合原始特征和扭曲特征。
*   **重建:** 从融合的特征中重建插值帧。

## 数据集

本项目使用（或设计为训练于）[Middlebury 光流数据集](https://vision.middlebury.edu/flow/data/) 进行训练和评估。 该数据集提供了一组校准的立体图像对，以及真实的光流场。

**注意:** 在训练之前，您需要下载并预处理 Middlebury 数据集。 有关更多详细信息，请参阅 [数据准备](#数据准备) 部分。 如果您计划使用自己的数据集，请确保对其进行适当的预处理和格式化，以使其与模型和训练脚本兼容。

## 依赖项

*   Python 3.7+
*   PyTorch 1.10+
*   Torchvision 0.11+ (需要 `DeformConv2d`)
*   YAML
*   tqdm
*   OpenCV (cv2)
*   （`requirements.txt` 中列出的其他依赖项）

要安装依赖项，请运行：

```bash
pip install -r requirements.txt
```

## 数据准备

1.  **下载 Middlebury 数据集:**
    *   从 [https://vision.middlebury.edu/flow/data/](https://vision.middlebury.edu/flow/data/) 下载所需的立体图像对。
    *   选择适当的数据集版本（例如，“其他数据集”）。

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
    *   文件名应为顺序的（例如，`frame0001.png`，`frame0002.png`，...）。

    您可以使用提供的 `src/utils/data_utils.py` 并对其进行修改，以根据所需的目录结构加载和预处理 Middlebury 数据集。 调整 `VideoDataset` 类以从 Middlebury 数据集加载图像对。 您可能需要编写自定义脚本以将 Middlebury 数据格式转换为 `VideoDataset` 期望的格式。

    **重要注意事项:**

    *   **数据增强:** 考虑使用数据增强技术（例如，随机裁剪、翻转、旋转）以提高模型的泛化能力。 在您的数据加载管道中实现这些增强。
    *   **文件格式:** 确保图像文件采用与 OpenCV 兼容的格式（例如，PNG，JPG）。
    *   **分辨率:** 模型可能具有特定的输入分辨率要求。 在预处理期间根据需要调整图像大小。
    *   **`.flo` 文件转换:** Middlebury 数据集包含 `.flo` 文件中的光流信息。 如果您的训练管道使用此信息，则需要将这些文件转换为合适的格式（例如，图像或 NumPy 数组）。 有现有的工具和库可用于读取 `.flo` 文件（在线搜索“python read .flo file”）。

## 训练

1.  **配置训练:**

    修改 `config/train_config.yaml` 文件以调整训练参数：

    ```yaml
    batch_size: 2
    learning_rate: 0.0003
    num_epochs: 20
    train_data_dir: 'data/processed/train'
    val_data_dir: 'data/processed/val'
    checkpoint_path: 'checkpoints'
    best_model_path: 'best_ema_vfi.pth'
    device: 'cuda'  # 或 'cpu'
    ```

    *   `batch_size`: 训练的批次大小。
    *   `learning_rate`: AdamW 优化器的学习率。
    *   `num_epochs`: 训练 epoch 的数量。
    *   `train_data_dir`: 训练数据目录的路径。
    *   `val_data_dir`: 验证数据目录的路径。
    *   `checkpoint_path`: 存储模型检查点的路径。
    *   `best_model_path`: 用于保存最佳模型的文件名。
    *   `device`: 指定用于训练的设备（'cuda' 或 'cpu'）。

2.  **开始训练:**

    运行 `train.py` 脚本:

    ```bash
    python train.py
    ```

    训练进度，包括损失和验证损失，将显示在控制台中。 最佳模型将保存到指定的 `best_model_path`。 强烈建议使用 TensorBoard 集成来监控训练进度。

## 评估

目前，没有专门的评估脚本。 可以通过修改 `train.py` 脚本在训练后对单独的验证或测试数据集运行推理来进行评估。 这将涉及：

1.  加载训练好的模型。
2.  使用与训练数据相同的预处理步骤准备测试数据集。
3.  迭代测试数据集，将图像对馈送到模型。
4.  计算插值帧和真实帧之间的适当评估指标（例如，PSNR，SSIM）（如果可用）。 如果没有真实数据，则需要进行定性评估。
5.  报告评估结果。

为此目的实现一个单独的 `eval.py` 脚本是建议的未来改进。

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

*   `--input_video`: 输入视频文件的路径。
*   `--output_video`: 输出视频文件的路径。
*   `--model_path`: 训练好的模型权重文件的路径。
*   `--target_fps`: 输出视频的目标帧率。
*   `--interpolation_factor`: 在每对输入帧之间插入的帧数。 此参数直接控制输出帧率。
*   `--frame_interval`: 处理来自输入视频的每 n 帧。 默认为 1（处理所有帧）。 大于 1 的值允许跳过帧以进行测试或性能原因。
*   `--device`: 用于推理的设备（'cuda' 或 'cpu'）。

**重要:** 确保已正确安装和配置 OpenCV (cv2) 以进行视频读取和写入。

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

*   本项目受到 [相关研究论文和开源实现](Add links or citations to related work if applicable) 的启发。 具体来说，EMA-VFI 架构建立在指数移动平均以实现时间一致性和可变形卷积以实现自适应特征融合的原理之上。 在此处引用相关论文。
*   Middlebury 数据集是光流和视频帧插值研究的宝贵资源。

## 许可证

本项目已获得 [MIT 许可证](LICENSE) 的许可。

## 贡献

欢迎贡献！ 请随时提交包含错误修复、改进或新功能的拉取请求。

## 联系方式

MiracleHcat@gmail.com