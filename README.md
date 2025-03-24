[English](README.md) | [简体中文](README_zh.md)

# Video Frame Interpolation using EMA-VFI

This project implements a video frame interpolation (VFI) model based on Exponential Moving Average (EMA) and Deformable Convolutions. It aims to generate intermediate frames between two given frames, thereby increasing the frame rate and improving the smoothness of videos.

## Model Architecture

The model architecture consists of the following key components:

*   **Feature Extraction:** Extracts deep features from the input frames using convolutional blocks.
*   **Context Encoding:** Encodes global context information to understand the scene.
*   **Motion Estimation:** Estimates the optical flow between the input frames.
*   **Warping:** Warps one frame to the viewpoint of the other using the estimated optical flow.
*   **Multi-Attention Fusion:** Fuses the original and warped features using modulated deformable convolutions.
*   **Reconstruction:** Reconstructs the interpolated frame from the fused features.

## Dataset

This project uses (or is designed to be trained on) the [Middlebury Optical Flow Dataset](https://vision.middlebury.edu/flow/data/) for training and evaluation. This dataset provides a set of calibrated stereo image pairs, along with ground truth optical flow fields.

**Note:** You'll need to download and pre-process the Middlebury dataset before training. Refer to the [Data Preparation](#data-preparation) section for more details.  If you plan to use your own dataset, make sure it's properly preprocessed and formatted to be compatible with the model and training script.

## Dependencies

*   Python 3.7+
*   PyTorch 1.10+
*   Torchvision 0.11+ (required for `DeformConv2d`)
*   YAML
*   tqdm
*   OpenCV (cv2)
*   (Other dependencies listed in `requirements.txt`)

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

1.  **Download the Middlebury Dataset:**
    *   Download the desired stereo image pairs from [https://vision.middlebury.edu/flow/data/](https://vision.middlebury.edu/flow/data/).
    *   Choose the appropriate dataset version (e.g., "Other data sets").

2.  **Pre-process the Dataset:**

    This project requires the data to be organized in a specific directory structure:

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

    Where:

    *   `data/processed/train/` contains the training scenes.
    *   `data/processed/val/` contains the validation scenes.
    *   Each scene directory contains a sequence of frames.
    *   The filenames should be sequential (e.g., `frame0001.png`, `frame0002.png`, ...).

    You can use the provided `src/utils/data_utils.py` and modify it to load and pre-process the Middlebury dataset according to the required directory structure.  Adapt the `VideoDataset` class to load image pairs from the Middlebury dataset. You'll likely need to write custom scripts to convert the Middlebury data format to the format expected by the `VideoDataset`.

    **Important Considerations:**

    *   **Data Augmentation:** Consider data augmentation techniques (e.g., random crops, flips, rotations) to improve the model's generalization ability.  Implement these augmentations within your data loading pipeline.
    *   **File Format:** Ensure the image files are in a format compatible with OpenCV (e.g., PNG, JPG).
    *   **Resolution:**  The model might have specific input resolution requirements.  Resize the images as necessary during pre-processing.
    *    **`.flo` file conversion:**  The Middlebury dataset includes optical flow information in `.flo` files. If your training pipeline uses this information, you'll need to convert these files into a suitable format (e.g., images or NumPy arrays).  There are existing tools and libraries available for reading `.flo` files (search online for "read .flo file python").

## Training

1.  **Configure the Training:**

    Modify the `config/train_config.yaml` file to adjust the training parameters:

    ```yaml
    batch_size: 2
    learning_rate: 0.0003
    num_epochs: 20
    train_data_dir: 'data/processed/train'
    val_data_dir: 'data/processed/val'
    checkpoint_path: 'checkpoints'
    best_model_path: 'best_ema_vfi.pth'
    device: 'cuda'  # or 'cpu'
    ```

    *   `batch_size`: Batch size for training.
    *   `learning_rate`: Learning rate for the AdamW optimizer.
    *   `num_epochs`: Number of training epochs.
    *   `train_data_dir`: Path to the training data directory.
    *   `val_data_dir`: Path to the validation data directory.
    *   `checkpoint_path`: Path to store model checkpoints.
    *   `best_model_path`: File name to save the best model.
    *   `device`: Specify the device to use for training ('cuda' or 'cpu').

2.  **Start Training:**

    Run the `train.py` script:

    ```bash
    python train.py
    ```

    The training progress, including the loss and validation loss, will be displayed in the console.  The best model will be saved to the specified `best_model_path`.  TensorBoard integration for monitoring training progress is highly recommended.

## Evaluation

Currently, there is no dedicated evaluation script.  Evaluation can be performed by modifying the `train.py` script to run inference on a separate validation or test dataset after training.  This would involve:

1.  Loading the trained model.
2.  Preparing the test dataset using the same pre-processing steps as the training data.
3.  Iterating through the test dataset, feeding image pairs to the model.
4.  Calculating appropriate evaluation metrics (e.g., PSNR, SSIM) between the interpolated frames and the ground truth frames (if available).  If ground truth is unavailable, qualitative assessment is required.
5.  Reporting the evaluation results.

Implementing a separate `eval.py` script for this purpose is a recommended future improvement.

## Inference

To perform video frame interpolation using a trained model, run the `inference.py` script (or your equivalent):

```bash
python inference.py --input_video input.mp4 --output_video output.mp4 --model_path best_ema_vfi.pth --target_fps 60 --interpolation_factor 1 --frame_interval 1 --device cuda
```

or, using the default values defined in the script:

```bash
python inference.py
```

Command-line arguments:

*   `--input_video`:  Path to the input video file.
*   `--output_video`: Path to the output video file.
*   `--model_path`: Path to the trained model weights file.
*   `--target_fps`: Target frame rate of the output video.
*   `--interpolation_factor`: Number of frames to insert between each pair of input frames.  This parameter directly controls the output frame rate.
*   `--frame_interval`:  Process every n-th frame from the input video. Default is 1 (process all frames).  A value greater than 1 allows skipping frames for testing or performance reasons.
*   `--device`: Device to use for inference ('cuda' or 'cpu').

**Important:**  Ensure that OpenCV (cv2) is installed and configured correctly for video reading and writing.

## Project Structure

```
.
├── config/
│   └── train_config.yaml         # Training configuration file
├── src/
│   ├── models/
│   │   └── ema_vfi.py             # EMA-VFI model definition
│   ├── utils/
│   │   ├── data_utils.py           # Dataset and data loading utilities
│   │   ├── loss_functions.py      # Loss function definitions
│   │   └── ...
├── train.py                       # Training script
├── inference.py                   # Inference script
├── README.md                      # This file
├── requirements.txt               # Project dependencies
└── LICENSE                        # License file
```

## Acknowledgements

*   This project is inspired by [relevant research papers and open-source implementations](Add links or citations to related work if applicable).  Specifically, the EMA-VFI architecture builds upon the principles of Exponential Moving Average for temporal consistency and Deformable Convolutions for adaptive feature fusion.  Cite relevant papers here.
*   The Middlebury dataset is a valuable resource for optical flow and video frame interpolation research.

## License

This project is licensed under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit pull requests with bug fixes, improvements, or new features.

## Contact

MiracleHcat@gmail.com