import requests
import os
import zipfile

def download_and_extract_zip(url, extract_to):
    """
    从 URL 下载 zip 文件并解压到指定目录。
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查是否成功下载

        # 创建目录
        os.makedirs(extract_to, exist_ok=True)

        # 下载 zip 文件到内存
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # 解压所有文件到指定目录
        zip_file.extractall(extract_to)

        print(f"成功下载并解压到: {extract_to}")

    except requests.exceptions.RequestException as e:
        print(f"下载文件时出错: {e}")
    except zipfile.BadZipFile as e:
        print(f"解压 zip 文件时出错: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

import io

# 示例用法（需要替换为 Middlebury 数据集的真实 URL）
url = "https://vision.middlebury.edu/flow/data/comp/zip/other-gray-allframes.zip"  # 例如："http://vision.middlebury.edu/flow/data/comp-scene/RubberWhale-perfect.zip"
extract_to = "data/gray"  #  指定解压目录

download_and_extract_zip(url, extract_to)