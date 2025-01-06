import glob
import numpy as np
import torch
import os
from unet import UNet
from pathlib import Path
from PIL import Image


def preprocess_image(image_path):
    """
    读取并预处理图片，转换为 PyTorch 所需格式。
    """
    # 打开图片并转为灰度
    image = Image.open(image_path).convert('L')
    # 转换为 numpy 数组
    image = np.array(image, dtype=np.float32)
    # 调整形状为 (batch_size, channel, height, width)
    image = image[np.newaxis, np.newaxis, ...]
    return torch.from_numpy(image)


def postprocess_prediction(prediction):
    """
    对预测结果进行后处理，生成二值化图像。
    """
    prediction = prediction[0][0].cpu().numpy()  # 提取预测结果
    prediction = (prediction >= 0.5) * 255  # 二值化
    return prediction.astype(np.uint8)


def save_result(prediction, save_path):
    """
    保存预测结果到文件。
    """
    result_image = Image.fromarray(prediction)
    result_image.save(save_path)


def main():
    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = UNet(n_channels=1, n_classes=1, bilinear=False)
    net.to(device=device)

    # 加载模型参数
    checkpoint = torch.load('model_pth', map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    # 测试集路径
    test_data_path = Path('./test_data/')
    test_images = list(test_data_path.glob('*.png'))

    # 遍历所有测试图片
    for test_image_path in test_images:
        save_res_path = test_image_path.with_name(f"{test_image_path.stem}_res.png")

        # 预处理图片
        img_tensor = preprocess_image(test_image_path)
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        with torch.no_grad():
            pred = net(img_tensor)

        # 后处理
        pred_processed = postprocess_prediction(pred)

        # 保存结果
        save_result(pred_processed, save_res_path)
        print(f"保存结果: {save_res_path}")


if __name__ == "__main__":
    main()
