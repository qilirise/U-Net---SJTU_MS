import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class SelfDataSet(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, '*.png'))

    def augment(self, image, flipcode):
        """
        图像翻转增强：
        - flipcode=0: 垂直翻转
        - flipcode=1: 水平翻转
        - flipcode=-1: 对角翻转
        """
        if flipcode == 0:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
        elif flipcode == 1:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flipcode == -1:
            return image.transpose(Image.ROTATE_180)
        return image

    def __getitem__(self, index):
        # 读取图片路径和对应标签路径
        image_path = self.imgs_path[index]
        label_path = image_path.replace('image', 'label')

        # 打开图片和标签，并转为灰度图
        image = Image.open(image_path).convert('L')  # 转为灰度图
        label = Image.open(label_path).convert('L')

        # 转换为 NumPy 数组
        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        # 归一化标签（如果最大值大于1）
        if label.max() > 1:
            label = label / 255.0

        # 图像增强（随机翻转）
        flipcode = random.choice([-1, 0, 1, None])
        if flipcode is not None:
            image = self.augment(Image.fromarray(image), flipcode)
            label = self.augment(Image.fromarray(label), flipcode)

            # 转回 NumPy 数组
            image = np.array(image, dtype=np.float32)
            label = np.array(label, dtype=np.float32)

        # 增加通道维度 (1, H, W)
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]

        return image, label

    def __len__(self):
        return len(self.imgs_path)


if __name__ == '__main__':
    data_path = "./train_image/"
    plate_dataset = SelfDataSet(data_path)
    print(f"数据集大小: {len(plate_dataset)}")

    # 使用 DataLoader 加载数据
    from torch.utils.data import DataLoader
    train_loader = DataLoader(dataset=plate_dataset, batch_size=5, shuffle=True)

    for batch_idx, (image, label) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}:")
        print(f"  图像形状: {image.shape}")
        print(f"  标签形状: {label.shape}")
