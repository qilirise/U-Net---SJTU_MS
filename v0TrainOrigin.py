import torch
import torch.optim
from dataloader import SelfDataSet
from log import Logger
from plot import plot_picture
import os
import torch.nn as nn
from unet import UNet
from torch.utils.data import DataLoader
from torch import optim
import time
from torch.cuda.amp import autocast, GradScaler  # 导入混合精度工具
import matplotlib.pyplot as plt


# 定义训练函数
def Train_Unet(net, device, data_path, batch_size=8, epochs=40, lr=0.0001):
    # 加载数据集（无数据增强）
    train_dataset = SelfDataSet(data_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义优化算法
    opt = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    # 定义损失函数 (BCE + Dice Loss)
    class DiceLoss(nn.Module):
        def forward(self, pred, target):
            smooth = 1.0
            pred = torch.sigmoid(pred)
            intersection = (pred * target).sum()
            return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    class BCEDiceLoss(nn.Module):
        def __init__(self):
            super(BCEDiceLoss, self).__init__()
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()

        def forward(self, pred, target):
            return self.bce(pred, target) + self.dice(pred, target)

    loss_fun = BCEDiceLoss()

    # 创建日志记录
    Unet_train_txt = Logger('Unet_train.txt')

    # 初始化梯度缩放器
    scaler = GradScaler()

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 最优模型的初始损失
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        i = 0
        start_time = time.perf_counter()

        for image, label in train_loader:
            opt.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            # 使用 autocast 进行混合精度计算
            with autocast():
                pred = net(image)
                loss = loss_fun(pred, label)

            # 使用梯度缩放进行反向传播
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)  # 梯度裁剪
            scaler.step(opt)
            scaler.update()

            i += 1
            running_loss += loss.item()

        # 计算每轮的平均损失
        loss_avg_epoch = running_loss / i
        Unet_train_txt.write(f"{loss_avg_epoch:.4f}\n")
        end_time = time.perf_counter()

        # 调整学习率
        scheduler.step()

        # 打印训练信息
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {loss_avg_epoch:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}, Time: {end_time - start_time:.2f}s")

        # 保存最优模型
        if loss_avg_epoch < best_loss:
            best_loss = loss_avg_epoch
            state = {'net': net.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}
            torch.save(state, 'model_best.pth')
            print(f"Epoch {epoch + 1}: Best model saved with loss {best_loss:.4f}")

    torch.cuda.empty_cache()
    Unet_train_txt.close()


def tryGPU(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# 主函数
if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True  # 提高卷积计算速度
    device = tryGPU()
    print(device)
    net = UNet(1, 1, bilinear=False)
    net.to(device=device)

    # 数据集路径
    data_path = "./train_image/"
    Train_Unet(net, device, data_path, epochs=10, batch_size=16, lr=0.01)

    # 绘制训练曲线
    opt = 'Adam'
    lr = 0.01
    epochs = 20
    plot_picture('Unet_train.txt')
    plt.savefig(f'{opt} + lr = {lr} + epochs = {epochs}.png')

