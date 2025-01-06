import torch
from torch import nn
from torch.nn import functional as F
from Oceanography.NetPrepares.GPU.tryGPU import tryGPU as tryGPU
from Oceanography.NetPrepares.Load.LoadData import LoadData as LoadData

# 定义动态注意力模块
class DynamicAttention(nn.Module):
    def __init__(self, input_channels, num_hiddens, dropout=0.1):
        super(DynamicAttention, self).__init__()
        self.W_q = nn.Linear(input_channels, num_hiddens, bias=False)
        self.W_k = nn.Linear(input_channels, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        batch_size, channels, height, width = queries.shape

        # 检查 input_channels 是否匹配
        assert channels == self.W_q.in_features, (
            f"Input channels ({channels}) do not match W_q input features ({self.W_q.in_features})"
        )

        # 调整形状为 (batch_size, num_elements, channels)
        queries = queries.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        keys = keys.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        values = values.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)

        # 计算注意力
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = F.softmax(scores, dim=-1)

        # 加权求和
        output = torch.bmm(self.dropout(self.attention_weights), values)
        return output.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)

# 自定义封装模块，用于 nn.Sequential
class AttentionBlock(nn.Module):
    def __init__(self, input_channels, num_hiddens, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = DynamicAttention(input_channels, num_hiddens, dropout)

    def forward(self, X):
        return self.attention(X, X, X, None)

# 定义残差块
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# 定义残差网络的层级
def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

# 网络层级定义
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 注意力模块
attention_block = AttentionBlock(input_channels=128, num_hiddens=128, dropout=0.1)

# 构建 ResNet 并加入注意力模块
net = nn.Sequential(
    b1, b2, b3, attention_block, b4, b5,  # 在 Flatten 之前加入注意力模块
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(512, 10)
)

# 检查网络输出形状
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# 模型训练及超参数
lr, num_epochs, batch_size = 0.05, 15, 128
nc_file = r"/data/EddyWave/Data_SCS.nc"
var_names=['SSH_Train','SSH_Val', 'SSH_Test']
train_iter = data_path = "./train_image/"
device = tryGPU()
train(net, train_iter, num_epochs, lr, device)
