import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# 1. 基础模型：仅原始特征训练（无特征蒸馏）
class RawFeatureModel(nn.Module):
    def __init__(self, input_dim):
        super(RawFeatureModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x.squeeze()

    def calculate_loss(self, x, y_true):
        pred = self.forward(x)
        loss = F.binary_cross_entropy(pred, y_true.float())
        return loss

# 2. 仅MMD损失模型（无重构损失）
class OnlyMMDModel(nn.Module):
    def __init__(self, platform_feat_dims, sigma=config.MMD_SIGMA):
        super(OnlyMMDModel, self).__init__()
        # 全局编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )
        # 平台编码器
        self.platform_encoders = nn.ModuleDict({
            platform: nn.Linear(feat_dim, config.HIDDEN_DIM)
            for platform, feat_dim in platform_feat_dims.items()
        })
        # 任务头
        self.task_heads = nn.ModuleDict({
            platform: nn.Linear(config.HIDDEN_DIM, 1)
            for platform in platform_feat_dims.keys()
        })
        # MMD损失
        self.mmd_loss = self._mmd_loss
        self.sigma = sigma

    def _mmd_loss(self, x, y):
        # 简化MMD计算
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2 * torch.mm(x, y.t())
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        xx = kernel.mean()
        yy = kernel.mean()
        xy = kernel.mean()
        return xx + yy - 2 * xy

    def forward(self, platform, x):
        platform_feat = self.platform_encoders[platform](x)
        global_feat = self.global_encoder(platform_feat)
        pred = torch.sigmoid(self.task_heads[platform](global_feat))
        return global_feat, pred.squeeze()

    def calculate_loss(self, platform, x, y_true, cross_feat=None):
        global_feat, pred = self.forward(platform, x)
        task_loss = F.binary_cross_entropy(pred, y_true.float())
        if cross_feat is not None:
            mmd_loss = self.mmd_loss(global_feat, cross_feat)
            total_loss = 0.7 * task_loss + 0.3 * mmd_loss
        else:
            total_loss = task_loss
        return total_loss, pred

# 3. LightGCN轻量化Baseline（简化版）
class LightGCN_Baseline(nn.Module):
    def __init__(self, input_dim):
        super(LightGCN_Baseline, self).__init__()
        self.embedding = nn.Linear(input_dim, config.HIDDEN_DIM)
        self.layer1 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        self.layer2 = nn.Linear(config.HIDDEN_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.embedding(x)
        x2 = self.layer1(x1)
        x = x1 + x2  # GCN残差连接
        pred = self.sigmoid(self.layer2(x))
        return pred.squeeze()

    def calculate_loss(self, x, y_true):
        pred = self.forward(x)
        loss = F.binary_cross_entropy(pred, y_true.float())
        return loss

# 获取Baseline模型
def get_baseline_model(model_name, input_dim=None, platform_feat_dims=None):
    if model_name == "RawFeature":
        return RawFeatureModel(input_dim)
    elif model_name == "OnlyMMD":
        return OnlyMMDModel(platform_feat_dims)
    elif model_name == "LightGCN_Baseline":
        return LightGCN_Baseline(input_dim)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")