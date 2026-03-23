import torch
import torch.nn as nn
import torch.nn.functional as F
import config

# MMD损失计算
class MMDLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(MMDLoss, self).__init__()
        self.sigma = sigma

    def gaussian_kernel(self, x, y):
        # 计算欧式距离
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2 * torch.mm(x, y.t())
        # 高斯核
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        return kernel

    def forward(self, x, y):
        # 计算MMD
        xx = self.gaussian_kernel(x, x).mean()
        yy = self.gaussian_kernel(y, y).mean()
        xy = self.gaussian_kernel(x, y).mean()
        mmd = xx + yy - 2 * xy
        return mmd

# 全局特征编码器（论文中的双塔轻量化架构）
class GlobalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=config.HIDDEN_DIM, dropout_rate=config.DROPOUT_RATE):
        super(GlobalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return x

# 平台专属解码器
class PlatformDecoder(nn.Module):
    def __init__(self, input_dim=config.HIDDEN_DIM, output_dim=1):
        super(PlatformDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc(x))
        return x

# 完整特征蒸馏模型
class FeatureDistillationModel(nn.Module):
    def __init__(self, platform_feat_dims):
        super(FeatureDistillationModel, self).__init__()
        # 全局编码器（共享）
        self.global_encoder = GlobalEncoder(input_dim=config.HIDDEN_DIM)  # 统一隐空间维度
        # 平台专属编码器（用于特征重构）
        self.platform_encoders = nn.ModuleDict({
            platform: GlobalEncoder(input_dim=feat_dim)
            for platform, feat_dim in platform_feat_dims.items()
        })
        # 平台专属解码器
        self.platform_decoders = nn.ModuleDict({
            platform: PlatformDecoder(input_dim=config.HIDDEN_DIM, output_dim=feat_dim)
            for platform, feat_dim in platform_feat_dims.items()
        })
        # 下游任务分类头
        self.task_heads = nn.ModuleDict({
            platform: PlatformDecoder(input_dim=config.HIDDEN_DIM, output_dim=1)
            for platform in platform_feat_dims.keys()
        })
        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        self.mmd_loss = MMDLoss(sigma=config.MMD_SIGMA)

    def forward(self, platform, x):
        # 平台专属编码
        platform_feat = self.platform_encoders[platform](x)
        # 全局编码
        global_feat = self.global_encoder(platform_feat)
        # 重构特征
        recon_feat = self.platform_decoders[platform](global_feat)
        # 下游任务预测
        task_pred = self.task_heads[platform](global_feat)
        return global_feat, recon_feat, task_pred

    def calculate_loss(self, platform, x, y_true):
        # 前向传播
        global_feat, recon_feat, task_pred = self.forward(platform, x)
        # 重构损失
        recon_loss = self.reconstruction_loss(recon_feat, x)
        # 下游任务损失（交叉熵）
        task_loss = F.binary_cross_entropy(task_pred.squeeze(), y_true.float())
        # 联合损失（重构+MMD，MMD在训练时跨平台计算）
        total_loss = config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss + (1 - config.RECONSTRUCTION_LOSS_WEIGHT) * task_loss
        return total_loss, recon_loss, task_loss, global_feat

if __name__ == "__main__":
    # 测试模型初始化
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}
    model = FeatureDistillationModel(platform_feat_dims)
    print("Model initialized successfully!")
    print(model)