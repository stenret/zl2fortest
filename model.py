import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# 高斯核函数（修复维度不匹配问题）
class GaussianKernel(nn.Module):
    def __init__(self, sigma=None):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma if sigma is not None else config.MMD_SIGMA

    def forward(self, x, y):
        """
        修复：先将x/y映射到统一维度，再计算高斯核
        x: (n1, d1)  源平台特征
        y: (n2, d2)  目标平台特征
        return: 高斯核矩阵 (n1, n2)
        """
        # 步骤1：统一特征维度为HIDDEN_DIM（128维）
        if x.shape[1] != config.HIDDEN_DIM:
            x_proj = nn.Linear(x.shape[1], config.HIDDEN_DIM).to(x.device)(x)
        else:
            x_proj = x

        if y.shape[1] != config.HIDDEN_DIM:
            y_proj = nn.Linear(y.shape[1], config.HIDDEN_DIM).to(y.device)(y)
        else:
            y_proj = y

        # 步骤2：计算欧式距离（修复矩阵乘法维度问题）
        x_norm = (x_proj ** 2).sum(1).view(-1, 1)
        y_norm = (y_proj ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2 * torch.mm(x_proj, y_proj.t())

        # 步骤3：高斯核计算
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        return kernel


# 特征蒸馏主模型（保留原逻辑，仅适配MMD损失）
class FeatureDistillationModel(nn.Module):
    def __init__(self, platform_feat_dims):
        super(FeatureDistillationModel, self).__init__()
        self.platform_feat_dims = platform_feat_dims

        # 平台编码器（不同维度输入→统一128维）
        self.platform_encoders = nn.ModuleDict({
            platform: nn.Linear(feat_dim, config.HIDDEN_DIM)
            for platform, feat_dim in platform_feat_dims.items()
        })

        # 全局编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE)
        )

        # 平台解码器
        self.platform_decoders = nn.ModuleDict({
            platform: nn.Linear(config.HIDDEN_DIM, feat_dim)
            for platform, feat_dim in platform_feat_dims.items()
        })

        # 下游任务头
        self.task_heads = nn.ModuleDict({
            platform: nn.Linear(config.HIDDEN_DIM, 1)
            for platform in platform_feat_dims.keys()
        })

        # 损失函数
        self.reconstruction_loss = nn.MSELoss()
        self.mmd_loss = GaussianKernel(config.MMD_SIGMA)  # 使用修复后的高斯核
        self.sigmoid = nn.Sigmoid()

    def forward(self, platform, x):
        # 平台编码
        platform_feat = self.platform_encoders[platform](x)
        # 全局编码
        global_feat = self.global_encoder(platform_feat)
        # 平台解码（重构）
        recon_feat = self.platform_decoders[platform](global_feat)
        # 下游任务预测
        pred = self.sigmoid(self.task_heads[platform](global_feat))
        return global_feat, recon_feat, pred.squeeze()

    def calculate_loss(self, platform, x, y_true, cross_platform_feat=None):
        global_feat, recon_feat, pred = self.forward(platform, x)

        # 1. 重构损失
        recon_loss = self.reconstruction_loss(recon_feat, x)

        # 2. 下游任务损失
        task_loss = F.binary_cross_entropy(pred, y_true.float())

        # 3. MMD损失（跨平台分布对齐）
        if cross_platform_feat is not None:
            mmd_kernel = self.mmd_loss(global_feat, cross_platform_feat)
            mmd_loss = mmd_kernel.mean()
        else:
            mmd_loss = torch.tensor(0.0).to(x.device)

        # 总损失
        total_loss = (
                config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + config.MMD_LOSS_WEIGHT * mmd_loss
                + task_loss
        )

        return total_loss, recon_loss, mmd_loss, task_loss