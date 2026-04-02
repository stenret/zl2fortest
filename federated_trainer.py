import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
import time
import numpy as np
import config
from model import FeatureDistillationModel
from utils import setup_logger, set_seed, calculate_metrics

logger = setup_logger()


class FederatedDistillationTrainer:
    """
    联邦蒸馏训练器：实现跨平台知识蒸馏，同时保护数据隐私
    
    核心思想：
    1. 各平台数据保留在本地，只训练本地模型
    2. 通过聚合参数构建全局模型（不交换原始数据）
    3. 全局模型作为"知识中介"进行蒸馏到各平台模型
    """
    
    def __init__(self, platform_configs, device="cuda"):
        """
        :param platform_configs: 平台配置字典
            {
                "Books": {"feat_dim": 6, "target": "CTR"},
                "Electronics": {"feat_dim": 7, "target": "CVR"},
                "Clothing": {"feat_dim": 5, "target": "Interaction"}
            }
        :param device: 计算设备
        """
        self.platform_configs = platform_configs
        self.device = device
        self.platform_feat_dims = {
            p: cfg["feat_dim"] for p, cfg in platform_configs.items()
        }
        
        # 初始化全局模型（所有平台共享的结构）
        self.global_model = FeatureDistillationModel(self.platform_feat_dims).to(device)
        
        # 各平台的本地模型
        self.platform_models = {
            p: FeatureDistillationModel(self.platform_feat_dims).to(device)
            for p in platform_configs.keys()
        }
        
        logger.info(f"联邦蒸馏训练器初始化完成")
        logger.info(f"  - 平台数量：{len(platform_configs)}")
        logger.info(f"  - 全局模型参数量：{self._count_parameters(self.global_model):.3f}M")
    
    def _count_parameters(self, model):
        """统计模型参数量（百万级）"""
        return sum(p.numel() for p in model.parameters()) / 1e6
    
    def train_federation_round(self, round_idx, dataloaders, local_epochs=3):
        """
        一轮联邦训练
        :param round_idx: 轮次索引
        :param dataloaders: 各平台的数据加载器字典
        :param local_epochs: 每个平台本地训练的 epoch 数
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Federated Round {round_idx + 1}/{config.FEDERATED_ROUNDS}")
        logger.info(f"{'='*60}")
        
        round_start_time = time.time()
        
        # ========== 阶段 1：各平台本地训练（数据不出本地） ==========
        local_updates = {}
        
        for platform in self.platform_configs.keys():
            logger.info(f"\n--- 平台 {platform} 本地训练 ---")
            
            # 获取当前平台的模型和数据
            local_model = self.platform_models[platform]
            platform_loader = dataloaders[platform]
            
            # 在本地数据上训练
            start_time = time.time()
            self._local_train(
                model=local_model,
                dataloader=platform_loader,
                platform=platform,
                epochs=local_epochs
            )
            end_time = time.time()
            
            logger.info(f"{platform} 本地训练耗时：{end_time - start_time:.2f}s")
            
            # 记录本地模型的参数更新
            local_updates[platform] = copy.deepcopy(local_model.state_dict())
        
        # ========== 阶段 2：联邦平均聚合（FedAvg） ==========
        logger.info("\n--- 聚合各平台参数 (FedAvg) ---")
        
        # 可选：添加差分隐私噪声
        if config.PRIVACY_NOISE_STD > 0:
            logger.info(f"添加差分隐私噪声：std={config.PRIVACY_NOISE_STD}")
            for platform in local_updates.keys():
                for key in local_updates[platform].keys():
                    noise = torch.randn_like(local_updates[platform][key]) * config.PRIVACY_NOISE_STD
                    local_updates[platform][key] += noise
        
        aggregated_state = self._aggregate_updates(local_updates)
        
        # 更新全局模型
        self.global_model.load_state_dict(aggregated_state)
        logger.info("全局模型已更新")
        
        # ========== 阶段 3：知识蒸馏（全局模型 → 各平台模型） ==========
        logger.info("\n--- 知识蒸馏（全局模型 → 各平台模型） ---")
        
        for platform in self.platform_configs.keys():
            logger.info(f"\n平台 {platform} 知识蒸馏...")
            
            self._knowledge_distill(
                teacher_model=self.global_model,
                student_model=self.platform_models[platform],
                dataloader=dataloaders[platform],
                platform=platform
            )
        
        round_end_time = time.time()
        logger.info(f"\nRound {round_idx + 1} 总耗时：{round_end_time - round_start_time:.2f}s")
        
        return self.global_model
    
    def _local_train(self, model, dataloader, platform, epochs=3):
        """
        本地训练（只使用本平台数据，不涉及跨平台数据交换）
        """
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
        
        best_loss = float("inf")
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in dataloader["train"]:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                # 前向传播
                global_feat, recon_feat, pred = model.forward(platform, X_batch)
                
                # 计算损失（只用本平台的任务损失 + 重构损失）
                # 注意：本地训练不使用 MMD 损失（避免需要其他平台数据）
                recon_loss = F.mse_loss(recon_feat, X_batch)
                task_loss = F.binary_cross_entropy(pred.squeeze(), y_batch.float())
                
                # 总损失
                loss = recon_loss + task_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            
            avg_train_loss = train_loss / len(dataloader["train"].dataset)
            
            # 验证
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in dataloader["val"]:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    global_feat, recon_feat, pred = model.forward(platform, X_batch)
                    recon_loss = F.mse_loss(recon_feat, X_batch)
                    task_loss = F.binary_cross_entropy(pred.squeeze(), y_batch.float())
                    val_loss += (recon_loss + task_loss).item() * X_batch.size(0)
            
            avg_val_loss = val_loss / len(dataloader["val"].dataset)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
            
            logger.info(
                f"{platform} Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )
    
    def _aggregate_updates(self, local_updates, weights=None):
        """
        联邦平均（FedAvg）聚合各平台参数
        :param local_updates: 各平台的模型参数字典
        :param weights: 各平台权重（默认按数据量加权）
        """
        if weights is None:
            # 简单平均
            weights = {p: 1.0 / len(local_updates) for p in local_updates.keys()}
        
        aggregated_state = {}
        
        # 获取所有参数键
        param_keys = local_updates[list(local_updates.keys())[0]].keys()
        
        for key in param_keys:
            # 加权平均各平台的参数
            aggregated_param = torch.zeros_like(
                local_updates[list(local_updates.keys())[0]][key]
            )
            
            for platform, update in local_updates.items():
                aggregated_param += weights[platform] * update[key].to(self.device)
            
            aggregated_state[key] = aggregated_param
        
        return aggregated_state
    
    def _knowledge_distill(self, teacher_model, student_model, dataloader, platform):
        """
        知识蒸馏：从全局模型（教师）蒸馏到平台模型（学生）
        """
        teacher_model.eval()
        student_model.train()
        
        optimizer = optim.Adam(student_model.parameters(), lr=config.LEARNING_RATE)
        
        # 蒸馏温度参数
        temperature = config.DISTILL_TEMPERATURE
        alpha = config.DISTILL_ALPHA  # 蒸馏损失权重
        
        total_distill_loss = 0.0
        total_hard_loss = 0.0
        
        for X_batch, y_batch in dataloader["train"]:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            optimizer.zero_grad()
            
            # ========== 教师输出（软标签） ==========
            with torch.no_grad():
                # 教师模型的前向传播
                teacher_global, teacher_recon, teacher_pred = teacher_model.forward(
                    platform, X_batch
                )
                
                # 生成软标签（soft targets）
                soft_logits = teacher_pred / temperature
                soft_targets = torch.sigmoid(soft_logits)
            
            # ========== 学生输出 ==========
            student_global, student_recon, student_pred = student_model.forward(
                platform, X_batch
            )
            
            # ========== 计算损失 ==========
            # 1. 硬标签损失（原始任务）
            hard_loss = F.binary_cross_entropy(student_pred.squeeze(), y_batch.float())
            
            # 2. 蒸馏损失（模仿教师的软标签分布）
            student_soft_logits = student_pred / temperature
            distill_loss = F.mse_loss(
                torch.sigmoid(student_soft_logits),
                soft_targets
            )
            
            # 3. 重构损失
            recon_loss = F.mse_loss(student_recon, X_batch)
            
            # 总损失
            # loss = (1 - alpha) * hard_loss + alpha * distill_loss + recon_loss
            loss = alpha * distill_loss + (1 - alpha) * hard_loss + recon_loss
            
            loss.backward()
            optimizer.step()
            
            total_distill_loss += distill_loss.item()
            total_hard_loss += hard_loss.item()
        
        avg_distill_loss = total_distill_loss / len(dataloader["train"])
        avg_hard_loss = total_hard_loss / len(dataloader["train"])
        
        logger.info(
            f"{platform} 蒸馏完成 - "
            f"Distill Loss: {avg_distill_loss:.6f}, "
            f"Hard Loss: {avg_hard_loss:.6f}"
        )
    
    def get_platform_model(self, platform):
        """获取指定平台的模型"""
        return self.platform_models[platform]
    
    def get_global_model(self):
        """获取全局模型"""
        return self.global_model
    
    def save_all_models(self, save_dir):
        """保存所有模型"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存全局模型
        global_model_path = os.path.join(save_dir, "global_fed_model.pth")
        torch.save(self.global_model.state_dict(), global_model_path)
        logger.info(f"全局模型已保存：{global_model_path}")
        
        # 保存各平台模型
        for platform, model in self.platform_models.items():
            platform_model_path = os.path.join(
                save_dir, 
                f"{platform}_fed_best_model.pth"
            )
            torch.save(model.state_dict(), platform_model_path)
            logger.info(f"{platform} 平台模型已保存：{platform_model_path}")


def evaluate_federated_model(model, dataloader, platform, device, model_name="Federated"):
    """
    评估联邦训练后的模型
    """
    from utils import calculate_metrics
    
    model.eval()
    all_y_true = []
    all_y_pred_prob = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader["test"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            _, _, y_pred_prob = model.forward(platform, X_batch)
            
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred_prob.extend(y_pred_prob.cpu().numpy())
    
    # 根据平台目标类型选择评估指标
    target_type = config.PLATFORM_CONFIG[platform]["target"]
    
    if target_type == "CTR" or target_type == "CVR_Classification":
        task_type = "classification"
    elif target_type == "Interaction":
        task_type = "ranking"
    else:
        task_type = "classification"
    
    metrics = calculate_metrics(
        np.array(all_y_true), 
        np.array(all_y_pred_prob), 
        task_type=task_type
    )
    
    logger.info(f"{model_name} - Platform {platform} ({target_type}) metrics: {metrics}")
    
    return metrics
