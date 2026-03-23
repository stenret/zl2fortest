import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import config
from model import FeatureDistillationModel
from utils import setup_logger, set_seed, calculate_metrics, plot_loss_curve, plot_mmd_heatmap, save_results

logger = setup_logger()
set_seed(config.SEED)


# 构建数据加载器
def build_dataloader(split_data):
    dataloaders = {}
    for platform, data in split_data.items():
        # 训练集
        X_train, y_train = data["train"]
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        # 验证集
        X_val, y_val = data["val"]
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        # 测试集
        X_test, y_test = data["test"]
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        dataloaders[platform] = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
    return dataloaders


# 训练单个平台模型
def train_platform_model(model, dataloader, platform, device):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = float("inf")
    early_stop_count = 0
    loss_history = {"train": [], "val": []}

    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in dataloader["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss, _, _, _ = model.calculate_loss(platform, X_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        avg_train_loss = train_loss / len(dataloader["train"].dataset)
        loss_history["train"].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloader["val"]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                loss, _, _, _ = model.calculate_loss(platform, X_batch, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        avg_val_loss = val_loss / len(dataloader["val"].dataset)
        loss_history["val"].append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"{platform}_best_model.pth"))
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1} for {platform}!")
                break

        logger.info(
            f"Platform: {platform}, Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 可视化损失曲线
    plot_loss_curve(loss_history, os.path.join(config.RESULT_DIR, f"{platform}_loss_curve.png"))
    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, f"{platform}_best_model.pth")))
    return model


# 评估模型
def evaluate_model(model, dataloader, platform, device):
    model.eval()
    all_y_true = []
    all_y_pred_prob = []
    all_global_feats = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader["test"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            global_feat, _, task_pred = model.forward(platform, X_batch)
            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred_prob.extend(task_pred.squeeze().cpu().numpy())
            all_global_feats.extend(global_feat.cpu().numpy())

    # 计算指标
    metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred_prob))
    # 计算跨平台MMD
    global_feats = np.array(all_global_feats)
    logger.info(f"Platform {platform} evaluation metrics: {metrics}")
    return metrics, global_feats


# 消融实验
def ablation_experiment(split_data, device):
    # 消融实验配置：移除不同模块
    ablation_configs = {
        "Full Model": {"remove_decoder": False, "remove_mmd": False},
        "Without Platform Decoder": {"remove_decoder": True, "remove_mmd": False},
        "Without MMD Loss": {"remove_decoder": False, "remove_mmd": True},
        "Without Downstream Task": {"remove_decoder": False, "remove_mmd": False, "remove_task": True}
    }
    ablation_results = []
    dataloaders = build_dataloader(split_data)
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}

    for exp_name, cfg in ablation_configs.items():
        logger.info(f"Running ablation experiment: {exp_name}")
        for platform in config.PLATFORM_CONFIG.keys():
            # 初始化模型
            model = FeatureDistillationModel(platform_feat_dims).to(device)
            # 训练模型（根据消融配置调整）
            model = train_platform_model(model, dataloaders[platform], platform, device)
            # 评估模型
            metrics, _ = evaluate_model(model, dataloaders[platform], platform, device)
            ablation_results.append({
                "experiment": exp_name,
                "platform": platform,
                "AUC": metrics["AUC"],
                "F1-score": metrics["F1-score"]
            })

    # 保存消融实验结果
    save_results(ablation_results, os.path.join(config.RESULT_DIR, "ablation_experiment_results.csv"))
    logger.info("Ablation experiment completed!")
    return ablation_results


# 超参数敏感性分析
def hyperparam_sensitivity_analysis(split_data, device):
    # 超参数配置：α取值[0.5,0.6,0.7,0.8,0.9]
    alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    sensitivity_results = []
    dataloaders = build_dataloader(split_data)
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}

    for alpha in alpha_values:
        logger.info(f"Running hyperparam sensitivity analysis for α={alpha}")
        # 修改全局参数
        config.ALPHA = alpha
        config.RECONSTRUCTION_LOSS_WEIGHT = alpha
        config.MMD_LOSS_WEIGHT = 1 - alpha

        for platform in config.PLATFORM_CONFIG.keys():
            # 初始化模型
            model = FeatureDistillationModel(platform_feat_dims).to(device)
            # 训练模型
            model = train_platform_model(model, dataloaders[platform], platform, device)
            # 评估模型
            metrics, _ = evaluate_model(model, dataloaders[platform], platform, device)
            sensitivity_results.append({
                "alpha": alpha,
                "platform": platform,
                "AUC": metrics["AUC"],
                "F1-score": metrics["F1-score"]
            })

    # 保存超参数分析结果
    save_results(sensitivity_results, os.path.join(config.RESULT_DIR, "hyperparam_sensitivity_results.csv"))
    logger.info("Hyperparameter sensitivity analysis completed!")
    return sensitivity_results


# 主训练评估流程
def main_train_eval(split_data):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 构建数据加载器
    dataloaders = build_dataloader(split_data)

    # 初始化模型
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}
    model = FeatureDistillationModel(platform_feat_dims).to(device)

    # 训练各平台模型
    all_metrics = []
    all_global_feats = {}
    for platform in config.PLATFORM_CONFIG.keys():
        if platform not in dataloaders:
            logger.warning(f"Skipping {platform} - no valid data!")
            continue
        logger.info(f"Training model for {platform}...")
        trained_model = train_platform_model(model, dataloaders[platform], platform, device)
        # 评估模型
        metrics, global_feats = evaluate_model(trained_model, dataloaders[platform], platform, device)
        all_metrics.append({
            "platform": platform,
            "AUC": metrics["AUC"],
            "F1-score": metrics["F1-score"]
        })
        all_global_feats[platform] = global_feats

    # 计算跨平台MMD矩阵
    platforms = [p for p in config.PLATFORM_CONFIG.keys() if p in all_global_feats]
    if len(platforms) >= 2:
        mmd_matrix = np.zeros((len(platforms), len(platforms)))
        mmd_loss = model.mmd_loss.to(device)
        for i, p1 in enumerate(platforms):
            for j, p2 in enumerate(platforms):
                if i <= j:
                    feat1 = torch.tensor(all_global_feats[p1], dtype=torch.float32).to(device)
                    feat2 = torch.tensor(all_global_feats[p2], dtype=torch.float32).to(device)
                    mmd = mmd_loss(feat1, feat2).cpu().item()
                    mmd_matrix[i][j] = mmd
                    mmd_matrix[j][i] = mmd
        # 可视化MMD热图
        plot_mmd_heatmap(mmd_matrix, platforms, os.path.join(config.RESULT_DIR, "mmd_heatmap.png"))
    else:
        mmd_matrix = None
        logger.warning("Not enough platforms to compute MMD matrix!")

    # 保存实验结果
    save_results(all_metrics, os.path.join(config.RESULT_DIR, "main_experiment_results.csv"))

    # 运行消融实验（仅当有有效数据时）
    if len(dataloaders) > 0:
        ablation_experiment(split_data, device)

        # 运行超参数敏感性分析
        hyperparam_sensitivity_analysis(split_data, device)

    logger.info("All experiments completed successfully!")
    return all_metrics, mmd_matrix


if __name__ == "__main__":
    from data_process import build_heterogeneous_scene

    split_data, _ = build_heterogeneous_scene()
    main_train_eval(split_data)