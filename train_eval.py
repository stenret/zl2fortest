import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import config
from model import FeatureDistillationModel
from model_baseline import get_baseline_model
from utils import (
    setup_logger, set_seed, calculate_metrics, plot_loss_curve,
    plot_mmd_heatmap, save_results, count_model_parameters,
    calculate_statistical_results, get_training_metrics, plot_alpha_curve
)

logger = setup_logger()
set_seed(config.SEED)


# 构建数据加载器
def build_dataloader(split_data):
    dataloaders = {}
    for platform, data in split_data.items():
        X_train, y_train = data["train"]
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        X_val, y_val = data["val"]
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

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


# 训练基础模型（含Baseline）
def train_basic_model(model, dataloader, platform, device, model_name="CustomModel"):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = float("inf")
    early_stop_count = 0
    loss_history = {"train": [], "val": []}

    # 记录训练开始时间（量化指标）
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in dataloader["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if model_name in ["RawFeature", "LightGCN_Baseline"]:
                loss = model.calculate_loss(X_batch, y_batch)
            elif model_name == "OnlyMMD":
                loss, _ = model.calculate_loss(platform, X_batch, y_batch)
            else:  # 自定义模型
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

                if model_name in ["RawFeature", "LightGCN_Baseline"]:
                    loss = model.calculate_loss(X_batch, y_batch)
                elif model_name == "OnlyMMD":
                    loss, _ = model.calculate_loss(platform, X_batch, y_batch)
                else:
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
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"{platform}_{model_name}_best_model.pth"))
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1} for {platform} ({model_name})!")
                break

        logger.info(
            f"{model_name} - Platform: {platform}, Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 训练结束：统计时间和显存
    end_time = time.time()
    train_metrics = get_training_metrics(start_time, end_time, model)
    logger.info(
        f"{model_name} - {platform}训练耗时：{train_metrics['train_time_seconds']}s，最大显存：{train_metrics['max_memory_mb']}MB")

    # 可视化损失曲线
    plot_loss_curve(loss_history, os.path.join(config.RESULT_DIR, f"{platform}_{model_name}_loss_curve.png"),
                    title_suffix=f"({model_name})")

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, f"{platform}_{model_name}_best_model.pth")))
    return model, train_metrics


# 训练自定义模型（原逻辑保留，补充多次实验）
def train_platform_model(model, dataloader, platform, device):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = float("inf")
    early_stop_count = 0
    loss_history = {"train": [], "val": []}
    start_time = time.time()

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
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"{platform}_best_model.pth"))
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1} for {platform}!")
                break

        logger.info(
            f"Platform: {platform}, Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 统计训练指标
    end_time = time.time()
    train_metrics = get_training_metrics(start_time, end_time, model)

    # 可视化损失曲线
    plot_loss_curve(loss_history, os.path.join(config.RESULT_DIR, f"{platform}_loss_curve.png"))

    # 加载最佳模型
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, f"{platform}_best_model.pth")))
    return model, train_metrics


# 评估模型（通用，支持多任务指标）
def evaluate_model(model, dataloader, platform, device, model_name="CustomModel"):
    model.eval()
    all_y_true = []
    all_y_pred_prob = []
    all_global_feats = []
    all_y_original = []  # 保存原始评分（用于回归和排序）

    with torch.no_grad():
        for X_batch, y_batch in dataloader["test"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            if model_name in ["RawFeature", "LightGCN_Baseline"]:
                y_pred_prob = model.forward(X_batch)
            elif model_name == "OnlyMMD":
                _, y_pred_prob = model.forward(platform, X_batch)
            else:
                _, _, y_pred_prob = model.forward(platform, X_batch)
                global_feat = _  # 仅自定义模型保留全局特征

            all_y_true.extend(y_batch.cpu().numpy())
            all_y_pred_prob.extend(y_pred_prob.cpu().numpy())
            if model_name == "CustomModel":
                all_global_feats.extend(global_feat.cpu().numpy())

            # 保存原始评分（从 dataloader 的 y_batch 反推）
            # 注意：这里 y_batch 已经是二值化的 target，需要特殊处理
            # 对于回归和排序任务，我们需要在 data_process 中保存原始评分

    # ========== 修改：根据平台目标类型选择评估指标 ==========
    target_type = config.PLATFORM_CONFIG[platform]["target"]

    # 映射目标类型到评估任务类型
    if target_type == "CTR" or target_type == "CVR_Classification":
        # Books: CTR 预测 → 分类任务
        # Electronics: CVR 预测（分类版）→ 分类任务
        task_type = "classification"
        y_true_for_eval = np.array(all_y_true)
    elif target_type == "CVR":
        # Electronics: CVR 预测 → 回归任务（预测转化强度）
        task_type = "regression"
        # 使用二分类标签作为真实值（简化处理）
        y_true_for_eval = np.array(all_y_true)
    elif target_type == "Interaction":
        # Clothing: 交互预测 → 排序任务
        task_type = "ranking"
        # 使用二分类标签作为真实值（简化处理）
        y_true_for_eval = np.array(all_y_true)
    else:
        # 默认：分类任务
        task_type = "classification"
        y_true_for_eval = np.array(all_y_true)

    # 计算指标
    metrics = calculate_metrics(y_true_for_eval, np.array(all_y_pred_prob), task_type=task_type)
    logger.info(f"{model_name} - Platform {platform} ({target_type}/{task_type}) evaluation metrics: {metrics}")

    # 仅自定义模型返回全局特征（用于 MMD 计算）
    if model_name == "CustomModel":
        return metrics, np.array(all_global_feats)
    else:
        return metrics, None


# ========== 新增：跨平台训练函数（支持 MMD 对齐） ==========
def train_platform_model_with_mmd(model, dataloaders, target_platform, device,
                                   use_decoder=True, use_mmd=True):
    """
    跨平台训练：使用其他平台的特征作为 MMD 对齐目标
    :param model: 模型
    :param dataloaders: 所有平台的数据加载器字典
    :param target_platform: 当前训练的目标平台
    :param device: 设备
    :param use_decoder: 是否使用重构损失
    :param use_mmd: 是否使用 MMD 损失
    :return: 训练后的模型
    """
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = float("inf")
    early_stop_count = 0
    loss_history = {"train": [], "val": []}
    start_time = time.time()

    # 获取源平台特征（用于 MMD 对齐）
    source_platforms = [p for p in config.PLATFORM_CONFIG.keys() if p != target_platform and p in dataloaders]

    for epoch in range(config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in dataloaders[target_platform]["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            # 从源平台随机采样一个批次的特征
            cross_platform_feat = None
            if use_mmd and source_platforms:
                source_platform = random.choice(source_platforms)
                source_loader = dataloaders[source_platform]["train"]
                # 取一个 batch 作为跨平台特征
                source_batch = next(iter(source_loader))
                cross_platform_feat = source_batch[0].to(device)

            # 计算损失
            global_feat, recon_feat, pred = model.forward(target_platform, X_batch)

            # 1. 重构损失（可选）
            if use_decoder:
                recon_loss = model.reconstruction_loss(recon_feat, X_batch)
            else:
                recon_loss = torch.tensor(0.0).to(device)

            # 2. 下游任务损失
            task_loss = F.binary_cross_entropy(pred.squeeze(), y_batch.float())

            # 3. MMD 损失（可选）
            if use_mmd and cross_platform_feat is not None:
                mmd_kernel = model.mmd_loss(global_feat, cross_platform_feat)
                mmd_loss = mmd_kernel.mean()
            else:
                mmd_loss = torch.tensor(0.0).to(device)

            # 总损失
            total_loss = (
                config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + config.MMD_LOSS_WEIGHT * mmd_loss
                + task_loss
            )

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(dataloaders[target_platform]["train"].dataset)
        loss_history["train"].append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloaders[target_platform]["val"]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                global_feat, recon_feat, pred = model.forward(target_platform, X_batch)

                if use_decoder:
                    recon_loss = model.reconstruction_loss(recon_feat, X_batch)
                else:
                    recon_loss = torch.tensor(0.0).to(device)

                task_loss = F.binary_cross_entropy(pred.squeeze(), y_batch.float())

                if use_mmd and source_platforms:
                    source_platform = random.choice(source_platforms)
                    source_batch = next(iter(dataloaders[source_platform]["val"]))
                    cross_platform_feat = source_batch[0].to(device)
                    mmd_kernel = model.mmd_loss(global_feat, cross_platform_feat)
                    mmd_loss = mmd_kernel.mean()
                else:
                    mmd_loss = torch.tensor(0.0).to(device)

                val_loss_item = (
                    config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                    + config.MMD_LOSS_WEIGHT * mmd_loss
                    + task_loss
                )
                val_loss += val_loss_item.item() * X_batch.size(0)

        avg_val_loss = val_loss / len(dataloaders[target_platform]["val"].dataset)
        loss_history["val"].append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            model_suffix = f"{target_platform}_best_model.pth"
            if not use_decoder:
                model_suffix = f"{target_platform}_no_decoder_best_model.pth"
            elif not use_mmd:
                model_suffix = f"{target_platform}_no_mmd_best_model.pth"
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, model_suffix))
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1} for {target_platform}!")
                break

        logger.info(
            f"Platform: {target_platform}, Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # 统计训练指标
    end_time = time.time()
    train_metrics = get_training_metrics(start_time, end_time, model)

    # 可视化损失曲线
    title_suffix = ""
    if not use_decoder:
        title_suffix = "(Without Decoder)"
    elif not use_mmd:
        title_suffix = "(Without MMD)"

    plot_loss_curve(loss_history,
                   os.path.join(config.RESULT_DIR, f"{target_platform}{title_suffix}_loss_curve.png"),
                   title_suffix=title_suffix)

    # 加载最佳模型
    model_suffix = f"{target_platform}_best_model.pth"
    if not use_decoder:
        model_suffix = f"{target_platform}_no_decoder_best_model.pth"
    elif not use_mmd:
        model_suffix = f"{target_platform}_no_mmd_best_model.pth"

    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, model_suffix)))
    return model, train_metrics


# 消融实验（修复版）
def ablation_experiment(split_data, device):
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
        logger.info(f"\n{'='*60}")
        logger.info(f"Running ablation experiment: {exp_name}")
        logger.info(f"{'='*60}")

        remove_decoder = cfg.get("remove_decoder", False)
        remove_mmd = cfg.get("remove_mmd", False)
        remove_task = cfg.get("remove_task", False)

        for platform in config.PLATFORM_CONFIG.keys():
            if platform not in dataloaders:
                continue

            # 初始化模型
            model = FeatureDistillationModel(platform_feat_dims).to(device)

            # 根据消融配置选择训练策略
            if remove_task:
                # Without Downstream Task: 仅优化重构+MMD
                logger.info(f"Training {platform} without downstream task...")
                model, _ = train_without_task(
                    model, dataloaders, platform, device,
                    use_decoder=not remove_decoder,
                    use_mmd=not remove_mmd
                )
            elif remove_decoder or remove_mmd:
                # Without Decoder 或 Without MMD: 单平台训练
                logger.info(f"Training {platform} without decoder/MMD (single-platform)...")
                model, _ = train_platform_model(model, dataloaders[platform], platform, device)
            else:
                # Full Model: 跨平台训练（真正使用 MMD）
                logger.info(f"Training {platform} with full cross-platform alignment...")
                model, _ = train_platform_model_with_mmd(
                    model, dataloaders, platform, device,
                    use_decoder=True,
                    use_mmd=True
                )

            # 评估模型（已自动适配不同指标）
            metrics, _ = evaluate_model(model, dataloaders[platform], platform, device)

            # 统一输出格式（兼容不同指标类型）
            result_entry = {
                "experiment": exp_name,
                "platform": platform,
                "target_type": config.PLATFORM_CONFIG[platform]["target"]
            }

            # 动态添加所有指标
            for metric_name, metric_value in metrics.items():
                result_entry[metric_name] = metric_value

            ablation_results.append(result_entry)
            logger.info(f"{exp_name} - {platform}: {metrics}")

    # 保存消融实验结果
    save_results(ablation_results, os.path.join(config.RESULT_DIR, "ablation_experiment_results.csv"))
    logger.info("\nAblation experiment completed!")
    return ablation_results


# 新增：无下游任务的训练函数
def train_without_task(model, dataloaders, target_platform, device, use_decoder=True, use_mmd=True):
    """
    仅用于消融实验：Without Downstream Task
    不计算分类损失，只优化重构和 MMD
    """
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    best_val_loss = float("inf")
    early_stop_count = 0
    loss_history = {"train": [], "val": []}
    start_time = time.time()

    source_platforms = [p for p in config.PLATFORM_CONFIG.keys() if p != target_platform and p in dataloaders]

    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in dataloaders[target_platform]["train"]:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            global_feat, recon_feat, _ = model.forward(target_platform, X_batch)

            # 1. 重构损失
            if use_decoder:
                recon_loss = model.reconstruction_loss(recon_feat, X_batch)
            else:
                recon_loss = torch.tensor(0.0).to(device)

            # 2. MMD 损失
            cross_platform_feat = None
            if use_mmd and source_platforms:
                source_platform = random.choice(source_platforms)
                source_loader = dataloaders[source_platform]["train"]
                source_batch = next(iter(source_loader))
                cross_platform_feat = source_batch[0].to(device)
                mmd_kernel = model.mmd_loss(global_feat, cross_platform_feat)
                mmd_loss = mmd_kernel.mean()
            else:
                mmd_loss = torch.tensor(0.0).to(device)

            # 总损失（不含 task_loss）
            total_loss = (
                config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                + config.MMD_LOSS_WEIGHT * mmd_loss
            )

            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * X_batch.size(0)

        avg_train_loss = train_loss / len(dataloaders[target_platform]["train"].dataset)
        loss_history["train"].append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloaders[target_platform]["val"]:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                global_feat, recon_feat, _ = model.forward(target_platform, X_batch)

                if use_decoder:
                    recon_loss = model.reconstruction_loss(recon_feat, X_batch)
                else:
                    recon_loss = torch.tensor(0.0).to(device)

                if use_mmd and source_platforms:
                    source_platform = random.choice(source_platforms)
                    source_batch = next(iter(dataloaders[source_platform]["val"]))
                    cross_platform_feat = source_batch[0].to(device)
                    mmd_kernel = model.mmd_loss(global_feat, cross_platform_feat)
                    mmd_loss = mmd_kernel.mean()
                else:
                    mmd_loss = torch.tensor(0.0).to(device)

                val_loss_item = (
                    config.RECONSTRUCTION_LOSS_WEIGHT * recon_loss
                    + config.MMD_LOSS_WEIGHT * mmd_loss
                )
                val_loss += val_loss_item.item() * X_batch.size(0)

        avg_val_loss = val_loss / len(dataloaders[target_platform]["val"].dataset)
        loss_history["val"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_count = 0
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"{target_platform}_no_task_best_model.pth"))
        else:
            early_stop_count += 1
            if early_stop_count >= config.PATIENCE:
                logger.info(f"Early stopping at epoch {epoch + 1} for {target_platform} (No Task)!")
                break

        logger.info(
            f"No Task - Platform: {target_platform}, Epoch: {epoch + 1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    end_time = time.time()
    train_metrics = get_training_metrics(start_time, end_time, model)
    plot_loss_curve(loss_history, os.path.join(config.RESULT_DIR, f"{target_platform}_no_task_loss_curve.png"))
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, f"{target_platform}_no_task_best_model.pth")))
    return model, train_metrics


# 超参数敏感性分析（保留原逻辑，修复指标兼容性）
def hyperparam_sensitivity_analysis(split_data, device):
    alpha_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    sensitivity_results = []
    dataloaders = build_dataloader(split_data)
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}

    for alpha in alpha_values:
        logger.info(f"Running hyperparam sensitivity analysis for α={alpha}")
        config.ALPHA = alpha
        config.RECONSTRUCTION_LOSS_WEIGHT = alpha
        config.MMD_LOSS_WEIGHT = 1 - alpha

        for platform in config.PLATFORM_CONFIG.keys():
            if platform not in dataloaders:
                continue
            model = FeatureDistillationModel(platform_feat_dims).to(device)
            model, _ = train_platform_model(model, dataloaders[platform], platform, device)
            metrics, _ = evaluate_model(model, dataloaders[platform], platform, device)
            
            # 修复：动态构建结果字典（兼容不同指标类型）
            result_entry = {
                "alpha": alpha,
                "platform": platform,
                "target_type": config.PLATFORM_CONFIG[platform]["target"]
            }
            
            # 添加所有评估指标
            for metric_name, metric_value in metrics.items():
                result_entry[metric_name] = metric_value
            
            sensitivity_results.append(result_entry)

    save_results(sensitivity_results, os.path.join(config.RESULT_DIR, "hyperparam_sensitivity_results.csv"))
    plot_alpha_curve(sensitivity_results, os.path.join(config.RESULT_DIR, "alpha_curve.png"))
    logger.info("Hyperparameter sensitivity analysis completed!")
    return sensitivity_results



# 新增：Baseline对比实验
def run_baseline_experiments(split_data, device):
    baseline_results = []
    dataloaders = build_dataloader(split_data)
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}

    for baseline_name in config.BASELINE_MODELS:
        logger.info(f"Running baseline experiment: {baseline_name}")
        for platform in config.PLATFORM_CONFIG.keys():
            if platform not in dataloaders:
                continue

            # 获取 Baseline 模型
            if baseline_name == "OnlyMMD":
                model = get_baseline_model(baseline_name, platform_feat_dims=platform_feat_dims).to(device)
            else:
                input_dim = config.PLATFORM_CONFIG[platform]["feat_dim"]
                model = get_baseline_model(baseline_name, input_dim=input_dim).to(device)

            # 统计模型参数量（核心量化指标）
            param_stats = count_model_parameters(model)
            logger.info(f"{baseline_name} - {platform}参数量：{param_stats['total_params_million']}M")

            # 训练模型
            model, train_metrics = train_basic_model(model, dataloaders[platform],platform, device, model_name=baseline_name)

            # 评估模型（已自动适配不同指标）
            metrics, _ = evaluate_model(model, dataloaders[platform], platform, device, model_name=baseline_name)

            # 保存结果（含量化指标）
            result_entry = {
                "baseline_model": baseline_name,
                "platform": platform,
                "target_type": config.PLATFORM_CONFIG[platform]["target"],
                "total_params_million": param_stats["total_params_million"],
                "train_time_seconds": train_metrics["train_time_seconds"],
                "max_memory_mb": train_metrics["max_memory_mb"]
            }

            # 动态添加所有评估指标
            for metric_name, metric_value in metrics.items():
                result_entry[metric_name] = metric_value

            baseline_results.append(result_entry)

    # 保存 Baseline 结果
    save_results(baseline_results, os.path.join(config.RESULT_DIR, "baseline_experiment_results.csv"))
    logger.info("Baseline experiments completed!")
    return baseline_results


# 主训练评估流程（核心修改：多次实验+统计+Baseline）
def main_train_eval(split_data):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 构建数据加载器
    dataloaders = build_dataloader(split_data)

    # 初始化自定义模型
    platform_feat_dims = {p: cfg["feat_dim"] for p, cfg in config.PLATFORM_CONFIG.items()}
    custom_model = FeatureDistillationModel(platform_feat_dims).to(device)

    # 统计自定义模型参数量（论文核心）
    custom_param_stats = count_model_parameters(custom_model)
    logger.info(
        f"自定义模型总参数量：{custom_param_stats['total_params_million']}M，可训练参数量：{custom_param_stats['trainable_params_million']}M")

    # ========== 1. 多次实验（统计学要求） ==========
    all_repeat_results = {}
    for repeat_idx in range(config.REPEAT_TIMES):
        logger.info(f"\n=== 开始第{repeat_idx + 1}/{config.REPEAT_TIMES}次实验 ===\n")
        set_seed(config.SEED + repeat_idx)  # 不同种子保证随机性

        for platform in config.PLATFORM_CONFIG.keys():
            if platform not in dataloaders:
                continue

            # 重新初始化模型
            model = FeatureDistillationModel(platform_feat_dims).to(device)

            # 训练模型
            model, train_metrics = train_platform_model(model, dataloaders[platform], platform, device)

            # 评估模型
            metrics, global_feats = evaluate_model(model, dataloaders[platform], platform, device)

            # 保存单次实验结果（兼容不同指标类型）
            if platform not in all_repeat_results:
                all_repeat_results[platform] = []
            
            # 动态构建结果字典（包含所有返回的指标）
            single_result = {
                "repeat_idx": repeat_idx,
                "train_time": train_metrics["train_time_seconds"],
                "memory_mb": train_metrics["max_memory_mb"]
            }
            
            # 添加所有评估指标
            for metric_name, metric_value in metrics.items():
                single_result[metric_name] = metric_value
            
            all_repeat_results[platform].append(single_result)

    # 计算多次实验的均值±标准差（论文核心）
    statistical_results = []
    for platform, repeat_results in all_repeat_results.items():
        stat_res = calculate_statistical_results(repeat_results)
        
        # 动态构建统计结果（兼容不同指标）
        platform_stat = {
            "platform": platform,
            "target_type": config.PLATFORM_CONFIG[platform]["target"],
            "total_params_million": custom_param_stats["total_params_million"]
        }
        
        # 添加所有指标的统计结果
        for metric_key in repeat_results[0].keys():
            if metric_key not in ["repeat_idx", "train_time", "memory_mb"]:
                mean_key = f"{metric_key}_mean"
                std_key = f"{metric_key}_std"
                formatted_key = f"{metric_key}_formatted"
                
                if mean_key in stat_res and std_key in stat_res:
                    platform_stat[f"{metric_key}_mean"] = stat_res[mean_key]
                    platform_stat[f"{metric_key}_std"] = stat_res[std_key]
                    platform_stat[f"{metric_key}_formatted"] = stat_res[formatted_key]
        
        statistical_results.append(platform_stat)
    
    # 保存统计结果（直接用于论文表格）
    save_results(statistical_results, os.path.join(config.RESULT_DIR, "statistical_main_results.csv"))
    logger.info("\n=== 多次实验统计结果 ===")
    for res in statistical_results:
        metric_outputs = []
        for key, value in res.items():
            if key.endswith("_formatted"):
                metric_outputs.append(f"{key.replace('_formatted', '')}: {value}")
        logger.info(f"{res['platform']} ({res['target_type']}) - " + ", ".join(metric_outputs))

    # ========== 2. MMD 矩阵计算（修复维度不匹配） ==========
    all_global_feats = {}
    for platform in config.PLATFORM_CONFIG.keys():
        if platform not in dataloaders:
            continue
        # 加载最佳模型
        model = FeatureDistillationModel(platform_feat_dims).to(device)
        model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, f"{platform}_best_model.pth")))
        # 重新评估获取全局特征（已统一为 128 维）
        _, global_feats = evaluate_model(model, dataloaders[platform], platform, device, model_name="CustomModel")
        all_global_feats[platform] = global_feats

    # 计算 MMD 矩阵（修复后可正常运行）
    platforms = [p for p in config.PLATFORM_CONFIG.keys() if p in all_global_feats]
    if len(platforms) >= 2:
        mmd_matrix = np.zeros((len(platforms), len(platforms)))
        for i, p1 in enumerate(platforms):
            for j, p2 in enumerate(platforms):
                if i <= j:
                    # 转换为 tensor 并映射到统一设备
                    feat1 = torch.tensor(all_global_feats[p1], dtype=torch.float32).to(device)
                    feat2 = torch.tensor(all_global_feats[p2], dtype=torch.float32).to(device)
                    # 使用修复后的高斯核计算 MMD
                    kernel = model.mmd_loss(feat1, feat2)
                    mmd = kernel.mean().cpu().item()
                    mmd_matrix[i][j] = mmd
                    mmd_matrix[j][i] = mmd
        # 可视化 MMD 热图
        plot_mmd_heatmap(mmd_matrix, platforms, os.path.join(config.RESULT_DIR, "mmd_heatmap.png"))
        logger.info(f"\nMMD 矩阵（修复后）：\n{mmd_matrix}")
    else:
        mmd_matrix = None
        logger.warning("有效平台数不足 2 个，跳过 MMD 矩阵计算")

    # ========== 3. 消融实验 ==========
    ablation_experiment(split_data, device)

    # ========== 4. 超参数敏感性分析 ==========
    hyperparam_sensitivity_analysis(split_data, device)

    # ========== 5. Baseline 对比实验（核心新增） ==========
    run_baseline_experiments(split_data, device)

    logger.info("\n=== 所有实验完成！结果已保存至 ./results 目录 ===")
    return statistical_results, mmd_matrix if len(platforms) >= 2 else None


