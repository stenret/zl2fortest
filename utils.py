import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.metrics import roc_auc_score, f1_score
import config


# 日志配置
def setup_logger():
    logger = logging.getLogger("FeatureDistillation")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "train.log"), encoding="utf-8")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算评估指标（支持多任务类型）
def calculate_metrics(y_true, y_pred_prob, task_type="classification"):
    """
    根据不同任务类型计算评估指标
    :param y_true: 真实标签
    :param y_pred_prob: 预测概率
    :param task_type: 任务类型 (classification/regression/ranking)
    :return: 指标字典
    """
    metrics = {}
    
    if task_type == "classification":
        # 分类任务：AUC + F1
        if len(np.unique(y_pred_prob)) == 1:
            auc = 0.5
            f1 = 0.0
        else:
            auc = roc_auc_score(y_true, y_pred_prob)
            y_pred = (y_pred_prob >= 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        metrics = {"AUC": auc, "F1-score": f1}
        
    elif task_type == "regression":
        # 回归任务：预测具体评分（1-5 分）
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        # 将概率映射回 1-5 分范围
        y_pred_rating = np.clip(y_pred_prob * 4 + 1, 1, 5)
        mse = mean_squared_error(y_true, y_pred_rating)
        mae = mean_absolute_error(y_true, y_pred_rating)
        r2 = r2_score(y_true, y_pred_rating)
        metrics = {"MSE": round(mse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}
    
    elif task_type == "ranking":
        # 排序任务：NDCG 指标
        try:
            from sklearn.metrics import ndcg_score
            # 需要至少 2 个样本
            if len(y_true) >= 2:
                # 修复 1：使用原始二值标签而不是重新阈值化
                y_true_rel = y_true.astype(int).reshape(1, -1)
                y_pred_rank = y_pred_prob.reshape(1, -1)
                
                # 修复 2：检查是否有正样本
                if np.sum(y_true_rel) > 0:
                    ndcg = ndcg_score(y_true_rel, y_pred_rank, k=min(10, len(y_true)))
                    metrics = {"NDCG": round(ndcg, 4)}
                else:
                    # 没有正样本，NDCG 无意义
                    print(f"警告：Ranking 任务中没有正样本，NDCG=0.0")
                    metrics = {"NDCG": 0.0}
            else:
                metrics = {"NDCG": 0.0}
        except Exception as e:
            # 修复：使用 print 代替 logger（因为此函数内无法访问 logger）
            print(f"警告：NDCG 计算失败：{e}")
            metrics = {"NDCG": 0.0}
    
    return metrics


# 可视化损失曲线
def plot_loss_curve(loss_history, save_path, title_suffix=""):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["train"], label="Train Loss")
    plt.plot(loss_history["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss Curve {title_suffix}")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# 可视化MMD热图
def plot_mmd_heatmap(mmd_matrix, platforms, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(mmd_matrix, annot=True, fmt=".6f", cmap="Blues",
                xticklabels=platforms, yticklabels=platforms)
    plt.title("MMD Distance Between Platforms (Lower = Better Alignment)")
    plt.xlabel("Platform")
    plt.ylabel("Platform")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# 生成超参数 α 性能曲线（论文专用，修复多指标兼容）
def plot_alpha_curve(hyper_results, save_path):
    import pandas as pd
    df = pd.DataFrame(hyper_results)

    alphas = sorted(df["alpha"].unique())
    platforms = df["platform"].unique()

    plt.style.use('default')
    
    # 检测有哪些指标列
    metric_cols = [col for col in df.columns if col not in ["alpha", "platform", "target_type"]]
    
    # 如果没有找到指标，使用默认的 AUC
    if not metric_cols:
        metric_cols = ["AUC"]
    
    # 选择第一个主要指标（优先级：AUC > NDCG > 其他）
    primary_metric = "AUC" if "AUC" in metric_cols else metric_cols[0]
    
    # 创建子图（每个平台一个子图）
    n_platforms = len(platforms)
    fig, axes = plt.subplots(1, n_platforms, figsize=(5 * n_platforms, 4))
    
    if n_platforms == 1:
        axes = [axes]
    
    for idx, platform in enumerate(platforms):
        ax = axes[idx]
        sub = df[df["platform"] == platform].sort_values("alpha")
        
        if primary_metric in sub.columns:
            metric_values = sub[primary_metric].values
            ax.plot(sub["alpha"], metric_values, marker="o", linewidth=2, label=platform)
            ax.set_xlabel("α (Reconstruction Loss Weight)", fontsize=11)
            ax.set_ylabel(primary_metric, fontsize=11)
            ax.set_title(f"{platform}: α vs {primary_metric}", fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"No {primary_metric} data", ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# 保存实验结果
def save_results(results, save_path):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8")
    return df


# 文本特征提取
def extract_text_features(text):
    if not text or len(text) < config.MIN_REVIEW_LENGTH:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    len_feat = len(text) / 1000.0
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    punctuations = [',', '.', '!', '?', ';', ':']
    punct_ratio = sum(1 for c in text if c in punctuations) / len(text)
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    positive_words = ["good", "great", "excellent", "perfect", "best", "love", "like"]
    pos_count = sum(1 for w in words if w.lower() in positive_words) / len(words) if words else 0.0
    neg_words = ["not", "no", "never", "none"]
    neg_count = sum(1 for w in words if w.lower() in neg_words) / len(words) if words else 0.0

    features = [
        min(len_feat, 1.0),
        min(avg_word_len / 10.0, 1.0),
        min(punct_ratio, 1.0),
        min(upper_ratio, 1.0),
        min(digit_ratio, 1.0),
        min(pos_count, 1.0),
        min(neg_count, 1.0)
    ]
    return features


# 新增：统计模型参数量（论文核心量化指标）
def count_model_parameters(model):
    """计算模型参数量，返回总参数量和可训练参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_million": round(total_params / 1e6, 3),
        "trainable_params_million": round(trainable_params / 1e6, 3)
    }


# 新增：计算多次实验的均值 ± 标准差（论文统计学要求）
def calculate_statistical_results(repeat_results):
    """
    repeat_results: 列表，每个元素是单次实验的 metrics 字典
    返回：均值±标准差的字典
    """
    if not repeat_results:
        return {}
    
    # 获取所有键，排除非数值类型
    metrics = []
    for key in repeat_results[0].keys():
        if key in ["repeat_idx", "model_type", "platform", "target_type"]:
            continue
        # 检查是否为数值类型
        sample_val = repeat_results[0][key]
        if isinstance(sample_val, (int, float)):
            metrics.append(key)
    
    stat_results = {}
    
    for metric in metrics:
        values = [r[metric] for r in repeat_results if isinstance(r.get(metric), (int, float))]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            # 格式：均值 ± 标准差（保留 4 位小数）
            stat_results[f"{metric}_mean"] = round(mean_val, 4)
            stat_results[f"{metric}_std"] = round(std_val, 4)
            stat_results[f"{metric}_formatted"] = f"{mean_val:.4f}±{std_val:.4f}"
    
    return stat_results


# 新增：统计训练时间和显存（轻量化量化指标）
def get_training_metrics(start_time, end_time, model):
    """返回训练耗时和显存占用"""
    train_time = end_time - start_time
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_used = 0.0

    return {
        "train_time_seconds": round(train_time, 2),
        "max_memory_mb": round(memory_used, 2)
    }