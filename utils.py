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


# 计算评估指标
def calculate_metrics(y_true, y_pred_prob):
    if len(np.unique(y_pred_prob)) == 1:
        auc = 0.5
        f1 = 0.0
    else:
        auc = roc_auc_score(y_true, y_pred_prob)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)
    return {"AUC": auc, "F1-score": f1}


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

# 生成超参数 α 性能曲线（论文专用）
def plot_alpha_curve(hyper_results, save_path):
    import pandas as pd
    df = pd.DataFrame(hyper_results)

    alphas = sorted(df["alpha"].unique())
    platforms = df["platform"].unique()

    plt.style.use('default')
    plt.figure(figsize=(7, 4))

    for platform in platforms:
        sub = df[df["platform"] == platform]
        sub = sub.sort_values("alpha")
        auc = sub["AUC"].values
        plt.plot(sub["alpha"], auc, marker="o", linewidth=2, label=platform)

    plt.xlabel("α (Reconstruction Loss Weight)", fontsize=11)
    plt.ylabel("AUC", fontsize=11)
    plt.title("Hyperparameter Sensitivity (α vs AUC)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
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


# 新增：计算多次实验的均值±标准差（论文统计学要求）
def calculate_statistical_results(repeat_results):
    """
    repeat_results: 列表，每个元素是单次实验的metrics字典
    返回：均值±标准差的字典
    """
    metrics = repeat_results[0].keys() if repeat_results else []
    stat_results = {}

    for metric in metrics:
        values = [r[metric] for r in repeat_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        # 格式：均值±标准差（保留4位小数）
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