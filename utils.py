import logging
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, f1_score
import config


# 日志配置
def setup_logger():
    logger = logging.getLogger("FeatureDistillation")
    logger.setLevel(logging.INFO)
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    # 文件处理器
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # 格式
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 添加处理器
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
    # 处理极端情况（所有预测值相同）
    if len(np.unique(y_pred_prob)) == 1:
        auc = 0.5
        f1 = 0.0
    else:
        # AUC
        auc = roc_auc_score(y_true, y_pred_prob)
        # F1-score（取0.5为阈值）
        y_pred = (y_pred_prob >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred)
    return {"AUC": auc, "F1-score": f1}


# 可视化损失曲线
def plot_loss_curve(loss_history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history["train"], label="Train Loss")
    plt.plot(loss_history["val"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# 可视化MMD热图
def plot_mmd_heatmap(mmd_matrix, platforms, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(mmd_matrix, annot=True, fmt=".6f", cmap="Blues",
                xticklabels=platforms, yticklabels=platforms)
    plt.title("MMD Distance Between Platforms")
    plt.xlabel("Platform")
    plt.ylabel("Platform")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# 保存实验结果
def save_results(results, save_path):
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False, encoding="utf-8")
    return df


# 文本特征提取（用于Amazon评论数据）
def extract_text_features(text):
    """提取文本的基础统计特征"""
    if not text or len(text) < config.MIN_REVIEW_LENGTH:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # 特征1：评论长度
    len_feat = len(text) / 1000.0
    # 特征2：平均单词长度
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    # 特征3：标点符号占比
    punctuations = [',', '.', '!', '?', ';', ':']
    punct_ratio = sum(1 for c in text if c in punctuations) / len(text)
    # 特征4：大写字母占比
    upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
    # 特征5：数字占比
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text)
    # 特征6：情感倾向（简单版：正面词汇数）
    positive_words = ["good", "great", "excellent", "perfect", "best", "love", "like"]
    pos_count = sum(1 for w in words if w.lower() in positive_words) / len(words) if words else 0.0
    # 特征7：否定词占比
    neg_words = ["not", "no", "never", "none"]
    neg_count = sum(1 for w in words if w.lower() in neg_words) / len(words) if words else 0.0

    # 归一化到[0,1]
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


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Utils module initialized successfully!")