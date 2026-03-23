import pandas as pd
import numpy as np
import os
import gzip
import json
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config
from utils import setup_logger, set_seed, extract_text_features

logger = setup_logger()
set_seed(config.SEED)


# 核心优化：分批读取JSON.gz文件（仅读取指定行数，不加载全量）
def batch_read_gzip(file_path, max_lines=100000):
    """
    分批读取gzip文件，仅读取前max_lines行
    :param file_path: 文件路径
    :param max_lines: 最大读取行数
    :return: 读取到的有效JSON记录列表
    """
    data = []
    line_count = 0

    try:
        # 流式逐行读取，控制内存占用
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 达到最大行数则停止读取
                if line_count >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue

                try:
                    # 解析JSON
                    review = json.loads(line)
                    data.append(review)
                    line_count += 1
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    except EOFError as e:
        logger.warning(f"文件读取到{line_count}行时遇到EOF错误，已读取的数据仍可用：{e}")
    except Exception as e:
        logger.error(f"读取文件失败：{e}")
        return []

    logger.info(f"从{os.path.basename(file_path)}读取到 {len(data)} 条有效记录（指定最大行数：{max_lines}）")
    return data


# 加载Amazon Reviews数据（分批读取+采样）
def load_amazon_data():
    platform_data = {}
    for platform, cfg in config.PLATFORM_CONFIG.items():
        file_path = os.path.join(config.DATA_DIR, cfg["file_name"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在：{file_path}\n请确认放入data目录下")

        # 1. 分批读取指定行数的数据（核心优化）
        raw_records = batch_read_gzip(file_path, max_lines=config.MAX_READ_LINES)
        if not raw_records:
            logger.error(f"{platform}平台无有效数据，跳过该平台")
            continue

        # 2. 提取核心字段
        processed_records = []
        for review in raw_records:
            record = {
                "reviewerID": review.get("reviewerID", ""),  # 用户ID
                "asin": review.get("asin", ""),  # 商品ID
                "overall": float(review.get("overall", 0.0)),  # 评分（1-5）
                "reviewText": review.get("reviewText", ""),  # 评论文本
                "helpful": review.get("helpful", [0, 0])  # 有用性投票
            }
            processed_records.append(record)

        # 3. 转换为DataFrame并过滤无效数据
        df = pd.DataFrame(processed_records)
        df = df[df["reviewerID"] != ""].reset_index(drop=True)  # 过滤空用户ID
        df = df[df["overall"] > 0].reset_index(drop=True)  # 过滤无效评分
        df = df[df["reviewText"].str.len() >= 1].reset_index(drop=True)  # 过滤空评论

        # 4. 采样（在读取的部分数据上进一步采样）
        if config.DATA_SAMPLE_RATIO < 1.0 and len(df) > 0:
            original_size = len(df)
            df = df.sample(frac=config.DATA_SAMPLE_RATIO, random_state=config.SEED).reset_index(drop=True)
            logger.info(f"{platform}平台数据采样：{original_size}条 → {len(df)}条（采样比例{config.DATA_SAMPLE_RATIO}）")

        if len(df) == 0:
            logger.error(f"{platform}平台无有效数据，跳过")
            continue
        logger.info(f"{platform}平台最终加载：{len(df)}条有效数据（仅读取全量的{config.MAX_READ_LINES / 1000000}M行）")
        platform_data[platform] = df
    return platform_data


# 数据预处理（保持原有逻辑）
def preprocess_data(raw_data):
    processed_data = {}
    scaler = MinMaxScaler()

    for platform, df in raw_data.items():
        # 1. 基础过滤（非空+最短评论长度）
        df = df.dropna(subset=["reviewerID", "reviewText", "overall"])
        df = df[df["reviewText"].str.len() >= config.MIN_REVIEW_LENGTH].reset_index(drop=True)
        if len(df) < 10:  # 过滤样本量过小的情况
            logger.warning(f"{platform}平台有效样本不足10条，跳过")
            continue

        # 2. 构建目标变量（评分>=4正样本1，<=2负样本0，过滤3分）
        df = df[df["overall"].isin([1.0, 2.0, 4.0, 5.0])].reset_index(drop=True)
        if len(df) == 0:
            logger.warning(f"{platform}平台无有效正负样本（仅3分），跳过")
            continue
        df["target"] = (df["overall"] >= 4.0).astype(int)

        # 3. 特征工程：提取7维文本统计特征
        df["text_features"] = df["reviewText"].apply(extract_text_features)
        feat_cols = ["len_feat", "avg_word_len", "punct_ratio", "upper_ratio", "digit_ratio", "pos_count", "neg_count"]
        df[feat_cols] = pd.DataFrame(df["text_features"].tolist(), index=df.index)

        # 4. 按平台配置筛选特征维度
        feat_dim = config.PLATFORM_CONFIG[platform]["feat_dim"]
        selected_feats = feat_cols[:feat_dim]

        # 5. 特征标准化（MinMax归一化到[0,1]）
        df[selected_feats] = scaler.fit_transform(df[selected_feats])

        # 6. 正负样本均衡（避免数据倾斜）
        pos_df = df[df["target"] == 1]
        neg_df = df[df["target"] == 0]
        if len(pos_df) == 0 or len(neg_df) == 0:
            logger.warning(f"{platform}平台仅单类样本，跳过均衡")
            df_balanced = df
        else:
            sample_size = min(len(pos_df), len(neg_df))
            pos_df = pos_df.sample(n=sample_size, random_state=config.SEED)
            neg_df = neg_df.sample(n=sample_size, random_state=config.SEED)
            df_balanced = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=config.SEED).reset_index(drop=True)

        # 7. 添加高斯噪声（避免模型过拟合/完美分类）
        noise = np.random.normal(0, 0.01, size=(df_balanced.shape[0], len(selected_feats)))
        df_balanced[selected_feats] = df_balanced[selected_feats] + noise
        df_balanced[selected_feats] = df_balanced[selected_feats].clip(0, 1)  # 限制在[0,1]

        # 8. 保存处理后的数据
        processed_data[platform] = {
            "features": df_balanced[selected_feats].values,
            "target": df_balanced["target"].values,
            "user_id": df_balanced["reviewerID"].values
        }
        logger.info(f"{platform}平台预处理完成：")
        logger.info(f"  - 特征维度：{len(selected_feats)} | 样本量：{len(df_balanced)}")
        logger.info(
            f"  - 正样本：{sum(processed_data[platform]['target'])} | 负样本：{len(processed_data[platform]['target']) - sum(processed_data[platform]['target'])}")

    return processed_data


# 划分训练/验证/测试集（适配小样本）
def split_dataset(processed_data):
    split_data = {}
    for platform, data in processed_data.items():
        X = data["features"]
        y = data["target"]
        if len(X) < 20:  # 样本量不足20条，不划分直接使用
            logger.warning(f"{platform}平台样本量过小，训练/验证/测试集复用")
            split_data[platform] = {
                "train": (X, y),
                "val": (X, y),
                "test": (X, y)
            }
            continue

        # 分层划分（保证正负样本比例）
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - config.TRAIN_RATIO, random_state=config.SEED, stratify=y
        )
        val_size = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=1 - val_size, random_state=config.SEED, stratify=y_temp
        )

        split_data[platform] = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)
        }
        logger.info(f"{platform}平台数据集划分：")
        logger.info(f"  - 训练集：{X_train.shape} | 验证集：{X_val.shape} | 测试集：{X_test.shape}")

    return split_data


# 构建四重异构场景（主入口）
def build_heterogeneous_scene():
    # 1. 加载原始数据（分批读取+采样）
    raw_data = load_amazon_data()
    if not raw_data:
        logger.error("所有平台均无有效数据，程序终止")
        return {}, {}

    # 2. 预处理数据
    processed_data = preprocess_data(raw_data)
    if not processed_data:
        logger.error("数据预处理后无有效数据，程序终止")
        return {}, {}

    # 3. 划分训练/验证/测试集
    split_data = split_dataset(processed_data)
    if not split_data:
        logger.error("数据集划分后无有效数据，程序终止")
        return {}, {}

    # 4. 统一跨平台用户ID
    all_user_ids = []
    for platform in split_data.keys():
        all_user_ids.extend(processed_data[platform]["user_id"])
    unique_user_ids = list(set(all_user_ids))
    user_id_map = {uid: i for i, uid in enumerate(unique_user_ids)}
    logger.info(f"跨平台用户ID统一完成：总唯一用户数 = {len(unique_user_ids)}")

    return split_data, user_id_map


if __name__ == "__main__":
    # 测试数据处理流程
    split_data, user_id_map = build_heterogeneous_scene()
    logger.info("数据处理模块测试完成！")