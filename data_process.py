import pandas as pd
import numpy as np
import os
import gzip
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import config
from utils import setup_logger, set_seed, extract_text_features

logger = setup_logger()
set_seed(config.SEED)


# 分批读取JSON.gz文件
def batch_read_gzip(file_path, max_lines=100000):
    data = []
    line_count = 0
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line_count >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    review = json.loads(line)
                    data.append(review)
                    line_count += 1
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    except EOFError as e:
        logger.warning(f"文件读取到{line_count}行时遇到EOF错误：{e}")
    except Exception as e:
        logger.error(f"读取文件失败：{e}")
        return []
    logger.info(f"从{os.path.basename(file_path)}读取到 {len(data)} 条有效记录（指定最大行数：{max_lines}）")
    return data


# 加载Amazon数据
def load_amazon_data():
    platform_data = {}
    for platform, cfg in config.PLATFORM_CONFIG.items():
        file_path = os.path.join(config.DATA_DIR, cfg["file_name"])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在：{file_path}")

        # 分批读取
        raw_records = batch_read_gzip(file_path, max_lines=config.MAX_READ_LINES)
        if not raw_records:
            logger.error(f"{platform}平台无有效数据，跳过")
            continue

        # 提取核心字段
        processed_records = []
        for review in raw_records:
            record = {
                "reviewerID": review.get("reviewerID", ""),
                "asin": review.get("asin", ""),
                "overall": float(review.get("overall", 0.0)),
                "reviewText": review.get("reviewText", ""),
                "helpful": review.get("helpful", [0, 0])
            }
            processed_records.append(record)

        # 转换为DataFrame并过滤
        df = pd.DataFrame(processed_records)
        df = df[df["reviewerID"] != ""].reset_index(drop=True)
        df = df[df["overall"] > 0].reset_index(drop=True)
        df = df[df["reviewText"].str.len() >= 1].reset_index(drop=True)

        # 采样
        if config.DATA_SAMPLE_RATIO < 1.0 and len(df) > 0:
            original_size = len(df)
            sample_ratio=cfg.get("sample")
            df = df.sample(frac=(config.DATA_SAMPLE_RATIO*sample_ratio), random_state=config.SEED).reset_index(drop=True)
            logger.info(f"{platform}平台数据采样：{original_size}条 → {len(df)}条（采样比例{config.DATA_SAMPLE_RATIO*sample_ratio}）")

        if len(df) == 0:
            logger.error(f"{platform}平台无有效数据，跳过")
            continue
        logger.info(f"{platform}平台最终加载：{len(df)}条有效数据")
        platform_data[platform] = df
    return platform_data


# 数据预处理
def preprocess_data(raw_data):
    processed_data = {}
    scaler = MinMaxScaler()

    for platform, df in raw_data.items():
        # 基础过滤
        df = df.dropna(subset=["reviewerID", "reviewText", "overall"])
        df = df[df["reviewText"].str.len() >= config.MIN_REVIEW_LENGTH].reset_index(drop=True)
        if len(df) < 10:
            logger.warning(f"{platform}平台有效样本不足 10 条，跳过")
            continue

        # ========== 新增：根据不同平台构建不同的目标变量（实现目标异构） ==========
        target_type = config.PLATFORM_CONFIG[platform]["target"]
        
        if target_type == "CTR":
            # Books 平台：CTR - 模拟点击行为
            # 假设：长评论 + 高分 = 深度点击/兴趣
            # 策略 1：评论长度 > 200 字符且评分>=4 为正样本
            df["target"] = ((df["reviewText"].str.len() > 200) & (df["overall"] >= 4.0)).astype(int)
            logger.info(f"{platform}平台使用 CTR 目标：长评论 (>200) 且高分 (≥4) → 正样本")
            
        elif target_type == "CVR" or target_type == "CVR_Classification":
            # Electronics 平台：CVR - 模拟转化行为
            # 策略：仅 5 星评为正样本（严格转化标准）
            df["target"] = (df["overall"] >= 5.0).astype(int)
            logger.info(f"{platform}平台使用 CVR 目标：仅 5 星评→正样本（严格转化）")

        elif target_type == "Interaction":
            # Clothing 平台：Interaction - 模拟交互行为
            # 策略：helpful 投票数>0 或评分>=4 为正样本
            df["helpful_votes"] = df["helpful"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
            df["target"] = ((df["helpful_votes"] > 0) | (df["overall"] >= 4.0)).astype(int)
            
            # ========== 新增：调试信息 ==========
            pos_count = ((df["helpful_votes"] > 0).sum(), (df["overall"] >= 4.0).sum())
            total_pos = df["target"].sum()
            logger.info(f"{platform}平台使用 Interaction 目标：有帮助投票或高分 (≥4) → 正样本")
            logger.info(f"  - Helpful 投票>0 的样本数：{pos_count[0]}")
            logger.info(f"  - 评分≥4 的样本数：{pos_count[1]}")
            logger.info(f"  - 总正样本数：{total_pos}, 正样本比例：{total_pos/len(df):.2%}")
            
        else:
            # 默认策略：使用原始评分阈值
            df = df[df["overall"].isin([1.0, 2.0, 4.0, 5.0])].reset_index(drop=True)
            df["target"] = (df["overall"] >= 4.0).astype(int)
            logger.info(f"{platform}平台使用默认目标：评分≥4→正样本")

        # 过滤无效目标
        if len(df) == 0:
            logger.warning(f"{platform}平台无有效正负样本，跳过")
            continue

        # 特征工程
        df["text_features"] = df["reviewText"].apply(extract_text_features)
        feat_cols = ["len_feat", "avg_word_len", "punct_ratio", "upper_ratio", "digit_ratio", "pos_count", "neg_count"]
        df[feat_cols] = pd.DataFrame(df["text_features"].tolist(), index=df.index)

        # 特征筛选
        feat_dim = config.PLATFORM_CONFIG[platform]["feat_dim"]
        selected_feats = feat_cols[:feat_dim]

        # 标准化
        df[selected_feats] = scaler.fit_transform(df[selected_feats])

        # 正负样本均衡
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

        # 添加高斯噪声
        noise = np.random.normal(0, 0.01, size=(df_balanced.shape[0], len(selected_feats)))
        df_balanced[selected_feats] = df_balanced[selected_feats] + noise
        df_balanced[selected_feats] = df_balanced[selected_feats].clip(0, 1)

        # 保存处理后的数据
        processed_data[platform] = {
            "features": df_balanced[selected_feats].values,
            "target": df_balanced["target"].values,
            "user_id": df_balanced["reviewerID"].values
        }
        
        # 如果有额外目标类型，也保存
        if "target_regression" in df_balanced.columns:
            processed_data[platform]["target_regression"] = df_balanced["target_regression"].values
        if "target_ranking" in df_balanced.columns:
            processed_data[platform]["target_ranking"] = df_balanced["target_ranking"].values
        
        logger.info(
            f"{platform}平台预处理完成：特征维度{len(selected_feats)}，样本量{len(df_balanced)}，正样本{sum(processed_data[platform]['target'])}，正样本比例={sum(processed_data[platform]['target'])/len(processed_data[platform]['target']):.2%}")

    return processed_data


# 划分训练/验证/测试集
def split_dataset(processed_data):
    split_data = {}
    for platform, data in processed_data.items():
        X = data["features"]
        y = data["target"]
        if len(X) < 20:
            logger.warning(f"{platform}平台样本量过小，训练/验证/测试集复用")
            split_data[platform] = {
                "train": (X, y),
                "val": (X, y),
                "test": (X, y)
            }
            continue

        # 分层划分
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
        logger.info(f"{platform}平台划分完成：训练集{X_train.shape}，验证集{X_val.shape}，测试集{X_test.shape}")

    return split_data


# 构建异构场景
def build_heterogeneous_scene():
    raw_data = load_amazon_data()
    if not raw_data:
        logger.error("所有平台均无有效数据，程序终止")
        return {}, {}

    processed_data = preprocess_data(raw_data)
    if not processed_data:
        logger.error("数据预处理后无有效数据，程序终止")
        return {}, {}

    split_data = split_dataset(processed_data)
    if not split_data:
        logger.error("数据集划分后无有效数据，程序终止")
        return {}, {}

    # 统一用户ID
    all_user_ids = []
    for platform in split_data.keys():
        all_user_ids.extend(processed_data[platform]["user_id"])
    unique_user_ids = list(set(all_user_ids))
    user_id_map = {uid: i for i, uid in enumerate(unique_user_ids)}
    logger.info(f"跨平台用户ID统一完成：总唯一用户数{len(unique_user_ids)}")

    return split_data, user_id_map


if __name__ == "__main__":
    split_data, user_id_map = build_heterogeneous_scene()
    logger.info("数据处理模块测试完成！")