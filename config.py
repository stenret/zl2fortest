import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 创建目录
for dir_path in [DATA_DIR, RESULT_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据配置（核心新增：分批读取参数）
DATASET_NAME = "AmazonReviews"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42  # 固定随机种子
POS_NEG_RATIO = 1.0  # 正负样本均衡比例
MIN_REVIEW_LENGTH = 10  # 最小评论长度（过滤短评论）

# 分批读取配置【核心新增】
MAX_READ_LINES = 50000  # 每个平台最多读取10万行数据（可按需调整，如5万/20万）
DATA_SAMPLE_RATIO = 0.5  # 在读取的10万行中再采样10%（最终每个平台仅用1万行）

# 平台配置（与Amazon数据集对齐）
PLATFORM_CONFIG = {
    "Books": {
        "file_name": "Books_5.json.gz",
        "feat_dim": 6,
        "target": "CTR",
        "is_long_tail": True  # 长尾平台
    },
    "Electronics": {
        "file_name": "Electronics_5.json.gz",
        "feat_dim": 7,
        "target": "CVR",
        "is_long_tail": False
    },
    "Clothing": {
        "file_name": "Clothing_Shoes_and_Jewelry_5.json.gz",
        "feat_dim": 5,
        "target": "Interaction",
        "is_long_tail": False
    }
}

# 模型配置（轻量化适配小内存）
HIDDEN_DIM = 128  # 全局编码器隐层维度
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 50  # 减少训练轮数，加快测试
BATCH_SIZE = 64  # 小内存推荐64/32
PATIENCE = 8  # 早停耐心值

# 损失函数配置（与论文一致）
ALPHA = 0.7  # 重构损失权重
MMD_SIGMA = 1.0  # 高斯核参数
RECONSTRUCTION_LOSS_WEIGHT = ALPHA
MMD_LOSS_WEIGHT = 1 - ALPHA

# 评估配置
METRICS = ["AUC", "F1-score"]
REPEAT_TIMES = 3  # 减少重复实验次数，加快测试