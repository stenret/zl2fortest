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

# 数据配置
DATASET_NAME = "AmazonReviews"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42  # 固定随机种子
POS_NEG_RATIO = 1.0  # 正负样本均衡比例
MIN_REVIEW_LENGTH = 10  # 最小评论长度

# 分批读取配置（建议增加数据量）
MAX_READ_LINES = 100000  # 增加到 10 万行
DATA_SAMPLE_RATIO = 1.0  # 全量数据

# 平台配置
PLATFORM_CONFIG = {
    "Books": {
        "file_name": "Books_5.json.gz",
        "feat_dim": 6,
        "target": "CTR",
        "is_long_tail": True,
        "sample":0.6
    },
    "Electronics": {
        "file_name": "Electronics_5.json.gz",
        "feat_dim": 7,
        "target": "CVR_Classification",  # 修改为分类版本
        "is_long_tail": False,
        "sample":1
    },
    "Clothing": {
        "file_name": "Clothing_Shoes_and_Jewelry_5.json.gz",
        "feat_dim": 5,
        "target": "Interaction",
        "is_long_tail": False,
        "sample":0.8
    }
}

# 模型配置（轻量化适配）
HIDDEN_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
PATIENCE = 8

# 损失函数配置
ALPHA = 0.7
MMD_SIGMA = 1.0
RECONSTRUCTION_LOSS_WEIGHT = ALPHA
MMD_LOSS_WEIGHT = 1 - ALPHA

# 新增：针对不同平台的差异化权重（适配消融实验）
PLATFORM_LOSS_WEIGHTS = {
    "Books": 0.5,      # 长尾数据降低重构权重
    "Electronics": 0.7,
    "Clothing": 0.6
}

# 评估配置（核心修改：增加重复实验次数）
METRICS = ["AUC", "F1-score"]
REPEAT_TIMES = 5  # 5 次重复实验（论文要求均值±标准差）

# Baseline 配置（新增）
BASELINE_MODELS = [
    "RawFeature",       # 基础模型：仅原始特征训练
    "OnlyMMD",          # 仅 MMD 损失（无重构）
    "LightGCN_Baseline" # 轻量化 SOTA 对比
]

# ========== 联邦蒸馏配置（新增） ==========
USE_FEDERATED_DISTILL = False  # 是否启用联邦蒸馏模式
FEDERATED_ROUNDS = 15          # 联邦训练轮数
LOCAL_EPOCHS = 3               # 每轮本地训练 epoch 数
DISTILL_TEMPERATURE = 3.0      # 蒸馏温度（软化概率分布，值越大分布越平滑）
DISTILL_ALPHA = 0.7            # 蒸馏损失权重（0.5-0.9 推荐）
PRIVACY_NOISE_STD = 0.0        # 添加到上传参数的噪声标准差（差分隐私）