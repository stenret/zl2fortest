import config
from utils import setup_logger
from data_process import build_heterogeneous_scene
from train_eval import main_train_eval

# 初始化日志
logger = setup_logger()

def main():
    logger.info("="*50)
    logger.info("Starting feature distillation experiment for cross-platform recommendation...")
    logger.info("="*50)
    # 构建异构场景数据
    split_data, user_id_map = build_heterogeneous_scene()
    # 主训练评估流程
    all_metrics, mmd_matrix = main_train_eval(split_data)
    # 打印结果
    logger.info("\n" + "="*50)
    logger.info("Main experiment results:")
    logger.info("="*50)
    for metric in all_metrics:
        logger.info(f"Platform: {metric['platform']}")
        logger.info(f"  - AUC: {metric['AUC']:.4f}")
        logger.info(f"  - F1-score: {metric['F1-score']:.4f}")
    if mmd_matrix is not None:
        logger.info("\nMMD matrix between platforms:")
        logger.info(mmd_matrix)
    logger.info("\nAll experiments finished! Results saved in ./results directory.")

if __name__ == "__main__":
    main()