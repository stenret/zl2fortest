import config
from utils import setup_logger
from data_process import build_heterogeneous_scene
from train_eval import main_train_eval

# 初始化日志
logger = setup_logger()


def main():
    logger.info("=" * 60)
    logger.info("Starting feature distillation experiment (Paper Version)")
    logger.info("=" * 60)

    # 构建异构场景数据
    split_data, user_id_map = build_heterogeneous_scene()
    if not split_data:
        logger.error("无有效数据，程序终止")
        return

    # 主训练评估流程（含多次实验+Baseline+统计）
    statistical_results, mmd_matrix = main_train_eval(split_data)

    # 打印最终统计结果（论文核心）
    logger.info("\n" + "=" * 60)
    logger.info("Final Statistical Results (Mean±Std)")
    logger.info("=" * 60)
    for res in statistical_results:
        logger.info(f"\nPlatform: {res['platform']} ({res['target_type']})")
        logger.info(f"  - Model Params: {res['total_params_million']}M")
        
        # 动态输出所有指标的均值±标准差
        for key, value in res.items():
            if key.endswith("_formatted") and key not in ["target_type_formatted"]:
                metric_name = key.replace("_formatted", "")
                logger.info(f"  - {metric_name}: {value}")

    if mmd_matrix is not None:
        logger.info(f"\nMMD Matrix:\n{mmd_matrix}")

    logger.info("\n" + "=" * 60)
    logger.info("Distribution Heterogeneity Summary")
    logger.info("=" * 60)
    for platform, cfg in config.PLATFORM_CONFIG.items():
        logger.info(f"\n{platform}:")
        logger.info(f"  - Feature Dimension: {cfg['feat_dim']}")
        logger.info(f"  - Target Type: {cfg['target']}")
        logger.info(f"  - Is Long-tail: {cfg.get('is_long_tail', False)}")
        logger.info(f"  - Sample Ratio: {cfg.get('sample', 1.0)}")

    logger.info("\nAll experiments finished! Results saved in ./results directory.")


if __name__ == "__main__":
    main()