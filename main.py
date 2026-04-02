import config
from utils import setup_logger
from data_process import build_heterogeneous_scene
from train_eval import main_train_eval
from federated_trainer import FederatedDistillationTrainer, evaluate_federated_model

# 初始化日志
logger = setup_logger()


def main():
    logger.info("=" * 60)
    
    if config.USE_FEDERATED_DISTILL:
        logger.info("Starting Federated Distillation Experiment")
    else:
        logger.info("Starting Feature Distillation Experiment (Paper Version)")
    
    logger.info("=" * 60)

    # 构建异构场景数据
    split_data, user_id_map = build_heterogeneous_scene()
    if not split_data:
        logger.error("无有效数据，程序终止")
        return

    # ========== 选择训练模式 ==========
    if config.USE_FEDERATED_DISTILL:
        # 联邦蒸馏模式
        statistical_results = run_federated_training(split_data)
        mmd_matrix = None  # 联邦模式下暂不计算 MMD 矩阵
    else:
        # 原有集中式训练模式
        statistical_results, mmd_matrix = main_train_eval(split_data)

    # 打印最终统计结果
    logger.info("\n" + "=" * 60)
    logger.info("Final Results Summary")
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

    logger.info("\n" + "=" * 60)
    if config.USE_FEDERATED_DISTILL:
        logger.info("Federated distillation experiment finished!")
    else:
        logger.info("All experiments finished! Results saved in ./results directory.")
    logger.info("=" * 60)


def run_federated_training(split_data):
    """
    运行联邦蒸馏训练流程
    """
    from train_eval import build_dataloader
    from utils import count_model_parameters, calculate_statistical_results, save_results, set_seed
    import numpy as np
    import torch
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 构建数据加载器
    dataloaders = build_dataloader(split_data)
    
    # 初始化联邦蒸馏训练器
    trainer = FederatedDistillationTrainer(
        platform_configs=config.PLATFORM_CONFIG,
        device=device
    )
    
    # 统计全局模型参数量
    global_param_stats = count_model_parameters(trainer.global_model)
    logger.info(
        f"全局模型总参数量：{global_param_stats['total_params_million']}M，"
        f"可训练参数量：{global_param_stats['trainable_params_million']}M"
    )
    
    # ========== 多轮联邦训练 ==========
    all_repeat_results = {}
    
    for repeat_idx in range(config.REPEAT_TIMES):
        logger.info(f"\n=== 开始第{repeat_idx + 1}/{config.REPEAT_TIMES}次联邦实验 ===")
        set_seed(config.SEED + repeat_idx)
        
        # 重新初始化训练器（每次重复用新模型）
        trainer = FederatedDistillationTrainer(
            platform_configs=config.PLATFORM_CONFIG,
            device=device
        )
        
        # 多轮联邦训练
        for round_idx in range(config.FEDERATED_ROUNDS):
            trainer.train_federation_round(
                round_idx=round_idx,
                dataloaders=dataloaders,
                local_epochs=config.LOCAL_EPOCHS
            )
        
        # 保存联邦模型
        trainer.save_all_models(
            os.path.join(config.MODEL_DIR, f"fed_repeat_{repeat_idx+1}")
        )
        
        # 评估各平台模型
        for platform in config.PLATFORM_CONFIG.keys():
            if platform not in dataloaders:
                continue
            
            model = trainer.get_platform_model(platform)
            metrics = evaluate_federated_model(
                model, dataloaders[platform], platform, device,
                model_name=f"Federated_Repeat{repeat_idx+1}"
            )
            
            # 保存单次实验结果
            if platform not in all_repeat_results:
                all_repeat_results[platform] = []
            
            single_result = {
                "repeat_idx": repeat_idx,
                "model_type": "Federated"
            }
            
            # 添加所有评估指标
            for metric_name, metric_value in metrics.items():
                single_result[metric_name] = metric_value
            
            all_repeat_results[platform].append(single_result)
    
    # 计算多次实验的均值 ± 标准差
    statistical_results = []
    for platform, repeat_results in all_repeat_results.items():
        stat_res = calculate_statistical_results(repeat_results)
        
        # 动态构建统计结果
        platform_stat = {
            "platform": platform,
            "target_type": config.PLATFORM_CONFIG[platform]["target"],
            "total_params_million": global_param_stats["total_params_million"],
            "training_mode": "Federated_Distillation"
        }
        
        # 添加所有指标的统计结果
        for metric_key in repeat_results[0].keys():
            if metric_key not in ["repeat_idx", "model_type"]:
                mean_key = f"{metric_key}_mean"
                std_key = f"{metric_key}_std"
                formatted_key = f"{metric_key}_formatted"
                
                if mean_key in stat_res and std_key in stat_res:
                    platform_stat[f"{metric_key}_mean"] = stat_res[mean_key]
                    platform_stat[f"{metric_key}_std"] = stat_res[std_key]
                    platform_stat[f"{metric_key}_formatted"] = stat_res[formatted_key]
        
        statistical_results.append(platform_stat)
    
    # 保存联邦训练结果
    save_results(
        statistical_results, 
        os.path.join(config.RESULT_DIR, "federated_statistical_results.csv")
    )
    
    logger.info("\n=== 联邦训练统计结果 ===")
    for res in statistical_results:
        metric_outputs = []
        for key, value in res.items():
            if key.endswith("_formatted"):
                metric_outputs.append(f"{key.replace('_formatted', '')}: {value}")
        logger.info(f"{res['platform']} ({res['target_type']}) - " + ", ".join(metric_outputs))
    
    return statistical_results


if __name__ == "__main__":
    main()