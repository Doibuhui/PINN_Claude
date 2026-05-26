"""
PINN故障诊断系统 - 主入口
支持两种分类器:
  - softmax: 原始端到端训练 (fc层 + CrossEntropyLoss)
  - elm: 两阶段训练 (PINN特征提取 + ELM解析分类)
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataloader import get_dataloaders
from pinn_model import PINNFaultDiagnosis
from train import Trainer
from evaluate import Evaluator
from elm_classifier import ELMWithPINN
from utils import set_seed, count_parameters, get_device


def main():
    config = Config()
    set_seed(config.SEED)
    
    print("=" * 60)
    print(f"PINN故障诊断系统 - 分类器: {config.CLASSIFIER.upper()}")
    print("=" * 60)
    
    device = get_device(config)
    
    # 加载数据
    print("\n加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建PINN模型...")
    model = PINNFaultDiagnosis(config)
    total_params, trainable_params = count_parameters(model)
    
    # 打印配置和参数量
    print(f"\n--- 当前配置 ---")
    print(f"  卷积类型:    {config.CONV_TYPE}")
    print(f"  残差块类型:  {config.RESIDUAL_TYPE}")
    print(f"  多头注意力:  {'ON' if config.USE_MHA else 'OFF'}")
    print(f"  输入通道:    {config.IN_CHANNELS}")
    print(f"  分类数:      {config.NUM_CLASSES}")
    print(f"\n--- 参数量分布 ---")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {name:20s} {params:>10,}")
    print(f"  {'─'*32}")
    print(f"  {'Total':20s} {total_params:>10,}")
    print(f"  {'Trainable':20s} {trainable_params:>10,}")
    print()
    
    # ==================== 阶段一: 训练 PINN 特征提取器 ====================
    print("\n" + "=" * 60)
    print("阶段一: PINN 特征提取器训练")
    print("=" * 60)
    trainer = Trainer(model, config, train_loader, val_loader)
    history = trainer.train()
    
    # 加载最佳模型
    evaluator = Evaluator(model, config, test_loader)
    evaluator.load_best_model()
    
    if config.CLASSIFIER == 'elm':
        # ==================== 阶段二: ELM 分类 ====================
        elm_model = ELMWithPINN(
            model, config,
            n_hidden=config.ELM_HIDDEN,
            activation=config.ELM_ACTIVATION,
            reg=config.ELM_REG
        )
        elm_model.fit_elm(train_loader, val_loader)
        results = elm_model.evaluate(test_loader, config.CLASS_NAMES)
    else:
        # ==================== Softmax 分类 (原始方式) ====================
        results = evaluator.evaluate()
        evaluator.print_results(results)
    
    print("\n训练和评估完成！")


if __name__ == '__main__':
    main()
