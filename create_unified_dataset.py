#!/usr/bin/env python3
"""
创建统一数据集的软连接脚本

将yzy_fold_box和fold_box两个数据集通过软连接整合到fold_box_unified中，
避免文件名冲突，使用前缀区分不同来源的数据。
"""

import os
import sys
from pathlib import Path

def create_unified_dataset():
    """创建统一数据集的软连接"""
    
    # 源数据路径
    yzy_fold_box_path = Path("/era-ai/lm/dataset/lmc/yzy_fold_box")
    fold_box_path = Path("/era-ai/lm/dataset/lmc/fold_box")
    
    # 目标统一数据路径
    unified_path = Path("/era-ai/lm/dataset/lmc/fold_box_unified")
    
    print(f"创建统一数据集: {unified_path}")
    print(f"源数据集1: {yzy_fold_box_path}")
    print(f"源数据集2: {fold_box_path}")
    
    # 验证源路径存在
    if not yzy_fold_box_path.exists():
        raise FileNotFoundError(f"源数据集不存在: {yzy_fold_box_path}")
    if not fold_box_path.exists():
        raise FileNotFoundError(f"源数据集不存在: {fold_box_path}")
    
    print("\n=== 整合 score_1 数据 (低质量) ===")
    score_1_unified = unified_path / "score_1"
    
    # 1. 先链接yzy_fold_box/score_1中的所有文件 (145个episodes)
    yzy_score_1 = yzy_fold_box_path / "score_1"
    if yzy_score_1.exists():
        episodes = sorted(yzy_score_1.glob("episode_*.hdf5"))
        print(f"从yzy_fold_box/score_1链接 {len(episodes)} 个episodes")
        
        for episode_file in episodes:
            # 使用yzy_前缀避免冲突
            target_name = f"yzy_{episode_file.name}"
            target_path = score_1_unified / target_name
            
            if not target_path.exists():
                os.symlink(episode_file, target_path)
                print(f"  ✓ {episode_file.name} -> {target_name}")
            else:
                print(f"  ⚠ 跳过已存在: {target_name}")
    
    # 2. 再链接fold_box/score_1中的文件 (21个episodes)
    fold_score_1 = fold_box_path / "score_1"
    if fold_score_1.exists():
        episodes = sorted(fold_score_1.glob("episode_*.hdf5"))
        print(f"从fold_box/score_1链接 {len(episodes)} 个episodes")
        
        for episode_file in episodes:
            # 使用fold_前缀避免冲突
            target_name = f"fold_{episode_file.name}"
            target_path = score_1_unified / target_name
            
            if not target_path.exists():
                os.symlink(episode_file, target_path)
                print(f"  ✓ {episode_file.name} -> {target_name}")
            else:
                print(f"  ⚠ 跳过已存在: {target_name}")
    
    print("\n=== 整合 score_5 数据 (高质量) ===")
    score_5_unified = unified_path / "score_5"
    
    # 1. 先链接fold_box/score_5中的所有文件 (147个episodes)
    fold_score_5 = fold_box_path / "score_5"
    if fold_score_5.exists():
        episodes = sorted(fold_score_5.glob("episode_*.hdf5"))
        print(f"从fold_box/score_5链接 {len(episodes)} 个episodes")
        
        for episode_file in episodes:
            # 使用fold_前缀
            target_name = f"fold_{episode_file.name}"
            target_path = score_5_unified / target_name
            
            if not target_path.exists():
                os.symlink(episode_file, target_path)
                print(f"  ✓ {episode_file.name} -> {target_name}")
            else:
                print(f"  ⚠ 跳过已存在: {target_name}")
    
    # 2. 再链接yzy_fold_box/score_5中的文件 (3个episodes)
    yzy_score_5 = yzy_fold_box_path / "score_5"
    if yzy_score_5.exists():
        episodes = sorted(yzy_score_5.glob("episode_*.hdf5"))
        print(f"从yzy_fold_box/score_5链接 {len(episodes)} 个episodes")
        
        for episode_file in episodes:
            # 使用yzy_前缀避免冲突
            target_name = f"yzy_{episode_file.name}"
            target_path = score_5_unified / target_name
            
            if not target_path.exists():
                os.symlink(episode_file, target_path)
                print(f"  ✓ {episode_file.name} -> {target_name}")
            else:
                print(f"  ⚠ 跳过已存在: {target_name}")
    
    # 统计结果
    print("\n=== 整合完成 ===")
    score_1_count = len(list(score_1_unified.glob("*.hdf5")))
    score_5_count = len(list(score_5_unified.glob("*.hdf5")))
    total_count = score_1_count + score_5_count
    
    print(f"score_1 (低质量): {score_1_count} episodes")
    print(f"score_5 (高质量): {score_5_count} episodes")
    print(f"总计: {total_count} episodes")
    
    print(f"\n✅ 统一数据集创建完成: {unified_path}")
    print(f"现在可以使用该路径进行数据转换和训练")

if __name__ == "__main__":
    try:
        create_unified_dataset()
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)