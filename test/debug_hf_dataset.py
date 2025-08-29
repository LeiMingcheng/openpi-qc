#!/usr/bin/env python3
"""
调试脚本：直接检查HuggingFace数据集
"""

import sys
import os
from pathlib import Path

# 设置环境变量
os.environ['HF_HOME'] = '/era-ai/lm/dataset/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/era-ai/lm/dataset/huggingface_cache/datasets'
os.environ['HF_LEROBOT_HOME'] = '/era-ai/lm/dataset/huggingface_cache/lerobot'

from datasets import load_dataset

def main():
    print("🔍 直接检查HuggingFace数据集...")
    
    try:
        # 直接加载HF数据集
        dataset = load_dataset('aloha_test_dataset', split='train')
        print(f"✅ 数据集加载成功")
        print(f"数据集长度: {len(dataset)}")
        print(f"列名: {dataset.column_names}")
        
        # 检查timestamp列
        timestamp_col = dataset['timestamp']
        print(f"\n🕐 timestamp列类型: {type(timestamp_col)}")
        print(f"timestamp示例: {timestamp_col[:3]}")
        
        # 检查是否有图像列
        image_columns = [col for col in dataset.column_names if 'image' in col.lower()]
        print(f"\n🖼️ 图像列: {image_columns}")
        
        # 检查第一行数据
        first_row = dataset[0]
        print(f"\n📋 第一行数据:")
        for key, value in first_row.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()