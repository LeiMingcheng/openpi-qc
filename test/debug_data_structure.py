#!/usr/bin/env python3
"""
调试脚本：检查aloha_test_dataset的实际数据结构
"""

import sys
import os
from pathlib import Path

# 设置环境变量（与训练脚本一致）
os.environ['HF_HOME'] = '/era-ai/lm/dataset/huggingface'
os.environ['HF_CACHE_HOME'] = '/era-ai/lm/dataset/huggingface_cache'
os.environ['HF_DATASETS_CACHE'] = '/era-ai/lm/dataset/huggingface_cache/datasets'
os.environ['HF_LEROBOT_HOME'] = '/era-ai/lm/dataset/huggingface_cache/lerobot'

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / "ac_training"))

# 应用PyAV patch
import lerobot.common.datasets.video_utils as video_utils
if not hasattr(video_utils, '_openpi_patched'):
    print("🔧 应用PyAV patch...")
    def patched_decode_video_frames(video_path, timestamps, tolerance_s, backend="pyav"):
        return video_utils.decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend="pyav")
    video_utils.decode_video_frames = patched_decode_video_frames
    video_utils._openpi_patched = True

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from datasets import load_dataset

def debug_hf_dataset_directly():
    """直接调试HuggingFace数据集，不通过LeRobot包装"""
    print("🔍 步骤1: 直接检查HuggingFace数据集...")
    
    try:
        # 尝试多种方式加载数据集
        dataset_paths = [
            'aloha_test_dataset',
            '/era-ai/lm/dataset/huggingface_cache/lerobot/aloha_test_dataset',
            '/era-ai/lm/dataset/huggingface/datasets/aloha_test_dataset'
        ]
        
        dataset = None
        for path in dataset_paths:
            try:
                print(f"  尝试路径: {path}")
                dataset = load_dataset(path, split='train')
                print(f"  ✅ 成功加载: {path}")
                break
            except Exception as e:
                print(f"  ❌ 失败: {e}")
        
        if dataset is None:
            print("❌ 所有路径都失败，尝试检查缓存目录")
            return
        
        print(f"\n📊 数据集基本信息:")
        print(f"  长度: {len(dataset)}")
        print(f"  列名: {dataset.column_names}")
        
        # 检查是否缺少必要的列
        required_columns = ['timestamp', 'action', 'observation.state']
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        print(f"  缺少的必要列: {missing_columns}")
        
        # 详细检查现有列
        print(f"\n🔍 详细检查现有列:")
        for col_name in dataset.column_names:
            col_data = dataset[col_name]
            print(f"  {col_name}:")
            print(f"    类型: {type(col_data)}")
            print(f"    长度: {len(col_data)}")
            
            # 检查前几个元素
            try:
                first_few = [col_data[i] for i in range(min(3, len(col_data)))]
                print(f"    前3个元素类型: {[type(x) for x in first_few]}")
                for i, elem in enumerate(first_few):
                    if hasattr(elem, 'shape'):
                        print(f"      [{i}] shape: {elem.shape}")
                    elif hasattr(elem, '__len__') and not isinstance(elem, str):
                        print(f"      [{i}] length: {len(elem)}")
                    print(f"      [{i}] sample: {str(elem)[:100]}...")
            except Exception as e:
                print(f"    访问元素失败: {e}")
        
        return dataset  # 返回数据集供后续使用
        
        # 尝试不同的访问方式
        print(f"\n🔍 尝试不同的timestamp访问方式:")
        try:
            # 方式1: 直接索引
            ts_0 = timestamp_col[0]
            print(f"  timestamp[0]: {type(ts_0)} = {ts_0}")
        except Exception as e:
            print(f"  timestamp[0] 失败: {e}")
        
        try:
            # 方式2: 转换为列表
            ts_list = list(timestamp_col)
            print(f"  list(timestamp): 长度={len(ts_list)}, 前3个={ts_list[:3]}")
            print(f"  list元素类型: {[type(x) for x in ts_list[:3]]}")
        except Exception as e:
            print(f"  list(timestamp) 失败: {e}")
        
        try:
            # 方式3: 转换为numpy
            import numpy as np
            ts_np = np.array(timestamp_col)
            print(f"  np.array(timestamp): {ts_np.shape} {ts_np.dtype}")
        except Exception as e:
            print(f"  np.array(timestamp) 失败: {e}")
        
        try:
            # 方式4: 转换为torch tensor
            import torch
            ts_torch = torch.tensor(list(timestamp_col))
            print(f"  torch.tensor(list(timestamp)): {ts_torch.shape} {ts_torch.dtype}")
        except Exception as e:
            print(f"  torch.tensor转换失败: {e}")
        
        # 检查第一行完整数据
        print(f"\n📋 第一行完整数据:")
        first_row = dataset[0]
        for key, value in first_row.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"    length: {len(value)}")
            print(f"    sample: {str(value)[:100]}...")
            
    except Exception as e:
        print(f"❌ HF数据集检查失败: {e}")
        import traceback
        traceback.print_exc()

def test_torch_stack_with_column():
    """测试torch.stack与Column对象的交互"""
    print("\n🔍 步骤2: 测试torch.stack与Column对象...")
    
    try:
        dataset = load_dataset('/era-ai/lm/dataset/huggingface_cache/lerobot/aloha_test_dataset', split='train')
        
        # 找到任何一个数值列进行测试（因为我们现在知道没有timestamp列）
        column_name = dataset.column_names[0]  # 使用第一个可用列
        col_data = dataset[column_name]
        
        print(f"测试列: {column_name}")
        print(f"  列类型: {type(col_data)}")
        print(f"  列的__class__: {col_data.__class__}")
        print(f"  列的模块: {col_data.__class__.__module__}")
        print(f"  是否是Column: {'Column' in col_data.__class__.__name__}")
        
        # 详细检查Column对象的属性
        print(f"\n🔍 Column对象详细信息:")
        print(f"  dir(col_data): {[attr for attr in dir(col_data) if not attr.startswith('_')]}")
        
        # 检查前几个元素
        print(f"\n🔍 列数据样本:")
        for i in range(min(3, len(col_data))):
            elem = col_data[i]
            print(f"  [{i}]: {type(elem)} = {str(elem)[:100]}...")
        
        # 尝试torch.stack
        print(f"\n🔍 尝试torch.stack(col_data):")
        import torch
        try:
            result = torch.stack(col_data)
            print(f"  ✅ 直接stack成功! 结果: {type(result)}, shape: {result.shape}")
        except Exception as e:
            print(f"  ❌ 直接stack失败: {e}")
            print(f"  错误类型: {type(e)}")
            
            # 尝试各种转换方法
            print(f"\n🔧 尝试修复方法:")
            
            # 方法1: 转换为列表再stack
            try:
                list_data = list(col_data)
                print(f"  list(col_data): 长度={len(list_data)}, 前3个类型={[type(x) for x in list_data[:3]]}")
                tensor_list = [torch.as_tensor(item) for item in list_data]
                result1 = torch.stack(tensor_list)
                print(f"  ✅ 方法1成功: list->tensor_list->stack, shape: {result1.shape}")
                return True
            except Exception as e1:
                print(f"  ❌ 方法1失败: {e1}")
            
            # 方法2: 直接转换Column
            try:
                result2 = torch.as_tensor(col_data)
                print(f"  ✅ 方法2成功: torch.as_tensor(column), shape: {result2.shape}")
                return True
            except Exception as e2:
                print(f"  ❌ 方法2失败: {e2}")
            
            # 方法3: 检查Column是否有特殊方法
            if hasattr(col_data, 'to_pylist'):
                try:
                    pylist = col_data.to_pylist()
                    result3 = torch.stack([torch.as_tensor(x) for x in pylist])
                    print(f"  ✅ 方法3成功: to_pylist->stack, shape: {result3.shape}")
                    return True
                except Exception as e3:
                    print(f"  ❌ 方法3失败: {e3}")
        
        return False
        
    except Exception as e:
        print(f"❌ 整体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_torch_stack_patch():
    """测试torch.stack的patch修复"""
    print("\n🔍 步骤3: 测试torch.stack patch修复...")
    
    # 应用patch
    import torch
    if not hasattr(torch, '_debug_patch_applied'):
        print("🔧 应用torch.stack Column兼容性patch...")
        original_stack = torch.stack
        
        def patched_stack(tensors, dim=0, *, out=None):
            print(f"  [PATCH] torch.stack调用, 输入类型: {type(tensors)}")
            if hasattr(tensors, '__class__'):
                print(f"  [PATCH] 输入类名: {tensors.__class__.__name__}")
                print(f"  [PATCH] 输入模块: {tensors.__class__.__module__}")
            
            # 检查是否是HF Dataset的Column对象
            if hasattr(tensors, '__class__') and 'Column' in tensors.__class__.__name__:
                print(f"  [PATCH] 检测到Column对象，进行转换...")
                # 尝试转换为tensor列表
                try:
                    tensor_list = [torch.as_tensor(item) for item in tensors]
                    print(f"  [PATCH] 转换成功: {len(tensor_list)}个tensor")
                    return original_stack(tensor_list, dim=dim, out=out)
                except Exception as e:
                    print(f"  [PATCH] 转换失败: {e}")
                    raise e
            else:
                print(f"  [PATCH] 正常tensor，直接调用原始stack")
                return original_stack(tensors, dim=dim, out=out)
        
        torch.stack = patched_stack
        torch._debug_patch_applied = True
        print("✅ Patch应用完成")
    
    # 重新测试Column对象
    print("\n🔧 重新测试Column对象处理:")
    success = test_torch_stack_with_column()
    return success

def test_lerobot_dataset_with_patch():
    """使用patch测试LeRobotDataset创建"""
    print("\n🔍 步骤4: 使用patch测试LeRobotDataset...")
    
    try:
        dataset = LeRobotDataset('aloha_test_dataset', skip_problematic_episodes=True)
        print(f"✅ LeRobotDataset创建成功!")
        print(f"  数据集长度: {len(dataset)}")
        
        # 尝试获取一个样本
        sample = dataset[0]
        print(f"✅ 样本获取成功!")
        print(f"  样本键: {list(sample.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ LeRobotDataset仍然失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 详细调试ALOHA Test Dataset的Column问题...")
    
    # 步骤1: 直接检查HF数据集
    dataset = debug_hf_dataset_directly()
    
    # 步骤2: 测试torch.stack问题
    original_success = test_torch_stack_with_column()
    
    # 步骤3: 测试patch修复
    patch_success = test_torch_stack_patch()
    
    # 步骤4: 测试LeRobotDataset with patch
    lerobot_success = test_lerobot_dataset_with_patch()
    
    # 总结
    print(f"\n📊 测试总结:")
    print(f"  原始torch.stack: {'✅' if original_success else '❌'}")
    print(f"  Patch修复测试: {'✅' if patch_success else '❌'}")
    print(f"  LeRobotDataset: {'✅' if lerobot_success else '❌'}")
    
    if patch_success and lerobot_success:
        print(f"\n🎉 修复验证成功! 可以应用到训练脚本中")
    else:
        print(f"\n⚠️ 修复需要进一步调整")

if __name__ == "__main__":
    main()