#!/usr/bin/env python3
"""Debug script to test resume logic path matching"""
import json
from pathlib import Path
import glob

def debug_path_matching():
    print("🔍 Debugging resume logic path matching...")
    
    # Load the progress file
    progress_file = Path("/era-ai/lm/dataset/huggingface_cache/lerobot/fold_box_unified/conversion_progress.json")
    print(f"📂 Progress file: {progress_file}")
    print(f"   Exists: {progress_file.exists()}")
    
    if not progress_file.exists():
        print("❌ Progress file not found!")
        return
    
    with open(progress_file, 'r') as f:
        raw_mapping = json.load(f)
    
    print(f"📊 Progress file contains {len(raw_mapping)} entries")
    
    # Sample some entries
    sample_entries = list(raw_mapping.items())[:5]
    print("📋 Sample progress entries:")
    for path, episode_id in sample_entries:
        print(f"   {episode_id}: {path}")
    
    # Now check what files we're actually trying to process
    h5_data_paths = [
        "/era-ai/lm/dataset/lmc/fold_box_unified/score_1",
        "/era-ai/lm/dataset/lmc/fold_box_unified/score_5"
    ]
    
    all_h5_files = []
    for data_path in h5_data_paths:
        pattern = str(Path(data_path) / "*.hdf5")
        files = sorted(glob.glob(pattern))
        all_h5_files.extend([Path(f) for f in files])
    
    print(f"🔍 Found {len(all_h5_files)} H5 files to potentially process")
    print("📋 Sample H5 files:")
    for f in all_h5_files[:5]:
        print(f"   {f}")
    
    # Test path matching logic
    converted_mapping = {}
    for file_path, episode_id in raw_mapping.items():
        if not file_path.startswith("__"):
            # Normalize path to absolute format using resolve()
            try:
                normalized_path = str(Path(file_path).resolve())
                converted_mapping[normalized_path] = episode_id
            except Exception as e:
                print(f"⚠️  Failed to resolve {file_path}: {e}")
                converted_mapping[file_path] = episode_id
            
            # Also store filename for fallback matching
            filename = Path(file_path).name
            converted_mapping[f"__filename__{filename}"] = episode_id
    
    print(f"🔧 After normalization: {len(converted_mapping)} entries in mapping")
    
    # Test matching for first few files
    print("\n🧪 Testing path matching for first 10 files:")
    files_to_process = []
    episodes_to_process = []
    
    for i, h5_file in enumerate(all_h5_files[:10]):
        h5_file_abs = str(h5_file.resolve())
        h5_file_name = h5_file.name
        
        # Check multiple path formats for better matching
        is_converted = (
            h5_file_abs in converted_mapping or
            str(h5_file) in converted_mapping or
            f"__filename__{h5_file_name}" in converted_mapping
        )
        
        print(f"   File {i}: {h5_file_name}")
        print(f"      Absolute path: {h5_file_abs}")
        print(f"      Is converted: {is_converted}")
        
        if is_converted:
            print(f"      ✅ SKIPPING (already converted)")
        else:
            print(f"      🔄 WILL PROCESS")
            files_to_process.append(h5_file)
            episodes_to_process.append(i)
    
    print(f"\n📊 Summary:")
    print(f"   Total H5 files found: {len(all_h5_files)}")
    print(f"   Files in progress mapping: {len([k for k in raw_mapping.keys() if not k.startswith('__')])}")
    print(f"   Files that would be processed: {len(files_to_process)}")
    print(f"   Files that would be skipped: {len(all_h5_files[:10]) - len(files_to_process)}")

if __name__ == "__main__":
    debug_path_matching()