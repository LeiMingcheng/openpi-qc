"""
ACRLPDDataConverter: H5 → LeRobot format conversion with reward assignment.

This module provides data conversion from raw H5 robot datasets to LeRobot format
optimized for ACRLPD training, including flexible reward assignment strategies.

Key features:
- Multi-robot support (ALOHA, DROID, etc.)
- Flexible reward assignment (success-based, folder-based, custom)
- Multi-modal data handling (images + state + language)
- Episode boundary detection
- LeRobot dataset creation with proper schema
"""

import dataclasses
import logging
from pathlib import Path
import shutil
import json
from typing import Dict, List, Tuple, Any, Optional, Callable, Literal, Union
from enum import Enum

import h5py
import numpy as np
import torch
import tqdm
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


class RewardStrategy(Enum):
    """Reward assignment strategies for ACRLPD training."""
    SUCCESS_BASED = "success_based"      # Based on task success indicators
    FOLDER_BASED = "folder_based"        # Based on folder structure (positive/negative)
    SCORE_BASED = "score_based"          # Based on numerical scores in data
    CUSTOM = "custom"                    # Custom reward function
    EXISTING = "existing"                # Use existing rewards in H5 data


@dataclasses.dataclass
class RewardConfig:
    """Configuration for reward assignment."""
    strategy: RewardStrategy
    positive_reward: float = 1.0
    negative_reward: float = 0.0
    success_key: str = "success"         # H5 key for success indicator
    score_threshold: float = 0.5         # Threshold for score-based rewards  
    custom_reward_fn: Optional[Callable] = None
    reward_shaping: bool = False         # Whether to apply reward shaping


@dataclasses.dataclass 
class RobotConfig:
    """Robot-specific configuration."""
    robot_type: str
    motors: List[str]
    cameras: List[str]
    fps: int = 50
    has_velocity: bool = False
    has_effort: bool = False
    state_keys: List[str] = None         # H5 keys for robot state
    action_keys: List[str] = None        # H5 keys for actions
    image_shape: Tuple[int, int, int] = (3, 480, 640)  # (C, H, W)


# Predefined robot configurations
ALOHA_CONFIG = RobotConfig(
    robot_type="aloha",
    motors=[
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", 
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper"
    ],
    cameras=["cam_high", "cam_left_wrist", "cam_right_wrist"],
    state_keys=["/observations/qpos"],
    action_keys=["/action"],
    has_velocity=True,
    has_effort=True
)

DROID_CONFIG = RobotConfig(
    robot_type="droid", 
    motors=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
    cameras=["exterior_image_1", "exterior_image_2", "wrist_image"],
    state_keys=["/observations/robot_state"],
    action_keys=["/actions"],
    fps=10
)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """LeRobot dataset creation configuration with performance optimizations."""
    use_videos: bool = True
    tolerance_s: float = 0.0001
    # Optimized parallelism settings based on OpenPI benchmarks
    image_writer_processes: int = 4   # Reduced to prevent resource contention
    image_writer_threads: int = 2     # Reduced to minimize context switching
    video_backend: str | None = "av"  # Use av (FFmpeg) backend - much faster than SVT-AV1
    mode: Literal["video", "image"] = "video"
    # Emergency speed options
    fast_mode: bool = False           # When True, switches to image mode for fastest conversion


DEFAULT_DATASET_CONFIG = DatasetConfig()

# Fast mode config for maximum speed (like OpenPI default)
FAST_DATASET_CONFIG = DatasetConfig(
    use_videos=True,  # Keep videos enabled but use image mode
    video_backend=None,  # Use default backend like OpenPI
    mode="image",  # Use image mode like OpenPI default
    fast_mode=True,
    image_writer_processes=2,  # Reduced for stability
    image_writer_threads=1
)


class ACRLPDDataConverter:
    """
    Convert H5 robot datasets to LeRobot format with ACRLPD reward assignment.
    
    This class handles the complete pipeline from raw H5 episodes to standardized
    LeRobot datasets ready for ACRLPD training.
    """
    
    def __init__(
        self,
        robot_config: RobotConfig,
        reward_config: RewardConfig,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG
    ):
        """
        Initialize ACRLPDDataConverter.
        
        Args:
            robot_config: Robot-specific configuration
            reward_config: Reward assignment configuration  
            dataset_config: LeRobot dataset creation configuration
        """
        self.robot_config = robot_config
        self.reward_config = reward_config
        self.dataset_config = dataset_config
        
        # Statistics tracking
        self.stats = {
            'episodes_processed': 0,
            'total_frames': 0,
            'positive_episodes': 0,
            'negative_episodes': 0,
            'average_episode_length': 0.0,
            'reward_distribution': {}
        }
    
    def convert_h5_to_lerobot(
        self,
        h5_data_path: Union[str, Path, List[Path]],
        output_repo_id: str,
        task_name: str = "acrlpd_task",
        episodes: Optional[List[int]] = None,
        push_to_hub: bool = False,
        resume: bool = False,
        force_restart: bool = False
    ) -> LeRobotDataset:
        """
        Main conversion pipeline from H5 to LeRobot format.
        
        Args:
            h5_data_path: Path to H5 data directory or list of episode files
            output_repo_id: LeRobot dataset repository ID
            task_name: Task name for the dataset
            episodes: Optional list of specific episodes to process
            push_to_hub: Whether to push to HuggingFace Hub
            resume: Resume from existing dataset instead of starting fresh
            force_restart: Force delete existing dataset and restart
            
        Returns:
            Created LeRobot dataset
        """
        logger.info(f"Starting H5 → LeRobot conversion with {self.reward_config.strategy.value} reward strategy")
        
        # Store repo_id for use in progress tracking
        self._current_repo_id = output_repo_id
        
        # Handle folder-based strategy with list of directories  
        if (self.reward_config.strategy == RewardStrategy.FOLDER_BASED and 
            isinstance(h5_data_path, list) and len(h5_data_path) == 2):
            # Special case: [positive_dir, negative_dir]
            positive_dir, negative_dir = h5_data_path
            positive_files = self._get_h5_files(Path(positive_dir))
            negative_files = self._get_h5_files(Path(negative_dir))
            
            if not positive_files and not negative_files:
                raise ValueError(f"No H5 files found in either {positive_dir} or {negative_dir}")
            
            # Validate robot configuration using first available file
            sample_file = positive_files[0] if positive_files else negative_files[0]
            self._validate_robot_config(sample_file)
            
            # Create empty LeRobot dataset
            dataset = self._create_lerobot_dataset(output_repo_id, resume=resume, force_restart=force_restart)
            
            # Process positive and negative episodes separately
            if positive_files:
                actual_episodes = episodes if episodes is not None else list(range(len(positive_files)))
                num_to_process = min(len(actual_episodes), len(positive_files))
                logger.info(f"Processing {num_to_process}/{len(positive_files)} positive episodes from {positive_dir}")
                self._populate_dataset_with_rewards(
                    dataset, positive_files, task_name, self.reward_config.positive_reward, episodes, resume=resume
                )
            
            if negative_files:
                actual_episodes = episodes if episodes is not None else list(range(len(negative_files)))
                num_to_process = min(len(actual_episodes), len(negative_files))
                logger.info(f"Processing {num_to_process}/{len(negative_files)} negative episodes from {negative_dir}")
                self._populate_dataset_with_rewards(
                    dataset, negative_files, task_name, self.reward_config.negative_reward, episodes, resume=resume
                )
        
        else:
            # Handle single directory or other strategies
            if self.reward_config.strategy == RewardStrategy.FOLDER_BASED:
                # For folder-based, validate by checking subdirectories
                base_path = Path(h5_data_path)
                sample_file = None
                
                # Find a sample file from any subdirectory for validation
                for subdir in ["score_5", "score_1", "positive", "negative", "good", "bad"]:
                    subdir_path = base_path / subdir
                    if subdir_path.exists():
                        subdir_files = self._get_h5_files(subdir_path)
                        if subdir_files:
                            sample_file = subdir_files[0]
                            break
                
                if sample_file is None:
                    raise ValueError(f"No H5 files found in subdirectories of {h5_data_path}")
                
                # Validate robot configuration using sample file
                self._validate_robot_config(sample_file)
                
                # Create empty LeRobot dataset
                dataset = self._create_lerobot_dataset(output_repo_id, resume=resume, force_restart=force_restart)
                
                # Process folder-based episodes
                dataset = self._process_folder_based_episodes(dataset, h5_data_path, task_name, episodes, resume=resume)
                
            else:
                # For other strategies, check the directory directly
                if isinstance(h5_data_path, (str, Path)):
                    h5_files = self._get_h5_files(Path(h5_data_path))
                else:
                    h5_files = [Path(p) for p in h5_data_path]
                
                if not h5_files:
                    raise ValueError(f"No H5 files found in {h5_data_path}")
                
                # Validate robot configuration
                self._validate_robot_config(h5_files[0])
                
                # Create empty LeRobot dataset
                dataset = self._create_lerobot_dataset(output_repo_id, resume=resume, force_restart=force_restart)
                
                # Process single source episodes
                dataset = self._process_single_source_episodes(dataset, h5_files, task_name, episodes, resume=resume)
        
        # Finalize dataset
        logger.info(f"Conversion completed: {self.stats}")
        
        if push_to_hub:
            logger.info("Pushing dataset to HuggingFace Hub...")
            dataset.push_to_hub()
        
        return dataset
    
    def _get_h5_files(self, data_path: Path) -> List[Path]:
        """Get sorted list of H5 episode files."""
        if data_path.is_file() and data_path.suffix == '.hdf5':
            return [data_path]
        elif data_path.is_dir():
            # 支持多种episode文件命名模式
            patterns = ["episode_*.hdf5", "*episode_*.hdf5", "*.hdf5"]
            all_files = []
            for pattern in patterns:
                files = sorted(data_path.glob(pattern))
                all_files.extend(files)
            # 去重并按名称排序
            return sorted(list(set(all_files)))
        else:
            raise ValueError(f"Invalid H5 data path: {data_path}")
    
    def _validate_robot_config(self, sample_file: Path):
        """Validate robot configuration against actual H5 data."""
        try:
            with h5py.File(sample_file, "r") as f:
                # Check required state keys exist
                for state_key in self.robot_config.state_keys:
                    if state_key not in f:
                        logger.warning(f"State key {state_key} not found in H5 data")
                
                # Check action keys exist
                for action_key in self.robot_config.action_keys:
                    if action_key not in f:
                        logger.warning(f"Action key {action_key} not found in H5 data")
                
                # Check camera data
                if "/observations/images" in f:
                    available_cameras = list(f["/observations/images"].keys())
                    missing_cameras = set(self.robot_config.cameras) - set(available_cameras)
                    if missing_cameras:
                        logger.warning(f"Missing cameras: {missing_cameras}")
                        # Update robot config to use available cameras
                        self.robot_config.cameras = [c for c in self.robot_config.cameras if c in available_cameras]
                
                logger.info(f"Using cameras: {self.robot_config.cameras}")
                
        except Exception as e:
            logger.warning(f"Could not validate robot config: {e}")
    
    def _create_lerobot_dataset(self, repo_id: str, resume: bool = False, force_restart: bool = False) -> LeRobotDataset:
        """Create LeRobot dataset with resume functionality."""
        
        dataset_path = Path(HF_LEROBOT_HOME / repo_id)
        
        # Handle existing dataset
        if dataset_path.exists():
            if resume:
                logger.info(f"🔄 Resume mode: Loading existing dataset from {dataset_path}")
                # Load existing dataset with tolerance for problematic episodes
                try:
                    existing_dataset = LeRobotDataset(
                        repo_id, 
                        tolerance_s=self.dataset_config.tolerance_s,
                        skip_problematic_episodes=True  # Skip episodes with timestamp issues
                    )
                    logger.info(f"📊 Found {existing_dataset.num_episodes} existing episodes")
                    return existing_dataset
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load existing dataset: {e}")
                    logger.info(f"🔄 Will create new dataset instead")
            elif force_restart:
                logger.info(f"🗑️  Force restart: Deleting existing dataset at {dataset_path}")
                shutil.rmtree(dataset_path)
            else:
                raise ValueError(
                    f"Dataset '{repo_id}' already exists at {dataset_path}. "
                    f"Use --resume to continue or --force-restart to delete and restart."
                )
        elif resume:
            logger.warning(f"⚠️  Resume mode requested but no existing dataset found at {dataset_path}")
            logger.info(f"🆕 Creating new dataset instead")
            
        # Create new dataset with feature schema
        # Build feature schema
        features = {
            "observation.state": {
                "dtype": "float32", 
                "shape": (len(self.robot_config.motors),),
                "names": [self.robot_config.motors]
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.robot_config.motors),),
                "names": [self.robot_config.motors]
            },
            # Add reward for ACRLPD training
            "reward": {
                "dtype": "float32",
                "shape": (1,),
            },
        }
        
        # Add velocity if available
        if self.robot_config.has_velocity:
            features["observation.velocity"] = {
                "dtype": "float32",
                "shape": (len(self.robot_config.motors),),
                "names": [self.robot_config.motors]
            }
        
        # Add effort if available  
        if self.robot_config.has_effort:
            features["observation.effort"] = {
                "dtype": "float32",
                "shape": (len(self.robot_config.motors),),
                "names": [self.robot_config.motors]
            }
        
        # Add camera features
        for camera in self.robot_config.cameras:
            features[f"observation.images.{camera}"] = {
                "dtype": self.dataset_config.mode,
                "shape": self.robot_config.image_shape,
                "names": ["channels", "height", "width"]
            }
        
        # Apply fast_mode optimizations if enabled
        actual_use_videos = self.dataset_config.use_videos and not self.dataset_config.fast_mode
        actual_mode = "image" if self.dataset_config.fast_mode else self.dataset_config.mode
        
        # Override video backend to None if using image mode
        actual_video_backend = None if self.dataset_config.fast_mode else self.dataset_config.video_backend
        
        # Update image features dtype based on mode
        for feature_key in list(features.keys()):
            if feature_key.startswith("observation.images."):
                features[feature_key]["dtype"] = actual_mode
        
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=self.robot_config.fps,
            robot_type=self.robot_config.robot_type,
            features=features,
            use_videos=actual_use_videos,
            tolerance_s=self.dataset_config.tolerance_s,
            image_writer_processes=self.dataset_config.image_writer_processes,
            image_writer_threads=self.dataset_config.image_writer_threads,
            video_backend=actual_video_backend
        )
    
    def _get_converted_episodes(self, dataset: LeRobotDataset = None, repo_id: str = None) -> Dict[str, int]:
        """Get mapping of already converted H5 files to episode IDs with improved path matching."""
        converted_mapping = {}
        
        # Get dataset path using HF_LEROBOT_HOME and repo_id
        from lerobot.common.constants import HF_LEROBOT_HOME
        if dataset is not None:
            dataset_path = Path(HF_LEROBOT_HOME) / dataset.repo_id
        elif repo_id is not None:
            dataset_path = Path(HF_LEROBOT_HOME) / repo_id
        else:
            logger.warning("⚠️  No dataset or repo_id provided for progress check")
            return converted_mapping
            
        progress_file = dataset_path / "conversion_progress.json"
        
        # Try to load progress file first
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    raw_mapping = json.load(f)
                
                # Normalize paths in the mapping for better matching
                for file_path, episode_id in raw_mapping.items():
                    if file_path.startswith("__"):
                        # Keep special markers as-is
                        converted_mapping[file_path] = episode_id
                    else:
                        # Store multiple path variants for robust matching
                        # 1. Original path as stored in progress file
                        converted_mapping[file_path] = episode_id
                        
                        # 2. Normalized absolute path (handles symlinks)
                        try:
                            normalized_path = str(Path(file_path).resolve())
                            if normalized_path != file_path:
                                converted_mapping[normalized_path] = episode_id
                        except Exception:
                            pass  # Continue if resolution fails
                        
                        # 3. Filename for fallback matching
                        filename = Path(file_path).name
                        converted_mapping[f"__filename__{filename}"] = episode_id
                
                actual_files_count = len([k for k in raw_mapping.keys() if not k.startswith('__')])
                total_mapping_entries = len(converted_mapping) - len([k for k in converted_mapping.keys() if k.startswith('__')])
                logger.info(f"📋 Loaded progress: {actual_files_count} files already converted ({total_mapping_entries} path variants stored)")
                logger.debug(f"🔍 Sample progress paths: {list(raw_mapping.keys())[:3]}")
                logger.debug(f"🔍 Sample mapping entries: {[k for k in list(converted_mapping.keys())[:6] if not k.startswith('__')]}")
                return converted_mapping
                
            except Exception as e:
                logger.warning(f"⚠️  Could not load progress file: {e}")
        
        # If no progress file, try to infer from dataset episodes count
        data_chunk_path = dataset_path / "data" / "chunk-000"
        if data_chunk_path.exists():
            import glob
            episode_files = glob.glob(str(data_chunk_path / "episode_*.parquet"))
            num_existing_episodes = len(episode_files)
            
            if num_existing_episodes > 0:
                logger.info(f"📊 Found {num_existing_episodes} existing episodes in dataset")
                logger.info(f"🔄 Resume mode: Will attempt to skip first {num_existing_episodes} H5 files")
                
                # Return a special marker to indicate we should skip first N files
                converted_mapping["__skip_first__"] = num_existing_episodes
                
        return converted_mapping
    
    def _find_reference_image_idx(self, frame_idx: int, image_indices: np.ndarray) -> int:
        """Find the best image reference for a frame (DEPRECATED - use batch version)."""
        # Fallback to simple nearest-previous strategy for single calls
        valid_indices = image_indices[image_indices <= frame_idx]
        if len(valid_indices) > 0:
            return len(valid_indices) - 1
        else:
            return 0
    
    def _find_reference_image_index(self, frame_idx: int, image_indices: np.ndarray) -> int:
        """Find the most recent image index for a given frame - simple and fast.
        
        Much faster than complex vectorized operations for typical use cases.
        """
        # Simple approach: find the largest image_indices[i] <= frame_idx
        for i in range(len(image_indices) - 1, -1, -1):
            if image_indices[i] <= frame_idx:
                return i
        return 0  # Use first image if no match found
    
    def _batch_compute_image_references(self, num_frames: int, image_indices: np.ndarray) -> np.ndarray:
        """Vectorized computation of image references for all frames in an episode.
        
        This replaces the need to call _find_reference_image_index for each frame,
        providing significant speedup for chunked data.
        
        Args:
            num_frames: Total number of frames in the episode
            image_indices: Array of frame indices that have actual images
            
        Returns:
            Array of shape (num_frames,) with image reference indices for each frame
        """
        # Create array of all frame indices
        frame_indices = np.arange(num_frames)
        
        # Use numpy searchsorted for efficient batch lookup
        # searchsorted finds insertion points, we want the previous valid image
        reference_positions = np.searchsorted(image_indices, frame_indices, side='right') - 1
        
        # Handle edge case: frames before first image should use first image
        reference_positions = np.clip(reference_positions, 0, len(image_indices) - 1)
        
        return reference_positions
    
    def _save_conversion_progress(self, dataset: LeRobotDataset, h5_file: Path, episode_id: int, repo_id: str = None):
        """Save conversion progress to track completed files with consistent path normalization."""
        # Get dataset path using HF_LEROBOT_HOME and repo_id
        from lerobot.common.constants import HF_LEROBOT_HOME
        if dataset is not None and hasattr(dataset, 'repo_id'):
            dataset_path = Path(HF_LEROBOT_HOME) / dataset.repo_id
        elif repo_id is not None:
            dataset_path = Path(HF_LEROBOT_HOME) / repo_id
        else:
            logger.warning("⚠️  No repo_id available for progress saving")
            return
            
        progress_file = dataset_path / "conversion_progress.json"
        
        # Load existing mapping
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    mapping = json.load(f)
            except Exception:
                mapping = {}
        else:
            mapping = {}
        
        # Update mapping with normalized absolute path to ensure consistency
        normalized_path = str(h5_file.resolve())  # Use resolve() for consistent absolute paths
        mapping[normalized_path] = episode_id
        
        # Save updated mapping with proper formatting
        try:
            with open(progress_file, 'w') as f:
                json.dump(mapping, f, indent=2)
            logger.debug(f"💾 Saved progress: {h5_file.name} → episode {episode_id}")
        except Exception as e:
            logger.warning(f"⚠️  Could not save progress: {e}")
    
    def _process_folder_based_episodes(
        self,
        dataset: LeRobotDataset,
        base_path: Union[str, Path],
        task_name: str,
        episodes: Optional[List[int]] = None,
        resume: bool = False
    ) -> LeRobotDataset:
        """Process episodes with folder-based reward assignment."""
        base_path = Path(base_path)
        
        # Process positive episodes (支持多种命名约定)
        positive_dirs = [
            base_path / "positive",  # 标准命名
            base_path / "score_5",   # 分数命名 (高分=正奖励)
            base_path / "good",      # 简单命名
        ]
        
        for positive_dir in positive_dirs:
            if positive_dir.exists():
                positive_files = self._get_h5_files(positive_dir)
                logger.info(f"Processing {len(positive_files)} positive episodes from {positive_dir.name}")
                self._populate_dataset_with_rewards(
                    dataset, positive_files, task_name, self.reward_config.positive_reward, episodes, resume=resume
                )
                break
        
        # Process negative episodes (支持多种命名约定)
        negative_dirs = [
            base_path / "negative",  # 标准命名
            base_path / "score_1",   # 分数命名 (低分=负奖励)
            base_path / "bad",       # 简单命名
        ]
        
        for negative_dir in negative_dirs:
            if negative_dir.exists():
                negative_files = self._get_h5_files(negative_dir)
                logger.info(f"Processing {len(negative_files)} negative episodes from {negative_dir.name}")
                self._populate_dataset_with_rewards(
                    dataset, negative_files, task_name, self.reward_config.negative_reward, episodes, resume=resume
                )
                break
        
        return dataset
    
    def _process_single_source_episodes(
        self,
        dataset: LeRobotDataset,
        h5_files: List[Path],
        task_name: str,
        episodes: Optional[List[int]] = None,
        resume: bool = False
    ) -> LeRobotDataset:
        """Process episodes from single source with reward computation."""
        if episodes is None:
            episodes = list(range(len(h5_files)))
        
        # 🔄 RESUME: Filter out already converted files
        files_to_process = []
        if resume:
            # Get progress even if dataset loading failed
            converted_mapping = self._get_converted_episodes(dataset=dataset, repo_id=getattr(dataset, 'repo_id', self._current_repo_id))
            original_count = len(episodes)
            
            # Check for special skip marker
            if "__skip_first__" in converted_mapping:
                skip_count = converted_mapping["__skip_first__"]
                # Skip first N episodes
                episodes_to_process = episodes[skip_count:] if skip_count < len(episodes) else []
                files_to_process = [h5_files[ep_idx] for ep_idx in episodes_to_process]
                
                skipped_count = len(episodes) - len(episodes_to_process)
                logger.info(f"🔄 Resume mode: Skipping first {skipped_count} episodes (already in dataset)")
                logger.info(f"📋 Remaining to process: {len(episodes_to_process)} episodes")
            else:
                # Use original file-based matching
                for ep_idx in episodes:
                    h5_file = h5_files[ep_idx]
                    if str(h5_file.absolute()) not in converted_mapping:
                        files_to_process.append(h5_file)
                
                skipped_count = original_count - len(files_to_process)
                if skipped_count > 0:
                    logger.info(f"🔄 Resume mode: Skipping {skipped_count} already converted episodes")
                    logger.info(f"📋 Remaining to process: {len(files_to_process)} episodes")
                else:
                    logger.info(f"🔄 Resume mode: No converted episodes found, starting from beginning")
        else:
            files_to_process = [h5_files[ep_idx] for ep_idx in episodes]
        
        for h5_file in tqdm.tqdm(files_to_process, desc="Processing episodes"):
            try:
                # Load episode data
                episode_data = self._load_episode_data(h5_file)
                
                # Assign rewards
                rewards = self._assign_episode_rewards(episode_data, h5_file)
                
                # Add to dataset
                self._add_episode_to_dataset(dataset, episode_data, rewards, task_name)
                
                # 💾 Save progress after successful conversion
                current_episode_id = dataset.num_episodes - 1
                self._save_conversion_progress(dataset, h5_file, current_episode_id, repo_id=self._current_repo_id)
                
            except Exception as e:
                logger.error(f"❌ Failed to convert {h5_file}: {e}")
                continue
        
        return dataset
    
    def _populate_dataset_with_rewards(
        self,
        dataset: LeRobotDataset,
        h5_files: List[Path],
        task_name: str,
        reward_value: float,
        episodes: Optional[List[int]] = None,
        resume: bool = False
    ):
        """Populate dataset with fixed reward value, supporting resume functionality."""
        if episodes is None:
            episodes = list(range(len(h5_files)))
        
        # 🔑 CRITICAL FIX: Filter episodes to valid range and limit processing
        valid_episodes = [ep_idx for ep_idx in episodes if ep_idx < len(h5_files)]
        if len(valid_episodes) != len(episodes):
            logger.warning(f"Filtered {len(episodes)-len(valid_episodes)} invalid episode indices")
        
        # 🔄 RESUME: Filter out already converted files with improved path matching
        original_count = len(valid_episodes)
        if resume:
            # Get progress even if dataset loading failed
            converted_mapping = self._get_converted_episodes(dataset=dataset, repo_id=getattr(dataset, 'repo_id', self._current_repo_id))
            logger.info(f"🔍 Resume debug: Found {len(converted_mapping)} entries in progress mapping")
            
            # Check for special skip marker
            if "__skip_first__" in converted_mapping:
                skip_count = converted_mapping["__skip_first__"]
                # Skip first N episodes
                episodes_to_process = valid_episodes[skip_count:] if skip_count < len(valid_episodes) else []
                h5_files_filtered = [h5_files[ep_idx] for ep_idx in episodes_to_process]
                
                skipped_count = len(valid_episodes) - len(episodes_to_process)
                logger.info(f"🔄 Resume mode: Skipping first {skipped_count} episodes (already in dataset)")
                logger.info(f"📋 Remaining to process: {len(episodes_to_process)} episodes")
                
                valid_episodes = episodes_to_process
            else:
                # Use improved file-based matching with multiple path formats
                files_to_process = []
                episodes_to_process = []
                
                for ep_idx in valid_episodes:
                    h5_file = h5_files[ep_idx]
                    h5_file_abs = str(h5_file.resolve())  # Use resolve() for consistent absolute paths
                    h5_file_name = h5_file.name
                    
                    # Check multiple path formats for better matching
                    is_converted = (
                        h5_file_abs in converted_mapping or
                        str(h5_file) in converted_mapping or
                        f"__filename__{h5_file_name}" in converted_mapping
                    )
                    
                    if not is_converted:
                        files_to_process.append(h5_file)
                        episodes_to_process.append(ep_idx)
                    else:
                        logger.debug(f"🎯 Skipping already converted: {h5_file_name}")
                
                skipped_count = original_count - len(episodes_to_process)
                if skipped_count > 0:
                    logger.info(f"🔄 Resume mode: Skipping {skipped_count} already converted episodes")
                    logger.info(f"📋 Remaining to process: {len(episodes_to_process)} episodes")
                    
                    # Debug: Show sample of files being processed vs skipped
                    if len(episodes_to_process) > 0:
                        sample_to_process = [h5_files[ep_idx].name for ep_idx in episodes_to_process[:3]]
                        logger.debug(f"🎯 Sample files to process: {sample_to_process}")
                else:
                    logger.info(f"🔄 Resume mode: No converted episodes found, starting from beginning")
                    
                valid_episodes = episodes_to_process
                h5_files_filtered = files_to_process
        else:
            h5_files_filtered = [h5_files[ep_idx] for ep_idx in valid_episodes]
        
        # Early return if no episodes to process
        if len(valid_episodes) == 0:
            logger.warning("⚠️  No episodes to process after filtering!")
            logger.info("✅ All episodes appear to be already converted. Conversion complete!")
            return
        
        logger.info(f"   Actually processing {len(valid_episodes)} episodes (indices: {valid_episodes[:5]}{'...' if len(valid_episodes) > 5 else ''})")
        
        # Process episodes with error recovery and progress tracking
        for i, h5_file in enumerate(tqdm.tqdm(h5_files_filtered, desc=f"Processing episodes (reward={reward_value})")):
            try:
                # Load episode data
                episode_data = self._load_episode_data(h5_file)
                
                # Create constant rewards
                num_frames = len(episode_data['state'])
                rewards = np.full(num_frames, reward_value, dtype=np.float32)
                
                # Add frames to dataset (but don't save episode yet)
                self._add_episode_frames_to_dataset(dataset, episode_data, rewards, task_name)
                
                # 🔑 CRITICAL FIX: Save episode after each H5 file (not after all files)
                dataset.save_episode()
                
                # 💾 Save progress after successful conversion
                current_episode_id = dataset.num_episodes - 1
                self._save_conversion_progress(dataset, h5_file, current_episode_id, repo_id=self._current_repo_id)
                
            except Exception as e:
                logger.error(f"❌ Failed to convert {h5_file}: {e}")
                # Continue with next file rather than stopping entire conversion
                continue
            
            # Update statistics for this episode
            self._update_episode_stats(rewards, num_frames)
    
    def _load_episode_data(self, ep_path: Path) -> Dict[str, Any]:
        """Load complete episode data from H5 file."""
        with h5py.File(ep_path, "r") as f:
            episode_data = {}
            
            # Load state data
            state_arrays = []
            for state_key in self.robot_config.state_keys:
                if state_key in f:
                    state_arrays.append(f[state_key][:])
            episode_data['state'] = np.concatenate(state_arrays, axis=-1) if len(state_arrays) > 1 else state_arrays[0]
            
            # Load action data
            action_arrays = []
            for action_key in self.robot_config.action_keys:
                if action_key in f:
                    action_arrays.append(f[action_key][:])
            episode_data['actions'] = np.concatenate(action_arrays, axis=-1) if len(action_arrays) > 1 else action_arrays[0]
            
            # Load velocity if available
            if self.robot_config.has_velocity and "/observations/qvel" in f:
                episode_data['velocity'] = f["/observations/qvel"][:]
            
            # Load effort if available
            if self.robot_config.has_effort and "/observations/effort" in f:
                episode_data['effort'] = f["/observations/effort"][:]
            
            # 🔑 CHECK: Detect chunk-type inference samples (YZY samples)
            if 'image_indices' in f and 'chunk_starts' in f:
                episode_data['image_indices'] = f['image_indices'][:]
                episode_data['chunk_starts'] = f['chunk_starts'][:]
                episode_data['is_chunked'] = True
                logger.info(f"Detected chunk-type sample: {ep_path.name} ({len(episode_data['image_indices'])} images, {len(episode_data['state'])} actions)")
            else:
                episode_data['is_chunked'] = False
            
            # Load images
            episode_data['images'] = self._load_episode_images(f)
            
            # Load task-specific data for reward computation
            if self.reward_config.strategy == RewardStrategy.SUCCESS_BASED:
                if self.reward_config.success_key in f:
                    episode_data['success'] = f[self.reward_config.success_key][:]
            elif self.reward_config.strategy == RewardStrategy.SCORE_BASED:
                if "score" in f:
                    episode_data['score'] = f["score"][:]
            elif self.reward_config.strategy == RewardStrategy.EXISTING:
                if "reward" in f:
                    episode_data['existing_reward'] = f["reward"][:]
            
        return episode_data
    
    def _load_episode_images(self, h5_file: h5py.File) -> Dict[str, np.ndarray]:
        """Load images for all cameras in an episode with optimized decompression and caching."""
        images = {}
        
        if "/observations/images" not in h5_file:
            return images
            
        import cv2
        from concurrent.futures import ThreadPoolExecutor
        
        def decompress_single_image(compressed_data):
            """Decompress a single image efficiently."""
            img = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)
            # Convert BGR to RGB in one operation
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        def decompress_image_batch_parallel(compressed_batch):
            """Decompress a batch of images using parallel processing."""
            if len(compressed_batch) <= 4:
                # For small batches, sequential is faster due to threading overhead
                return [decompress_single_image(data) for data in compressed_batch]
            
            # For larger batches, use parallel processing
            with ThreadPoolExecutor(max_workers=min(4, len(compressed_batch))) as executor:
                decompressed = list(executor.map(decompress_single_image, compressed_batch))
            return decompressed
            
        for camera in self.robot_config.cameras:
            if camera not in h5_file["/observations/images"]:
                continue
                
            camera_data = h5_file[f"/observations/images/{camera}"]
            
            # Handle compressed vs uncompressed images
            if camera_data.ndim == 4:
                # Uncompressed - direct load
                images[camera] = camera_data[:]
            else:
                # Compressed - optimized parallel batch decompression
                num_images = len(camera_data)
                
                if num_images == 0:
                    images[camera] = np.array([])
                    continue
                
                # Use parallel decompression for better performance
                img_list = decompress_image_batch_parallel(camera_data)
                
                images[camera] = np.array(img_list, dtype=np.uint8)
        
        return images
    
    def _assign_episode_rewards(self, episode_data: Dict[str, Any], ep_path: Path) -> np.ndarray:
        """Assign rewards to episode based on strategy."""
        num_frames = len(episode_data['state'])
        
        if self.reward_config.strategy == RewardStrategy.SUCCESS_BASED:
            # Use success indicator
            if 'success' in episode_data:
                success = episode_data['success']
                if isinstance(success, (int, float, bool)):
                    # Single success value for entire episode
                    reward_value = self.reward_config.positive_reward if success else self.reward_config.negative_reward
                    rewards = np.full(num_frames, reward_value, dtype=np.float32)
                else:
                    # Per-step success values
                    rewards = np.where(success, self.reward_config.positive_reward, self.reward_config.negative_reward).astype(np.float32)
            else:
                logger.warning(f"No success indicator found in {ep_path}, using negative reward")
                rewards = np.full(num_frames, self.reward_config.negative_reward, dtype=np.float32)
        
        elif self.reward_config.strategy == RewardStrategy.SCORE_BASED:
            # Use numerical score with threshold
            if 'score' in episode_data:
                scores = episode_data['score'] 
                if isinstance(scores, (int, float)):
                    # Single score for entire episode
                    reward_value = self.reward_config.positive_reward if scores >= self.reward_config.score_threshold else self.reward_config.negative_reward
                    rewards = np.full(num_frames, reward_value, dtype=np.float32)
                else:
                    # Per-step scores
                    rewards = np.where(scores >= self.reward_config.score_threshold, 
                                     self.reward_config.positive_reward, self.reward_config.negative_reward).astype(np.float32)
            else:
                logger.warning(f"No score found in {ep_path}, using negative reward")
                rewards = np.full(num_frames, self.reward_config.negative_reward, dtype=np.float32)
        
        elif self.reward_config.strategy == RewardStrategy.EXISTING:
            # Use existing rewards
            if 'existing_reward' in episode_data:
                rewards = episode_data['existing_reward'].astype(np.float32)
            else:
                logger.warning(f"No existing rewards in {ep_path}, using zero rewards")
                rewards = np.zeros(num_frames, dtype=np.float32)
        
        elif self.reward_config.strategy == RewardStrategy.CUSTOM:
            # Apply custom reward function
            if self.reward_config.custom_reward_fn is not None:
                rewards = self.reward_config.custom_reward_fn(episode_data, ep_path)
                rewards = rewards.astype(np.float32)
            else:
                raise ValueError("Custom reward function not provided")
        
        else:
            raise ValueError(f"Unknown reward strategy: {self.reward_config.strategy}")
        
        # Apply reward shaping if enabled
        if self.reward_config.reward_shaping:
            rewards = self._apply_reward_shaping(rewards)
        
        return rewards
    
    def _apply_reward_shaping(self, rewards: np.ndarray) -> np.ndarray:
        """Apply reward shaping (e.g., gamma discounting)."""
        # Simple example: add small step penalty
        step_penalty = -0.01
        shaped_rewards = rewards + step_penalty
        
        # Keep final reward as is to preserve success/failure signal
        if len(rewards) > 0:
            shaped_rewards[-1] = rewards[-1]
        
        return shaped_rewards
    
    def _add_episode_frames_to_dataset(
        self,
        dataset: LeRobotDataset,
        episode_data: Dict[str, Any],
        rewards: np.ndarray,
        task_name: str
    ):
        """Add frames of one episode to LeRobot dataset (without saving episode)."""
        num_frames = len(episode_data['state'])
        is_chunked = episode_data.get('is_chunked', False)
        
        if is_chunked:
            # 🔑 CHUNK-TYPE SAMPLE: Use intelligent image reference + metadata
            image_indices = episode_data['image_indices'] 
            chunk_starts = episode_data['chunk_starts']
            
            # Create episode metadata for data loader compatibility
            episode_metadata = {
                'is_chunked': True,
                'actual_image_frames': image_indices.tolist(),
                'chunk_starts': chunk_starts.tolist(),
                'total_frames': num_frames,
                'num_actual_images': len(image_indices),
                'sparse_ratio': len(image_indices) / num_frames
            }
            
            # Store metadata in episode-level info (can be accessed by data loader)
            try:
                dataset.episode_data[len(dataset.episode_data_index)] = episode_metadata
            except AttributeError:
                # Fallback if episode_data attribute doesn't exist
                logger.debug("Could not store episode metadata - LeRobot version may not support it")
            
            # 🚀 VECTORIZED BATCH OPTIMIZATION: Pre-compute all image references at once
            # This replaces the expensive per-frame _find_reference_image_index calls
            logger.debug(f"Computing image references for {num_frames} frames with {len(image_indices)} images")
            image_ref_indices = self._batch_compute_image_references(num_frames, image_indices)
            
            for frame_idx in range(num_frames):
                frame = {
                    "observation.state": torch.from_numpy(episode_data['state'][frame_idx].astype(np.float32)),
                    "action": torch.from_numpy(episode_data['actions'][frame_idx].astype(np.float32)),
                    "reward": np.array([rewards[frame_idx]], dtype=np.float32),
                    "task": task_name,
                }
                
                # Add velocity if available
                if 'velocity' in episode_data:
                    frame["observation.velocity"] = torch.from_numpy(episode_data['velocity'][frame_idx].astype(np.float32))
                
                # Add effort if available
                if 'effort' in episode_data:
                    frame["observation.effort"] = torch.from_numpy(episode_data['effort'][frame_idx].astype(np.float32))
                
                # 🚀 OPTIMIZED IMAGE REFERENCE: Use pre-computed vectorized lookup
                ref_img_idx = image_ref_indices[frame_idx]
                for camera, img_array in episode_data['images'].items():
                    if camera in self.robot_config.cameras:
                        frame[f"observation.images.{camera}"] = img_array[ref_img_idx]
                
                dataset.add_frame(frame)
        
        else:
            # 🔑 NORMAL SAMPLE: Original logic unchanged for compatibility
            for frame_idx in range(num_frames):
                frame = {
                    "observation.state": torch.from_numpy(episode_data['state'][frame_idx].astype(np.float32)),
                    "action": torch.from_numpy(episode_data['actions'][frame_idx].astype(np.float32)),
                    "reward": np.array([rewards[frame_idx]], dtype=np.float32),
                    "task": task_name,
                }
                
                # Add velocity if available
                if 'velocity' in episode_data:
                    frame["observation.velocity"] = torch.from_numpy(episode_data['velocity'][frame_idx].astype(np.float32))
                
                # Add effort if available
                if 'effort' in episode_data:
                    frame["observation.effort"] = torch.from_numpy(episode_data['effort'][frame_idx].astype(np.float32))
                
                # Add images (every frame has image)
                for camera, img_array in episode_data['images'].items():
                    if camera in self.robot_config.cameras:
                        frame[f"observation.images.{camera}"] = img_array[frame_idx]
                
                dataset.add_frame(frame)
        
        # 🔑 CRITICAL FIX: Don't save episode here! Let caller handle episode saving
        # This ensures each H5 file = 1 episode, not all H5 files = 1 episode
    
    def _add_episode_to_dataset(
        self,
        dataset: LeRobotDataset,
        episode_data: Dict[str, Any],
        rewards: np.ndarray,
        task_name: str
    ):
        """Add complete episode to LeRobot dataset (legacy method for _process_single_source_episodes)."""
        # Add frames
        self._add_episode_frames_to_dataset(dataset, episode_data, rewards, task_name)
        
        # Save episode
        dataset.save_episode()
        
        # Update statistics
        self._update_episode_stats(rewards, len(episode_data['state']))
    
    def _update_episode_stats(self, rewards: np.ndarray, num_frames: int):
        """Update conversion statistics."""
        self.stats['episodes_processed'] += 1
        self.stats['total_frames'] += num_frames
        
        # Classify episode as positive/negative based on final reward
        final_reward = rewards[-1] if len(rewards) > 0 else 0
        if final_reward > 0:
            self.stats['positive_episodes'] += 1
        else:
            self.stats['negative_episodes'] += 1
        
        # Update average episode length
        self.stats['average_episode_length'] = self.stats['total_frames'] / self.stats['episodes_processed']
        
        # Update reward distribution
        unique_rewards, counts = np.unique(rewards, return_counts=True)
        for reward_val, count in zip(unique_rewards, counts):
            reward_key = f"{reward_val:.1f}"
            self.stats['reward_distribution'][reward_key] = self.stats['reward_distribution'].get(reward_key, 0) + count
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get comprehensive conversion statistics."""
        return {
            **self.stats,
            'robot_type': self.robot_config.robot_type,
            'reward_strategy': self.reward_config.strategy.value,
            'cameras_used': self.robot_config.cameras,
            'motors_count': len(self.robot_config.motors)
        }


def create_acrlpd_data_converter(
    robot_type: str = "aloha",
    reward_strategy: str = "success_based",
    custom_robot_config: Optional[RobotConfig] = None,
    **kwargs
) -> ACRLPDDataConverter:
    """
    Factory function to create ACRLPDDataConverter with common configurations.
    
    Args:
        robot_type: Robot type ("aloha", "droid") 
        reward_strategy: Reward assignment strategy
        custom_robot_config: Custom robot configuration (overrides robot_type)
        **kwargs: Additional arguments for RewardConfig and DatasetConfig
        
    Returns:
        ACRLPDDataConverter instance
    """
    # Get robot configuration
    if custom_robot_config is not None:
        robot_config = custom_robot_config
    elif robot_type.lower() == "aloha":
        robot_config = ALOHA_CONFIG
    elif robot_type.lower() == "droid":
        robot_config = DROID_CONFIG  
    else:
        raise ValueError(f"Unsupported robot type: {robot_type}. Use custom_robot_config for custom robots.")
    
    # Create reward configuration
    reward_config = RewardConfig(
        strategy=RewardStrategy(reward_strategy),
        **{k: v for k, v in kwargs.items() if k in RewardConfig.__dataclass_fields__}
    )
    
    # Create dataset configuration - use fast config if fast_mode is requested
    if kwargs.get('fast_mode', False):
        # Override with fast mode settings like OpenPI
        dataset_config = FAST_DATASET_CONFIG
    else:
        dataset_config = DatasetConfig(
            **{k: v for k, v in kwargs.items() if k in DatasetConfig.__dataclass_fields__}
        )
    
    return ACRLPDDataConverter(
        robot_config=robot_config,
        reward_config=reward_config,
        dataset_config=dataset_config
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert H5 robot data to LeRobot format for ACRLPD")
    parser.add_argument("--input-dir", type=str, required=True, help="Input H5 data directory")
    parser.add_argument("--output-repo-id", type=str, required=True, help="Output LeRobot repo ID")
    parser.add_argument("--robot-type", type=str, default="aloha", choices=["aloha", "droid"])
    parser.add_argument("--reward-strategy", type=str, default="success_based", 
                       choices=["success_based", "folder_based", "score_based", "existing"])
    parser.add_argument("--task-name", type=str, default="acrlpd_task")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing dataset instead of starting fresh")
    parser.add_argument("--force-restart", action="store_true",
                       help="Force delete existing dataset and restart from beginning")
    parser.add_argument("--fast-mode", action="store_true",
                       help="Enable fast mode (uses image format instead of video for maximum speed)")
    
    args = parser.parse_args()
    
    # Create converter with optimized settings
    converter = create_acrlpd_data_converter(
        robot_type=args.robot_type,
        reward_strategy=args.reward_strategy,
        fast_mode=args.fast_mode
    )
    
    # Validate arguments
    if args.resume and args.force_restart:
        raise ValueError("Cannot use --resume and --force-restart together. Choose one.")
    
    # Convert data
    dataset = converter.convert_h5_to_lerobot(
        h5_data_path=args.input_dir,
        output_repo_id=args.output_repo_id,
        task_name=args.task_name,
        push_to_hub=args.push_to_hub,
        resume=args.resume,
        force_restart=args.force_restart
    )
    
    # Print statistics
    stats = converter.get_conversion_stats()
    print("Conversion completed successfully!")
    print(f"Statistics: {stats}")