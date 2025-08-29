"""
Simple ACRLPD Data Converter: Optimized H5 → LeRobot format conversion.

Based on OpenPI's efficient converter design with ACRLPD-specific features:
- Folder-based reward assignment (score_5=1.0, score_1=0.0)  
- Chunk-type sparse image data support
- Q-chunking compatibility
- High-performance parallelism (10 processes, 5 threads)
- Optional resume functionality

Expected performance: 30-40x faster than complex ACRLPD converter.
"""

import dataclasses
import json
import logging
from pathlib import Path
import shutil
from typing import Literal, Optional, List

import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import numpy as np
import torch
import tqdm
import tyro

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    """High-performance dataset configuration based on OpenPI."""
    use_videos: bool = True
    tolerance_s: float = 0.0001
    # OpenPI's proven high-performance settings
    image_writer_processes: int = 10  
    image_writer_threads: int = 5
    video_backend: str | None = None
    mode: Literal["video", "image"] = "image"  # Image mode for maximum speed


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str = "aloha",
    mode: Literal["video", "image"] = "image",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    """Create empty LeRobot dataset with ACRLPD features (reward field)."""
    
    # ALOHA robot configuration
    motors = [
        "right_waist", "right_shoulder", "right_elbow", "right_forearm_roll", 
        "right_wrist_angle", "right_wrist_rotate", "right_gripper",
        "left_waist", "left_shoulder", "left_elbow", "left_forearm_roll",
        "left_wrist_angle", "left_wrist_rotate", "left_gripper"
    ]
    cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

    # Essential features for ACRLPD/Q-chunking
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        },
        "action": {
            "dtype": "float32", 
            "shape": (len(motors),),
            "names": [motors],
        },
        # CRITICAL: Reward field for reinforcement learning
        "reward": {
            "dtype": "float32",
            "shape": (1,),
        },
    }

    # Optional velocity and effort
    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [motors],
        }

    # Camera features
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    # Clean existing dataset if needed
    dataset_path = Path(HF_LEROBOT_HOME / repo_id)
    if dataset_path.exists():
        logger.info(f"Removing existing dataset at {dataset_path}")
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=50,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: List[Path]) -> List[str]:
    """Get available cameras from H5 files."""
    with h5py.File(hdf5_files[0], "r") as ep:
        if "/observations/images" in ep:
            return [key for key in ep["/observations/images"].keys() if "depth" not in key]
        return []


def has_velocity(hdf5_files: List[Path]) -> bool:
    """Check if velocity data available."""
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/qvel" in ep


def has_effort(hdf5_files: List[Path]) -> bool:
    """Check if effort data available.""" 
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/observations/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: List[str]) -> dict[str, np.ndarray]:
    """Load images with OpenPI's simple decompression strategy."""
    imgs_per_cam = {}
    
    for camera in cameras:
        if f"/observations/images/{camera}" not in ep:
            continue
            
        camera_data = ep[f"/observations/images/{camera}"]
        uncompressed = camera_data.ndim == 4

        if uncompressed:
            imgs_array = camera_data[:]
        else:
            # Simple sequential decompression (OpenPI approach)
            import cv2
            imgs_array = []
            for data in camera_data:
                img = cv2.imdecode(data, 1)
                imgs_array.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    
    return imgs_per_cam


def find_image_reference_for_frame(frame_idx: int, image_indices: np.ndarray) -> int:
    """Find the best image reference for chunk data - simple and fast."""
    for i in range(len(image_indices) - 1, -1, -1):
        if image_indices[i] <= frame_idx:
            return i
    return 0


def load_raw_episode_data(ep_path: Path, reward_value: float = 0.0):
    """Load episode data with chunk support and reward assignment."""
    with h5py.File(ep_path, "r") as ep:
        # Core data
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])
        
        # Optional data
        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        # Chunk detection (YZY format)
        is_chunk = 'image_indices' in ep and 'chunk_starts' in ep
        image_indices = None
        if is_chunk:
            image_indices = ep['image_indices'][:]
            logger.debug(f"Chunk episode: {ep_path.name} ({len(image_indices)} images, {len(state)} actions)")

        # Load images
        cameras = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        available_cameras = []
        if "/observations/images" in ep:
            available_cameras = [cam for cam in cameras if cam in ep["/observations/images"]]
        
        imgs_per_cam = load_raw_images_per_camera(ep, available_cameras)

        # Create rewards (constant per episode based on folder)
        num_frames = len(state)
        rewards = np.full(num_frames, reward_value, dtype=np.float32)

    return imgs_per_cam, state, action, velocity, effort, rewards, is_chunk, image_indices


def get_resume_info(repo_id: str) -> int:
    """Simple resume: count existing episodes."""
    dataset_path = Path(HF_LEROBOT_HOME / repo_id)
    data_chunk_path = dataset_path / "data" / "chunk-000"
    
    if not data_chunk_path.exists():
        return 0
    
    import glob
    episode_files = glob.glob(str(data_chunk_path / "episode_*.parquet"))
    return len(episode_files)


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: List[Path],
    task: str,
    reward_value: float = 0.0,
    episodes: Optional[List[int]] = None,
    resume: bool = False,
) -> LeRobotDataset:
    """Populate dataset with high performance and chunk support."""
    
    if episodes is None:
        episodes = list(range(len(hdf5_files)))
    
    files_to_process = []
    
    if resume:
        existing_count = get_resume_info(dataset.repo_id)
        if existing_count > 0:
            logger.info(f"Resume: Skipping first {existing_count} episodes")
            episodes = episodes[existing_count:]
        
        files_to_process = [hdf5_files[ep_idx] for ep_idx in episodes if ep_idx < len(hdf5_files)]
    else:
        files_to_process = [hdf5_files[ep_idx] for ep_idx in episodes if ep_idx < len(hdf5_files)]
    
    logger.info(f"Processing {len(files_to_process)} episodes with reward={reward_value}")

    for ep_path in tqdm.tqdm(files_to_process, desc=f"Processing episodes (reward={reward_value})"):
        try:
            imgs_per_cam, state, action, velocity, effort, rewards, is_chunk, image_indices = load_raw_episode_data(
                ep_path, reward_value
            )
            
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                    "reward": np.array([rewards[i]], dtype=np.float32),
                    "task": task,
                }

                # Add optional data
                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]

                # Handle chunk vs normal images
                if is_chunk and image_indices is not None:
                    # Find reference image for this frame
                    img_ref_idx = find_image_reference_for_frame(i, image_indices)
                    for camera, img_array in imgs_per_cam.items():
                        if len(img_array) > img_ref_idx:
                            frame[f"observation.images.{camera}"] = img_array[img_ref_idx]
                else:
                    # Normal: one image per frame
                    for camera, img_array in imgs_per_cam.items():
                        if len(img_array) > i:
                            frame[f"observation.images.{camera}"] = img_array[i]

                dataset.add_frame(frame)

            dataset.save_episode()
            
        except Exception as e:
            logger.error(f"Failed to process {ep_path}: {e}")
            continue

    return dataset


def convert_acrlpd_data(
    input_dir: str,
    repo_id: str, 
    task: str = "fold_box_unified",
    *,
    episodes: Optional[List[int]] = None,
    push_to_hub: bool = False,
    resume: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """
    Main conversion function: Folder-based ACRLPD conversion.
    
    Expected folder structure:
    input_dir/
      score_5/  -> reward = 1.0 (positive)
      score_1/  -> reward = 0.0 (negative)
    """
    
    logger.info(f"Starting ACRLPD conversion: {input_dir} -> {repo_id}")
    
    # Find positive and negative episodes
    positive_dir = Path(input_dir) / "score_5"
    negative_dir = Path(input_dir) / "score_1"
    
    positive_files = []
    negative_files = []
    
    if positive_dir.exists():
        positive_files = sorted(positive_dir.glob("*episode_*.hdf5"))
        logger.info(f"Found {len(positive_files)} positive episodes in {positive_dir}")
    
    if negative_dir.exists():
        negative_files = sorted(negative_dir.glob("*episode_*.hdf5"))
        logger.info(f"Found {len(negative_files)} negative episodes in {negative_dir}")
    
    if not positive_files and not negative_files:
        raise ValueError(f"No episodes found in {input_dir}/score_5 or {input_dir}/score_1")

    # Use first available file to detect features
    sample_file = positive_files[0] if positive_files else negative_files[0]
    all_files = positive_files + negative_files

    # Create dataset
    dataset = create_empty_dataset(
        repo_id,
        robot_type="aloha",
        mode=mode,
        has_velocity=has_velocity([sample_file]),
        has_effort=has_effort([sample_file]),
        dataset_config=dataset_config,
    )

    # Process positive episodes (reward=1.0)
    if positive_files:
        dataset = populate_dataset(
            dataset,
            positive_files,
            task=task,
            reward_value=1.0,
            episodes=episodes,
            resume=resume,
        )
        # After positive, disable resume for negative processing
        resume = False

    # Process negative episodes (reward=0.0)  
    if negative_files:
        dataset = populate_dataset(
            dataset,
            negative_files,
            task=task,
            reward_value=0.0,
            episodes=episodes,
            resume=resume,
        )

    logger.info(f"Conversion completed: {dataset.num_episodes} episodes")

    if push_to_hub:
        logger.info("Pushing to Hub...")
        dataset.push_to_hub()

    return dataset


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s',
        force=True
    )
    # 强制刷新输出
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    tyro.cli(convert_acrlpd_data)