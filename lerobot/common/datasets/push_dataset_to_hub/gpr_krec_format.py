"""
This script loads a GPR dataset from KREC files and converts it to lerobot dataset format.

Example Usage:
    python lerobot/common/datasets/push_dataset_to_hub/gpt_krec_format.py --raw_dir /path/to/krec/files
    
    python lerobot/common/datasets/push_dataset_to_hub/gpt_krec_format.py --raw_dir /home/kasm-user/ali_repos/kmodel/data/datasets/krec_data/dec_3__11_10am_og_krecs_edited/2024-12-03_17-47-30/
"""

import argparse
from pathlib import Path
from pprint import pprint

import os
from datetime import datetime

import h5py
import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value, Image
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from tqdm import tqdm

import krec
from scipy.spatial.transform import Rotation as R

import decord

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
)
from lerobot.common.datasets.utils import hf_transform_to_torch

import time

import shutil
from PIL import Image as PILImage
from lerobot.common.datasets.push_dataset_to_hub.utils import save_images_concurrently

KREC_VIDEO_WIDTH = 640
KREC_VIDEO_HEIGHT = 480

def get_krec_file_type(file_path: str) -> str:
    """Determine if the file is a direct KREC file or MKV-embedded KREC.

    Returns:
        'krec' for .krec files
        'mkv' for .krec.mkv files
        raises RuntimeError for invalid extensions
    """
    if file_path.endswith(".krec"):
        return "krec"
    elif file_path.endswith(".krec.mkv"):
        return "mkv"
    else:
        error_msg = (
            f"Invalid file extension. Expected '.krec' or '.krec.mkv', got: {file_path}"
        )
        raise RuntimeError(error_msg)


def load_krec_direct(krec_file_path: str) -> krec.KRec:
    """Load a KREC file directly."""
    return krec.KRec.load(krec_file_path)


def load_krec_from_mkv(mkv_file_path: str) -> krec.KRec:
    """Load a KREC file from an MKV file into a manually created temp directory."""

    if not os.path.exists(mkv_file_path):
        raise FileNotFoundError(f"File not found: {mkv_file_path}")

    # Create a parent temp directory if it doesn't exist
    parent_temp_dir = os.path.join(os.path.dirname(mkv_file_path), "temp")
    os.makedirs(parent_temp_dir, exist_ok=True)

    # Create timestamped subdirectory inside parent temp directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    temp_dir = os.path.join(parent_temp_dir, f"temp_{timestamp}")
    os.makedirs(temp_dir, exist_ok=True)

    base_name = os.path.basename(mkv_file_path).split(".krec.mkv")[0]
    krec_file_path = os.path.join(temp_dir, f"{base_name}_from_mkv.krec")

    # Extract and load from temp directory
    krec.extract_from_video(mkv_file_path, krec_file_path)
    return krec.KRec.load(krec_file_path)


def load_krec(file_path: str) -> krec.KRec:
    """Smart loader that handles both direct KREC and MKV-embedded KREC files."""
    file_type = get_krec_file_type(file_path)

    if file_type == "krec":
        return load_krec_direct(file_path)
    else:  # file_type == 'mkv'
        return load_krec_from_mkv(file_path)

def convert_quaternion_to_euler(quat):
    """
    Convert Quarternion (xyzw) to Euler angles (rpy) 
    """
    # Normalize
    quat = quat / np.linalg.norm(quat)
    euler = R.from_quat(quat).as_euler('xyz')

    return euler


def check_format(raw_dir) -> bool:
    """Verify KREC files have expected structure"""
    print(f"[DEBUG] Checking format for directory: {raw_dir}")
    krec_paths = list(raw_dir.glob("*.krec.mkv"))
    assert len(krec_paths) > 0, "No KREC files found"
    print(f"[DEBUG] Found {len(krec_paths)} KREC files")

    for krec_path in krec_paths:
        print(f"[DEBUG] Checking file: {krec_path}")
        krec_obj = load_krec_from_mkv(str(krec_path))
        first_frame = krec_obj[0]
        
        # Verify required data exists
        assert len(first_frame.get_actuator_states()) > 0, "No actuator states found"
        assert len(first_frame.get_actuator_commands()) > 0, "No actuator commands found"
        assert first_frame.get_imu_values() is not None, "No IMU values found"


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    start_time = time.time()
    print(f"[TIMING] Starting load_from_raw")
    
    print(f"[DEBUG] Loading raw data from: {raw_dir}")
    krec_files = sorted(raw_dir.glob("*.krec.mkv"))
    num_episodes = len(krec_files)
    print(f"[DEBUG] Found {len(krec_files)} total KREC files")

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    print(f"[DEBUG] Processing episodes: {list(ep_ids)}")

    for ep_idx in tqdm(ep_ids):
        ep_start = time.time()
        print(f"[TIMING] Starting episode {ep_idx}")
        
        ep_path = krec_files[ep_idx]
        print(f"[DEBUG] Processing episode {ep_idx} from file: {ep_path}")
        krec_obj = load_krec_from_mkv(str(ep_path))
        
        # Initialize video reader
        video_reader = decord.VideoReader(str(ep_path), ctx=decord.cpu(0))
        
        num_frames = len(krec_obj)
        first_frame = krec_obj[0]
        num_joints = len(first_frame.get_actuator_states())

        # Initialize numpy arrays instead of torch tensors
        joint_pos = np.zeros((num_frames, num_joints), dtype=np.float32)
        joint_vel = np.zeros((num_frames, num_joints), dtype=np.float32)
        ang_vel = np.zeros((num_frames, 3), dtype=np.float32)
        euler_rotation = np.zeros((num_frames, 3), dtype=np.float32)
        prev_actions = np.zeros((num_frames, num_joints), dtype=np.float32)
        curr_actions = np.zeros((num_frames, num_joints), dtype=np.float32)

        video_frames = None
        if video:
            # Load all frames at once
            video_frames_batch = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
            if video_frames_batch.shape[1:3] != (KREC_VIDEO_HEIGHT, KREC_VIDEO_WIDTH):
                raise ValueError(
                    f"Video frame dimensions {video_frames_batch.shape[1:3]} do not match expected dimensions "
                    f"({KREC_VIDEO_HEIGHT}, {KREC_VIDEO_WIDTH})"
                )

            ep_video_dir = raw_dir / "ep_videos" 
            tmp_imgs_dir = ep_video_dir / "tmp_images"
            if ep_video_dir.exists():
                shutil.rmtree(ep_video_dir)
            ep_video_dir.mkdir(parents=True, exist_ok=True)
            tmp_imgs_dir.mkdir(parents=True, exist_ok=True)

            # Save frames as images and encode to video
            save_images_concurrently(video_frames_batch, tmp_imgs_dir)
            
            # Encode images to mp4 video
            fname = f"camera_episode_{ep_idx:06d}.mp4"
            video_path = ep_video_dir / fname
            encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

            # Clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # Store video frame references
            video_frames = [
                {"path": f"ep_videos/{fname}", "timestamp": i / fps} 
                for i in range(num_frames)
            ]
        
        # Fill data from KREC frames
        for frame_idx, frame in enumerate(krec_obj):
            # Joint positions and velocities
            for j, state in enumerate(frame.get_actuator_states()):
                joint_pos[frame_idx, j] = state.position
                joint_vel[frame_idx, j] = state.velocity

            # Actions (commands)
            for j, cmd in enumerate(frame.get_actuator_commands()):
                curr_actions[frame_idx, j] = cmd.position

            # IMU data
            imu = frame.get_imu_values()
            if imu and imu.gyro:
                ang_vel[frame_idx] = torch.tensor(
                    [imu.gyro.x, imu.gyro.y, imu.gyro.z], dtype=torch.float32
                )
            if imu and imu.quaternion:
                quat = torch.tensor(
                    [imu.quaternion.x, imu.quaternion.y, imu.quaternion.z, imu.quaternion.w]
                )
                curr_euler = convert_quaternion_to_euler(quat)
                euler_rotation[frame_idx] = torch.tensor(curr_euler, dtype=torch.float32)

        # Set previous actions (shifted by 1)
        prev_actions[1:] = curr_actions[:-1]
        prev_actions[0] = curr_actions[0]  # First frame uses same action

        # Create done signal (True for last frame)
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        ep_dict = {
            "observation.state": torch.from_numpy(joint_pos),
            "observation.joint_pos": torch.from_numpy(joint_pos),
            "observation.joint_vel": torch.from_numpy(joint_vel),
            "observation.ang_vel": torch.from_numpy(ang_vel),
            "observation.euler_rotation": torch.from_numpy(euler_rotation),
            "observation.images.camera": video_frames if video else None,
            "prev_actions": torch.from_numpy(prev_actions),
            "action": torch.from_numpy(curr_actions),
            "episode_index": torch.tensor([ep_idx] * num_frames),
            "frame_index": torch.arange(0, num_frames, 1),
            "timestamp": torch.arange(0, num_frames, 1) / fps,
            "next.done": done,
        }
        ep_dicts.append(ep_dict)

        print(f"[TIMING] Episode {ep_idx} took {time.time() - ep_start:.2f} seconds")
    
    print(f"[TIMING] Total load_from_raw took {time.time() - start_time:.2f} seconds")
    print(f"[DEBUG] Concatenating {len(ep_dicts)} episodes")
    data_dict = concatenate_episodes(ep_dicts)
    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    
    return data_dict


def to_hf_dataset(data_dict, video) -> Dataset:
    start_time = time.time()
    print("[TIMING] Starting to_hf_dataset conversion")
    
    print("[DEBUG] Converting to HuggingFace dataset format")
    print(f"[DEBUG] Input data_dict keys: {list(data_dict.keys())}")
    features = {
        "observation.state": Sequence(
            length=data_dict["observation.state"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.joint_pos": Sequence(
            length=data_dict["observation.joint_pos"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.joint_vel": Sequence(
            length=data_dict["observation.joint_vel"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.ang_vel": Sequence(
            length=data_dict["observation.ang_vel"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "observation.euler_rotation": Sequence(
            length=data_dict["observation.euler_rotation"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "prev_actions": Sequence(
            length=data_dict["prev_actions"].shape[1],
            feature=Value(dtype="float32", id=None),
        ),
        "action": Sequence(
            length=data_dict["action"].shape[1], 
            feature=Value(dtype="float32", id=None)
        ),
        "observation.images.camera": VideoFrame() if video else None,
        "episode_index": Value(dtype="int64", id=None),
        "frame_index": Value(dtype="int64", id=None),
        "timestamp": Value(dtype="float32", id=None),
        "next.done": Value(dtype="bool", id=None),
        "index": Value(dtype="int64", id=None),
    }

    print("[DEBUG] Creating HuggingFace dataset")
    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    print(f"[DEBUG] Dataset size: {len(hf_dataset)}")
    print("[DEBUG] Setting transform function")
    hf_dataset.set_transform(hf_transform_to_torch)

    print(f"[TIMING] to_hf_dataset took {time.time() - start_time:.2f} seconds")
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    total_start = time.time()
    print("[TIMING] Starting full conversion process")
    
    print(f"[DEBUG] Starting conversion from raw to LeRobot format")
    print(f"[DEBUG] Parameters:")
    print(f"[DEBUG] - raw_dir: {raw_dir}")
    print(f"[DEBUG] - videos_dir: {videos_dir}")
    print(f"[DEBUG] - fps: {fps}")
    print(f"[DEBUG] - video: {video}")
    print(f"[DEBUG] - episodes: {episodes}")
    print(f"[DEBUG] - encoding: {encoding}")
    check_format(raw_dir)

    if fps is None:
        fps = 50  # Default FPS for your dataset
        print(f"[DEBUG] Using default FPS: {fps}")

    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dict, video)
    print("[DEBUG] Calculating episode data index")
    episode_data_index = calculate_episode_data_index(hf_dataset)

    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    print(f"[DEBUG] Final info: {info}")

    print(f"[TIMING] Total conversion took {time.time() - total_start:.2f} seconds")
    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert GPR KREC dataset to LeRobot format"
    )
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Directory containing raw KREC files"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="data/temp",
        help="Directory for video output (default: data/temp)",
    )
    parser.add_argument(
        "--fps", type=int, default=50, help="Frames per second (default: 50)"
    )
    parser.add_argument(
        "--video", action="store_true", help="Enable video processing (default: False)"
    )

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    videos_dir = Path(args.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print("Converting raw KREC data to LeRobot format...")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir, videos_dir=videos_dir, fps=args.fps, video=args.video
    )
    print("Conversion completed!")
    print("\nDataset info:")
    pprint(hf_dataset)
