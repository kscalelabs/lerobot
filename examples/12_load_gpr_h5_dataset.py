"""
This script demonstrates loading and testing the GPR (General Purpose Robot) from HDF5 files as a Lerobot dataset locally. (not working yet)

Example Usage:
    python examples/12_load_gpr_dataset.py --raw_dir /path/to/h5/files
"""

from pathlib import Path
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from pprint import pprint
import shutil
import argparse
import numpy as np

from lerobot.common.datasets.push_dataset_to_hub.gpr_h5_format import (
    from_raw_to_lerobot_format,
)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

GPR_FEATURES = {
    "observation.joint_pos": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["joint_positions"],
    },
    "observation.joint_vel": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["joint_velocities"],
    },
    "observation.ang_vel": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["angular_velocity"],
    },
    "observation.euler_rotation": {
        "dtype": "float32",
        "shape": (3,),
        "names": ["euler_angles"],
    },
    "observation.video": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["frames"],
    },
    "action": {
        "dtype": "float32",
        "shape": (10,),
        "names": ["joint_commands"],
    },
}

TOLERANCE_S = 0.03
REPO_ID = "gpr_test_h5"


def generate_test_video_frame(width: int, height: int, frame_idx: int) -> Image:
    """
    Generates a dummy video frame with a white square that moves based on the frame index.
    :param width: Width of the video frame.
    :param height: Height of the video frame.
    :param frame_idx: Index of the frame to determine the position of the white square.
    :return: PIL Image object.
    """
    frame = Image.new("RGB", (width, height), "black")  # Create a black frame
    draw = ImageDraw.Draw(frame)
    square_size = min(width, height) // 4
    x = (frame_idx * 10) % (width - square_size)
    y = (frame_idx * 10) % (height - square_size)
    draw.rectangle(
        [x, y, x + square_size, y + square_size], fill="white"
    )  # Add a white square that moves
    return frame


def test_gpr_dataset(raw_dir: Path, videos_dir: Path, fps: int):
    # Setup paths
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Convert raw data to LeRobot format
    print("Converting raw data to LeRobot format...")
    hf_dataset, episode_data_index, info = from_raw_to_lerobot_format(
        raw_dir=raw_dir,
        videos_dir=videos_dir,
        fps=fps,  # Your simulation fps
        video=False,  # No video data
    )

    # Delete the existing dataset folder if it exists
    dataset_path = Path.home() / ".cache/huggingface/lerobot/" / REPO_ID
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)

    # import pdb; pdb.set_trace()

    # Create dataset instance
    print("\nCreating dataset...")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=fps,
        features=GPR_FEATURES,
        tolerance_s=TOLERANCE_S,  # timestep indexing tolerance in seconds based on fps
        use_videos=True
    )

    print("Camera keys:", dataset.meta.camera_keys)

    episodes = range(len(episode_data_index["from"]))
    for ep_idx in episodes:
        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            frame_data = hf_dataset[i]

            video_frame = generate_test_video_frame(
                width=640, height=480, frame_idx=frame_idx
            )  # dummy images

            frame = {
                key: frame_data[key].numpy().astype(np.float32)
                for key in [
                    "observation.joint_pos",
                    "observation.joint_vel",
                    "observation.ang_vel",
                    "observation.euler_rotation",
                    "action",
                ]
            }
            frame["observation.video"] = np.array(video_frame)
            frame["timestamp"] = frame_data["timestamp"]

            dataset.add_frame(frame)

        dataset.save_episode(
            task="walk forward",
            encode_videos=True,
        )  # You might want to customize this task description

    dataset.consolidate()

    #########################################################
    # From this point on its copy paste from lerobot/examples/1_load_lerobot_dataset.py

    # And see how many frames you have:
    print(f"Selected episodes: {dataset.episodes}")
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # Or simply load the entire dataset:
    print(f"Number of episodes selected: {dataset.num_episodes}")
    print(f"Number of frames selected: {dataset.num_frames}")

    # The previous metadata class is contained in the 'meta' attribute of the dataset:
    print(dataset.meta)

    # LeRobotDataset actually wraps an underlying Hugging Face dataset
    # (see https://huggingface.co/docs/datasets for more information).
    print(dataset.hf_dataset)

    # LeRobot datasets also subclasses PyTorch datasets so you can do everything you know and love from working
    # with the latter, like iterating through the dataset.
    # The __getitem__ iterates over the frames of the dataset. Since our datasets are also structured by
    # episodes, you can access the frame indices of any episode using the episode_data_index. Here, we access
    # frame indices associated to the first episode:
    episode_index = 0
    from_idx = dataset.episode_data_index["from"][episode_index].item()
    to_idx = dataset.episode_data_index["to"][episode_index].item()

    # Then we grab all the image frames from the first camera:
    # import pdb; pdb.set_trace()
    camera_key = dataset.meta.camera_keys[0]
    frames = [dataset[idx][camera_key] for idx in range(from_idx, to_idx)]

    # The objects returned by the dataset are all torch.Tensors
    print(f"type(frames[0])={type(frames[0])}")
    print(f"frames[0].shape={frames[0].shape}")

    # Since we're using pytorch, the shape is in pytorch, channel-first convention (c, h, w).
    # We can compare this shape with the information available for that feature
    print(f"dataset.features camera keys")
    pprint(dataset.features[camera_key])
    # # In particular:
    print(f"camera key shape {dataset.features[camera_key]['shape']}")
    # The shape is in (h, w, c) which is a more universal format.

    # For many machine learning applications we need to load the history of past observations or trajectories of
    # future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
    # differences with the current loaded frame. For instance:
    delta_timestamps = {
        # 0 is current frame, -1/fps is 1 frame in the past
        "observation.joint_pos": [0, -1 / dataset.fps],
        "observation.joint_vel": [0, -1 / dataset.fps],
        "observation.ang_vel": [0, -1 / dataset.fps],
        "observation.euler_rotation": [0, -1 / dataset.fps],
        "observation.video": [0, -1 / dataset.fps],
        # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
        "action": [t / dataset.fps for t in range(64)],
    }
    # Note that in any case, these delta_timestamps values need to be multiples of (1/fps) so that added to any
    # timestamp, you still get a valid timestamp.
    # local_files_only=True to load from local cache
    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        delta_timestamps=delta_timestamps,
        local_files_only=True,
        tolerance_s=TOLERANCE_S,
    )
    print(f"\n{dataset[0]['observation.video'].shape=}")  # (4, c, h, w)
    print(f"{dataset[0]['observation.joint_pos'].shape=}")  # (6, c)
    print(f"{dataset[0]['action'].shape=}\n")  # (64, c)

    # Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
    # PyTorch datasets.
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=32,
        shuffle=True,
    )

    for batch in dataloader:
        print(f"{batch['observation.joint_pos'].shape=}")
        print(f"{batch['observation.joint_vel'].shape=}")
        print(f"{batch['observation.ang_vel'].shape=}")
        print(f"{batch['observation.euler_rotation'].shape=}")
        print(f"{batch['observation.video'].shape=}")
        print(f"{batch['action'].shape=}")
        break
    
"""
# CHANGE REPO ID
python lerobot/scripts/visualize_dataset.py \
    --repo-id {REPO_ID} \
    --root /home/kasm-user/.cache/huggingface/lerobot/{REPO_ID} \
    --local-files-only 1 \
    --episode-index 0
    
python lerobot/scripts/visualize_dataset.py \
    --repo-id gpr_test_h5 \
    --root /home/kasm-user/.cache/huggingface/lerobot/gpr_test_h5 \
    --local-files-only 1 \
    --episode-index 0

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and test GPR dataset")
    parser.add_argument(
        "--raw_dir", type=str, required=True, help="Directory containing raw HDF5 files"
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

    args = parser.parse_args()

    test_gpr_dataset(
        raw_dir=Path(args.raw_dir), videos_dir=Path(args.videos_dir), fps=args.fps
    )
