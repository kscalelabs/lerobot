"""
This script demonstrates loading and testing the GPR (General Purpose Robot) 
from KREC format as a Lerobot dataset locally.

Run:
    python examples/13_load_gpr_krec_dataset.py --raw_dir /path/to/krec/files

Run visualize script to check the dataset:
    python lerobot/scripts/visualize_dataset.py \
        --repo-id {REPO_ID} \
        --root .cache/huggingface/lerobot/{REPO_ID} \
        --local-files-only 1 \
        --episode-index 0
        
    python lerobot/scripts/visualize_dataset.py \
        --repo-id gpr_test_krec \
        --root ~/.cache/huggingface/lerobot/gpr_test_krec \
        --local-files-only 1 \
        --episode-index 0
"""
import argparse
import shutil
from pathlib import Path
from pprint import pprint

import decord
import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.gpr_krec_format import \
    from_raw_to_lerobot_format

NUM_ACTUATORS = 5
KREC_VIDEO_WIDTH = 128
KREC_VIDEO_HEIGHT = 128
TOLERANCE_S = 0.03
REPO_ID = "gpr_test_krec"

GPR_FEATURES = {
    "observation.joint_pos": {
        "dtype": "float32",
        "shape": (NUM_ACTUATORS,),
        "names": ["joint_positions"],
    },
    "observation.joint_vel": {
        "dtype": "float32",
        "shape": (NUM_ACTUATORS,),
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
    "observation.images": {
        "dtype": "video",
        "shape": (KREC_VIDEO_HEIGHT, KREC_VIDEO_WIDTH, 3),
        "names": ["frames"],
    },
    "action": {
        "dtype": "float32",
        "shape": (NUM_ACTUATORS,),
        "names": ["joint_commands"],
    },
}


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


def load_video_frame(video_frame_data: dict, video_readers: dict, root_dir: Path) -> torch.Tensor:
    """Load a specific frame from a video file using timestamp information.
    
    Args:
        video_frame_data: Dictionary containing 'path' and 'timestamp' keys
        video_readers: Dictionary mapping video paths to VideoReader objects
        root_dir: Root directory where videos are stored
    
    Returns:
        torch.Tensor: Video frame in (C, H, W) format, normalized to [0,1]
    """
    video_path = root_dir / video_frame_data['path']
    
    # Reuse existing VideoReader or create new one
    if str(video_path) not in video_readers:
        video_readers[str(video_path)] = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    vr = video_readers[str(video_path)]
    
    # Convert timestamp to frame index
    fps = 30  # len(vr) / vr.get_avg_duration()
    frame_idx = int(video_frame_data['timestamp'] * fps)
    
    # Load the specific frame, clamping frame_idx to valid range
    frame_idx = min(max(frame_idx, 0), len(vr) - 1)  # Clamp between 0 and last frame
    frame = vr[frame_idx].asnumpy()
    frame = torch.from_numpy(frame).float()
    frame = frame.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
    frame = frame / 255.0
    
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
        video=True,  # Video data
    )

    # Delete the existing dataset folder if it exists
    dataset_path = Path.home() / ".cache/huggingface/lerobot/" / REPO_ID
    if dataset_path.exists():
        print(f"Deleting existing dataset folder: {dataset_path}")
        shutil.rmtree(dataset_path)

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

    # Add video_readers dictionary to cache VideoReader objects
    video_readers = {}

    episodes = range(len(episode_data_index["from"]))
    for ep_idx in episodes:
        print(f"Processing episode {ep_idx}")
        from_idx = episode_data_index["from"][ep_idx].item()
        to_idx = episode_data_index["to"][ep_idx].item()
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            frame_data = hf_dataset[i]
            video_frame = load_video_frame(
                frame_data["observation.images.camera"], 
                video_readers=video_readers,  # Pass the video_readers dictionary
                root_dir=raw_dir
            )

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
            frame["observation.images"] = np.array(video_frame)
            frame["timestamp"] = frame_data["timestamp"]

            dataset.add_frame(frame)
        print(f"Saving episode {ep_idx}")
        dataset.save_episode(
            task="walk forward",
            encode_videos=True,
        )  # You might want to customize this task description
        print(f"Done saving episode {ep_idx}")

    print("Consolidating dataset...")
    dataset.consolidate()
    print("Done consolidating dataset")
    video_readers.clear()

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
        "observation.images": [0, -1 / dataset.fps],
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
    print(f"\n{dataset[0]['observation.images'].shape=}")  # (4, c, h, w)
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
        print(f"{batch['observation.images'].shape=}")
        print(f"{batch['action'].shape=}")
        break


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
