from pathlib import Path

data_dir = Path("data")
cub_dir = data_dir / "CUB_200_2011"
cub_images_dir = cub_dir / "images"

dataset_dir = cub_dir / "dataset"

train_dir = dataset_dir / "train_corners"
val_dir = dataset_dir / "train_crop"
test_dir = dataset_dir / "test_crop"
