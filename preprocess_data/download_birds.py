import logging
import sys
import tarfile
from pathlib import Path

import gdown

URL = "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"

data_dir = Path("data")
cub_dir = data_dir / "CUB_200_2011"
cub_tarball_path = cub_dir / "CUB_200_2011.tgz"

overwrite = False
extract = True
delete_tar = True


def download_cub():
    if not cub_dir.exists():
        cub_dir.mkdir(parents=True)
    if not cub_tarball_path.exists() or overwrite:
        log.info(f"Downloading CUB_200_2011 to {cub_tarball_path}")
        gdown.download(URL, str(cub_tarball_path), quiet=False)
    else:
        log.info(f"Using existing CUB_200_2011 at {cub_tarball_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(Path(__file__).name.rstrip(".py"))

    if delete_tar and not extract:
        raise ValueError(f"Passing {delete_tar=} when {extract=} is disallowed")

    if (cub_dir / "images").exists() and not overwrite:
        log.info(
            f"{cub_dir/'images'} exists and {overwrite=}, skipping download and extraction"
        )
        sys.exit()

    if cub_tarball_path.exists() and not overwrite:
        log.info(f"Skipping download b/c {cub_tarball_path} exists and {overwrite=}")
    else:
        download_cub()

    if extract:
        log.info(f"Extracting CUB data to: {cub_dir}")
        with tarfile.open(cub_tarball_path, "r:gz") as tar:
            tar.extractall(cub_dir)

        if delete_tar:
            log.info(f"Deleting CUB data tarball: {cub_tarball_path}")
            cub_tarball_path.unlink()

    log.info(f"CUB data is ready to be used in {cub_dir}")
