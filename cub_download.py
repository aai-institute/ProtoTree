import logging
import sys
import tarfile
from pathlib import Path

import gdown

import config as c

URL = "https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45"

cub_tarball_path = c.cub_dir / "CUB_200_2011.tgz"

overwrite = False
extract = True
delete_tar = True


def download_cub():
    if not c.cub_dir.exists():
        c.cub_dir.mkdir(parents=True)
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

    if c.cub_images_dir.exists() and not overwrite:
        log.info(
            f"{c.cub_images_dir} exists and {overwrite=}, skipping download and extraction"
        )
        sys.exit()

    if cub_tarball_path.exists() and not overwrite:
        log.info(f"Skipping download b/c {cub_tarball_path} exists and {overwrite=}")
    else:
        download_cub()

    if extract:
        log.info(f"Extracting CUB data to: {c.cub_dir}")
        with tarfile.open(cub_tarball_path, "r:gz") as tar:
            tar.extractall(c.cub_dir)

        if delete_tar:
            log.info(f"Deleting CUB data tarball: {cub_tarball_path}")
            cub_tarball_path.unlink()

    log.info(f"CUB data is ready to be used in {c.cub_dir}")
