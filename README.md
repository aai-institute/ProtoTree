# ProtoTrees

Refactored version of https://github.com/M-Nauta/ProtoTree and parts of https://github.com/cfchen-duke/ProtoPNet to make them more modular and easier to use.
This will probably be turned into a Python package and moved to a new repository.

## Setup
### For all datasets
1. Create a Python >=3.10 environment.
2. Install requirements from `requirements.txt` and `requirements-download.txt` (e.g. `pip install -r requirements.txt`).
3. Install [Graphviz](https://graphviz.org/). With the current code you need to be able to call `dot` from the terminal.
4. You can train the tree model and see its performance on the test set with `python train_prototree.py`. 

### For CUB dataset
1. Run `cub_download.py`.
2. Run `cub_preprocess.py`.
3. (Optional, but recommended) Download a [ResNet50 pretrained on iNaturalist2017](https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU) (filename on Google Drive: `BBN.iNaturalist2017.res50.180epoch.best_model.pth`) and place it in the folder `src/features/state_dicts`.

### For development
Currently, all these steps are only done manually on a development machine. We should set up a pipeline that does these things automatically and reproducibly.
1. Install requirements from `requirements-dev.txt`.
2. You can run tests with the command `pytest`.
3. You can lint the code with `black src` and `black tests`.
4. You can check types with `MYPYPATH=src mypy src --explicit-package-bases`. Note that on the first run it will be helpful to
