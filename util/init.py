import argparse
import os
import pickle
from pathlib import Path
from typing import Union

import torch

from prototree.prototree import ProtoTree


def load_state(directory_path: str, device):
    with open(directory_path + "/tree.pkl", "rb") as f:
        tree = pickle.load(f)
        state = torch.load(directory_path + "/model_state.pth", map_location=device)
        tree.load_state_dict(state)
    return tree


def load_things_into_tree(
    tree: ProtoTree,
    optimizer: torch.optim.Optimizer,
    state_dict_dir_tree: Union[str, Path] = None,
    state_dict_dir_net: Union[str, Path] = None,
    disable_cuda=False,
    freeze_epochs=0,
    disable_derivative_free_leaf_optim=False,
    scheduler: torch.optim.lr_scheduler.MultiStepLR = None,
):
    """

    :param tree:
    :param optimizer:
    :param state_dict_dir_tree:
    :param state_dict_dir_net:
    :param scheduler: has to be passed and will be updated to reflect the amount of epochs already trained
    :param disable_cuda:
    :param freeze_epochs:
    :param disable_derivative_free_leaf_optim:
    :return:
    """
    epoch = 0

    # NOTE: TRAINING FURTHER FROM A CHECKPOINT DOESN'T SEEM TO WORK CORRECTLY.
    # EVALUATING A TRAINED PROTOTREE FROM A CHECKPOINT DOES WORK.
    if state_dict_dir_tree:
        tree, epoch = _load_tree_and_reset_scheduler(
            state_dict_dir_tree,
            disable_cuda,
            disable_derivative_free_leaf_optim,
            freeze_epochs,
            optimizer,
            scheduler,
        )
    # load pretrained conv network
    elif state_dict_dir_net:
        _load_weights_to_tree(tree, state_dict_dir_net)
    else:
        init_tree_weights(tree)
    return tree, epoch


def init_tree_weights(tree: ProtoTree):
    with torch.no_grad():
        tree.init_prototype_layer()
        tree.add_on.apply(_xavier_on_conv)


def _load_weights_to_tree(tree: ProtoTree, state_dict_dir_net: Union[str, Path]):
    state_dict_dir_net = Path(state_dict_dir_net)
    # TODO: why here without no_grad?
    tree.init_prototype_layer()
    # strict is False so when loading pretrained model, ignore the linear classification layer
    tree.net.load_state_dict(
        torch.load(state_dict_dir_net / "model_state.pth"), strict=False
    )
    # TODO: We load this from the same file?!
    tree.add_on.load_state_dict(
        torch.load(state_dict_dir_net / "model_state.pth"), strict=False
    )


def _load_tree_and_reset_scheduler(
    state_dict_dir_tree: Union[str, Path],
    disable_cuda: bool,
    disable_derivative_free_leaf_optim: bool,
    freeze_epochs: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.MultiStepLR,
):
    state_dict_dir_tree = Path(state_dict_dir_tree)
    if not disable_cuda and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")
    # TODO: remove hardcoded stuff
    tree = torch.load(state_dict_dir_tree / "model.pth", map_location=device)
    # TODO: this can be buggy, it seems. Horrible way of recovering the epoch
    epoch = int(state_dict_dir_tree.split("epoch_")[-1]) + 1
    print("Train further from epoch: ", epoch, flush=True)
    optimizer.load_state_dict(
        torch.load(state_dict_dir_tree / "optimizer_state.pth", map_location=device)
    )
    if epoch > freeze_epochs:
        for parameter in tree.net.parameters():
            parameter.requires_grad = True
    if not disable_derivative_free_leaf_optim:
        for leaf in tree.leaves:
            # TODO: mutating private fields
            leaf.dist_params.requires_grad = False

    if (state_dict_dir_tree / "scheduler_state.pth").exists():
        # TODO: wtf? mutating fields, including private ones...
        scheduler.last_epoch = epoch - 1
        scheduler._step_count = epoch

    return tree, epoch


def _xavier_on_conv(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(
            m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
        )


def _kaiming_on_conv(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
