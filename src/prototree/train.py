import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree

log = logging.getLogger(__name__)


def train_epoch(
    model: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    progress_desc: str = "Train Epoch",
) -> dict:
    n_batches = len(train_loader)

    tqdm_loader = tqdm(
        train_loader,
        desc=progress_desc,
    )
    tqdm_loader.update()  # Stops earlier logging appearing after tqdm starts showing progress.
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    for batch_num, (x, y) in enumerate(tqdm_loader):
        model.train()
        optimizer.zero_grad()
        x, y = x.to(model.device), y.to(model.device)
        logits, node_to_prob, predicting_leaves = model.forward(x)
        loss = F.nll_loss(logits, y)
        loss.backward()
        optimizer.step()

        model.tree_section.update_leaf_distributions(y, logits.detach(), node_to_prob)

        model.eval()
        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / len(x)
        tqdm_loader.set_postfix_str(f"batch: loss={loss.item():.5f}, {acc=:.5f}")
        total_loss += loss.item()
        total_acc += acc

        if (
            batch_num == n_batches - 1
        ):  # TODO: Hack due to https://github.com/tqdm/tqdm/issues/1369
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            tqdm_loader.set_postfix_str(
                f"average: loss={avg_loss:.5f}, acc={avg_acc:.5f}"
            )

    return {
        "loss": avg_loss,
        "train_accuracy": avg_acc,
    }


@dataclass
class NonlinearOptimParams:
    optim_type: Literal["SGD", "Adam", "AdamW"]
    backbone_name: str
    momentum: float
    weight_decay: float
    lr: float
    lr_block: float
    lr_backbone: float
    freeze_epochs: int
    dataset: str  # TODO: We shouldn't have dataset specific stuff here.


@dataclass
class NonlinearSchedulerParams:
    optim_params: NonlinearOptimParams
    milestones: list[int]
    gamma: float


def get_nonlinear_scheduler(model: ProtoTree, params: NonlinearSchedulerParams):
    optimizer, params_to_freeze, params_to_train = get_nonlinear_optimizer(
        model, params.optim_params
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=params.milestones, gamma=params.gamma
    )

    return (
        optimizer,
        scheduler,
        params.optim_params.freeze_epochs,
        params_to_freeze,
        params_to_train,
    )


def get_nonlinear_optimizer(
    model: ProtoTree, optim_params: NonlinearOptimParams
) -> tuple[torch.optim.Optimizer, list[Parameter], list[Parameter]]:
    """
    :return: the optimizer, parameter set that can be frozen, and parameter set of the net that will be trained
    """
    params_to_freeze = []
    params_to_train = []

    dist_params = []
    for name, param in model.named_parameters():
        # TODO: what is this?
        if "dist_params" in name:
            dist_params.append(param)

    # set up optimizer
    if "resnet50_inat" in optim_params.backbone_name or (
        "resnet50" in optim_params.backbone_name and optim_params.dataset == "CARS"
    ):
        # TODO: Seems to defeat the point of encapsulation if we're accessing the backbone directly.
        for name, param in model.proto_base.backbone.named_parameters():
            # TODO: improve this logic
            if "layer4.2" not in name:
                params_to_freeze.append(param)
            else:
                params_to_train.append(param)

        param_list = [
            {
                "params": params_to_freeze,
                "lr": optim_params.lr_backbone,
                "weight_decay_rate": optim_params.weight_decay,
            },
            {
                "params": params_to_train,
                "lr": optim_params.lr_block,
                "weight_decay_rate": optim_params.weight_decay,
            },
            {
                # TODO: Seems to defeat the point of encapsulation if we're accessing the add_on directly.
                "params": model.proto_base.add_on.parameters(),
                "lr": optim_params.lr_block,
                "weight_decay_rate": optim_params.weight_decay,
            },
            {
                # TODO: Seems to defeat the point of encapsulation if we're accessing the prototype_layer directly.
                "params": model.proto_base.prototype_layer.parameters(),
                "lr": optim_params.lr,
                "weight_decay_rate": 0,
            },
        ]

    if optim_params.optim_type == "SGD":
        # TODO: why no momentum for the prototype layer?
        # add momentum to the first three entries of paramlist
        for i in range(3):
            param_list[i]["momentum"] = optim_params.momentum
        # TODO: why pass momentum here explicitly again? Which one is taken?
        optimizer = torch.optim.SGD(
            param_list, lr=optim_params.lr, momentum=optim_params.momentum
        )
    elif optim_params.optim_type == "Adam":
        optimizer = torch.optim.Adam(param_list, lr=optim_params.lr, eps=1e-07)
    elif optim_params.optim_type == "AdamW":
        optimizer = torch.optim.AdamW(
            param_list,
            lr=optim_params.lr,
            eps=1e-07,
            weight_decay=optim_params.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optim_params.optim_type}. Supported optimizers are SGD, Adam, and AdamW."
        )
    return optimizer, params_to_freeze, params_to_train
