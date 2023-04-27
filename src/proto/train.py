import logging
from dataclasses import dataclass
from typing import Literal

import torch
import torch.optim
import torch.utils.data
from torch.nn import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

log = logging.getLogger(__name__)


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


def get_nonlinear_scheduler(
    model, params: NonlinearSchedulerParams
) -> tuple[list[Optimizer], list[MultiStepLR]]:
    optimizer = get_nonlinear_optimizer(model, params.optim_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=params.milestones, gamma=params.gamma
    )

    # TODO(Hack): It's difficult to add these extra items in a way that Lightning will accept them. Do we need to
    #  extend the scheduler class? That would be quite a lot of boilerplate/abstraction to do something so simple.
    scheduler.freeze_epochs = params.optim_params.freeze_epochs

    return [optimizer], [scheduler]


def get_nonlinear_optimizer(
    model, optim_params: NonlinearOptimParams
) -> torch.optim.Optimizer:
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
        # TODO: Why no momentum for the prototype layer?
        #  Add momentum to the first three entries of paramlist
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

    # TODO(Hack): It's difficult to add these extra items in a way that Lightning will accept them. Do we need to
    #  extend the optimizer class?  That would be quite a lot of boilerplate/abstraction to do something so simple.
    optimizer.params_to_freeze = params_to_freeze
    optimizer.params_to_train = params_to_train
    return optimizer


def freezable_step(scheduler: MultiStepLR, current_epoch: int, params_to_freeze: list):
    # TODO: Extend the scheduler class and make this a method? It seems hard to do this generically in a way that
    #  doesn't conflict with the restrictive Lightning API.
    if current_epoch > 0:
        scheduler.step()

    maybe_freeze(scheduler.freeze_epochs, current_epoch, params_to_freeze)


def maybe_freeze(freeze_epochs: int, current_epoch: int, params_to_freeze: list):
    # TODO: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html ?
    if freeze_epochs > 0:
        if current_epoch == 0:
            log.info(f"Freezing network for {freeze_epochs} epochs.")
            for param in params_to_freeze:
                param.requires_grad = False
        elif current_epoch == freeze_epochs + 1:
            log.info(f"Unfreezing network on epoch {current_epoch}.")
            for param in params_to_freeze:
                param.requires_grad = True
