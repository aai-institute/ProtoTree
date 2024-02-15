import logging
from argparse import Namespace
from pathlib import Path
import json
import yaml

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from src.util.data import get_dataloader
from src.core.models import ProtoTree, ProtoPNet
from src.core.eval import eval_model
from src.core.optim import (
    NonlinearOptimParams,
    NonlinearSchedulerParams,
)
from src.util.args import get_args
from src.util.score import globale_scores

from src.visualize.create.patches import save_patch_visualizations
from src.visualize.create.tree import save_tree_visualization
from src.visualize.create.explanation.prototypes import save_prototypes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_prototree")


def train_prototree(config: dict):
    # data and paths
    dataset = config["dataset"]
    output_dir = Path(config["output_dir"]).resolve()  # Absolute path makes it easier for Graphviz to find images.

    # training hardware
    #milestones = config.ge.milestones_list # TODOs understand what they are
    gamma = config["gamma"]

    # Optimizer args
    optim_type = config["optimizer"]
    batch_size = config["batch_size"]
    lr_main = config["lr_main"]
    lr_backbone = config["lr_backbone"]
    momentum = config["momentum"]
    weight_decay_main = config["weight_decay_main"]
    weight_decay_backbone = config["weight_decay_backbone"]
    gradient_leaf_opt = config["gradient_leaf_opt"]
    milestones = config["milestones"]

    # Training loop args
    disable_cuda = config["disable_cuda"]
    epochs = config["epochs"]
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    freeze_epochs = config["freeze_epochs"]
    project_from_epoch = config["project_from_epoch"]

    if project_from_epoch >= 0:
        project_epochs = set((i for i in range(project_from_epoch, epochs)))
    else:
        project_epochs = set()

    # prototree specifics
    leaf_pruning_multiplier = config["leaf_pruning_multiplier"]

    # Model checkpoint specifics
    every_n_epochs = config["every_n_epochs"]
    save_top_k = config["save_top_k"]
    
    # Architecture args
    model_type = config["model_type"]
    backbone_name = config["backbone"]
    pretrained = not config["disable_pretrained"]
    h_proto = config["W1"]
    w_proto = config["H1"]
    channels_proto = config["num_features"]
    depth =config["depth"]

    # Prototype explanation args
    explain_prototypes = config["explain"]
    img_modifications = config["img_modifications"]

    log.info(f"Training and testing ProtoTree with {config=}.")

    # PREPARE DATA
    train_dir = Path(config["dataset_dir"]) / config["train_dir"]
    val_dir = Path(config["dataset_dir"]) / config["val_dir"]
    img_size = config["img_size"]

    train_loader = get_dataloader(dataset_dir=train_dir, img_size=img_size, augment=True, train=True, loader_batch_size=batch_size)
    val_loader = get_dataloader(dataset_dir=val_dir, img_size=img_size, augment=False)

    class_names = train_loader.dataset.classes
    num_classes = len(class_names)
    log.info(f"Num classes: {num_classes}")

    # PREPARE PRUNING
    # Leaves that didn't learn anything will have a distribution close to torch.ones(num_classes) / num_classes, so we
    # prune leaves where no probability is fairly close to the "no-learning" value of 1 / num_classes. The
    # multiplication by leaf_pruning_multiplier determines what's considered close enough to "no-learning".
    # TODO: Perhaps we should instead tune this threshold with a search algorithm.
    leaf_pruning_threshold = leaf_pruning_multiplier / num_classes

    n_training_batches = len(train_loader)
    leaf_opt_ewma_alpha = 1 / n_training_batches

    nonlinear_optim_params = NonlinearOptimParams(
        optim_type=optim_type,
        backbone_name=backbone_name,
        momentum=momentum,
        lr_main=lr_main,
        lr_backbone=lr_backbone,
        weight_decay_main=weight_decay_main,
        weight_decay_backbone=weight_decay_backbone,
        freeze_epochs=freeze_epochs,
        dataset=dataset,
    )
    nonlinear_scheduler_params = NonlinearSchedulerParams(
        optim_params=nonlinear_optim_params, milestones=milestones, gamma=gamma
    )

    # PREPARE MODEL
    match model_type:
        case "prototree":
            model = ProtoTree(
                h_proto=h_proto,
                w_proto=w_proto,
                channels_proto=channels_proto,
                num_classes=num_classes,
                depth=depth,
                leaf_pruning_threshold=leaf_pruning_threshold,
                leaf_opt_ewma_alpha=leaf_opt_ewma_alpha,
                project_epochs=project_epochs,
                nonlinear_scheduler_params=nonlinear_scheduler_params,
                gradient_leaf_opt=gradient_leaf_opt,
                backbone_name=backbone_name,
                pretrained=pretrained,
            )
        case "protopnet":
            model = ProtoPNet(
                h_proto=h_proto,
                w_proto=w_proto,
                channels_proto=channels_proto,
                num_classes=num_classes,
                prototypes_per_class=10,
                project_epochs=project_epochs,
                nonlinear_scheduler_params=nonlinear_scheduler_params,
                backbone_name=backbone_name,
                pretrained=pretrained,
            )
        case _:
            raise ValueError(f"Unknown model type {model_type}.")

    # TRAIN
    log.info("Starting training.")
    checkpoint_callback = ModelCheckpoint(dirpath="output",
                                          filename="{epoch}-{step}-{Val acc:.2f}", monitor="Val acc",
                                          save_last=True, every_n_epochs=every_n_epochs, save_top_k=save_top_k) 
    ckpt_output_dir = Path("./output") / model_type
    trainer = pl.Trainer(
        accelerator="cpu" if disable_cuda else "auto",
        detect_anomaly=False,
        max_epochs=epochs,
        limit_val_batches=n_training_batches // 5,
        devices=1,  # TODO: Figure out why the model doesn't work on multiple devices.
        default_root_dir=ckpt_output_dir
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log.info("Finished training.")
    log.info(f"{checkpoint_callback.best_model_path=}")

    # SAVE PROTOTYPES INFO FOR LATER VISUALIZATION
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches" / model_type
    score_dir = output_dir / "scores" / model_type
    save_patch_visualizations(model.proto_patch_matches, patches_dir, save_as_json=True)

    # COMPUTE AND SAVE GLOBAL SCORES
    # TODO: proto explanation implemented for batch_size = 1 (maybe we want to change it)
    if explain_prototypes:
        dataloader = get_dataloader(dataset_dir=train_dir, img_size=img_size, augment=False, explain=explain_prototypes, modifications=img_modifications, loader_batch_size=1, num_workers=4)
        _, proto_expl = eval_model(model, dataloader, explain=True)
        global_expl = globale_scores(scores=proto_expl, out_dir=score_dir)
    else:
        global_expl = None

    # SAVE VISUALIZATIONS
    with open(patches_dir / "proto_info.json") as f:
        prototypes_info = json.load(f)

    match model_type:
        case "prototree":
            tree_dir = vis_dir/ "tree"
            with open(patches_dir / "proto_info.json") as f:
                prototypes_info = json.load(f)
            save_tree_visualization(model, prototypes_info, tree_dir, class_names, global_scores=global_expl)
        case "protopnet":
            save_prototypes(proto_info=prototypes_info, img_size=img_size, global_expl=global_expl)
        case _:
            raise ValueError(f"Unknown model type {model_type}.")



if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            log.warning(
                "lovely_tensors not installed, not monkey patching. "
                "For more efficient debugging, we recommend installing it with `pip install lovely-tensors`."
            )

    parsed_args = vars(get_args())
    parsed_args =  {k: v for k, v in parsed_args.items() if v is not None}

    with open(parsed_args["config_file"], "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    config.update(parsed_args)

    train_prototree(config)
