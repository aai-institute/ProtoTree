import logging
from argparse import Namespace
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from src.core.models import ProtoTree, ProtoPNet
from src.core.eval import eval_model, single_leaf_eval
from src.core.optim import (
    NonlinearOptimParams,
    NonlinearSchedulerParams,
)
from src.util.args import get_args
from src.util.data import get_dataloaders
from src.visualize.create.explanation.decision_flows import (
    save_decision_flow_visualizations,
)
from src.visualize.create.explanation.multi_patch import save_multi_patch_visualizations
from src.visualize.create.patches import save_patch_visualizations
from src.visualize.create.tree import save_tree_visualization
from src.visualize.prepare.explanations import data_explanations

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_prototree")


# TODO: remove dependency on args everywhere
def train_prototree(args: Namespace):
    # data and paths
    dataset = args.dataset
    output_dir = Path(
        args.output_dir
    ).resolve()  # Absolute path makes it easier for Graphviz to find images.

    # training hardware
    milestones = args.milestones_list
    gamma = args.gamma

    # Optimizer args
    optim_type = args.optimizer
    batch_size = args.batch_size
    lr_main = args.lr_main
    lr_backbone = args.lr_backbone
    momentum = args.momentum
    weight_decay_main = args.weight_decay_main
    weight_decay_backbone = args.weight_decay_backbone
    gradient_leaf_opt = args.gradient_leaf_opt

    # Training loop args
    disable_cuda = args.disable_cuda
    epochs = args.epochs
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    freeze_epochs = args.freeze_epochs

    if args.project_from_epoch >= 0:
        project_epochs = set((i for i in range(args.project_from_epoch, epochs)))
    else:
        project_epochs = set()

    # prototree specifics
    leaf_pruning_multiplier = args.leaf_pruning_multiplier

    # Architecture args
    model_type = args.model_type
    backbone_name = args.backbone
    pretrained = not args.disable_pretrained
    h_proto = args.H1
    w_proto = args.W1
    channels_proto = args.num_features
    depth = args.depth

    log.info(f"Training and testing ProtoTree with {args=}.")

    # PREPARE DATA
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=4
    )

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
    # TODO: maybe put arguments of ModelCheckpoint in args. So the saving can be custom
    checkpoint_callback = ModelCheckpoint(dirpath="output",
                                          filename="{epoch}-{step}-{Val acc:.2f}", monitor="Val acc", #Val avg acc
                                          save_last=True, every_n_epochs=2, save_top_k=1) 
    trainer = pl.Trainer(
        accelerator="cpu" if disable_cuda else "auto",
        detect_anomaly=False,
        max_epochs=epochs,
        limit_val_batches=n_training_batches // 5,
        devices=1,  # TODO: Figure out why the model doesn't work on multiple devices.
        default_root_dir=f"output_{model_type}"
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    log.info("Finished training.")
    checkpoint_callback.best_model_path

    if DEBUG:
        # Load the previously saved model and check if the leaves parameters are well saved
        import numpy as np
        leaves_paramas = np.array([leave.dist_params.cpu().numpy() for leave in model.tree_section.leaves])
        np.save(trainer.checkpoint_callback.dirpath, leaves_paramas)
        #model = ProtoTree.load_from_checkpoint(f"{trainer.logger.root_dir}/version_{trainer.logger.version}/checkpoints/epoch={}-step={}.ckpt")
    # EVALUATE AND ANALYSE TRAINED TREE
    model = model.eval()

    if model_type == "prototree":
        model.log_state()
        model.prune(leaf_pruning_threshold)
        pruned_acc = eval_model(model, test_loader)
        log.info(f"\nTest acc. after pruning: {pruned_acc:.3f}")
        model.log_state()
        single_leaf_eval(model, test_loader, "Pruned")

        def explanations_provider():
            return data_explanations(
                model, test_loader, class_names
            )  # This is lazy to enable iterator reuse.

        model.print()

    # SAVE VISUALIZATIONS
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches"
    save_patch_visualizations(model.proto_patch_matches, patches_dir)
    if model_type == "prototree":
        save_tree_visualization(model, patches_dir, vis_dir / "tree", class_names)
        save_multi_patch_visualizations(
            explanations_provider(), vis_dir / "explanations"
        )
        save_decision_flow_visualizations(
            explanations_provider(), patches_dir, vis_dir / "explanations"
        )


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

    parsed_args = get_args()
    train_prototree(parsed_args)
