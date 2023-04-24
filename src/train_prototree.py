from argparse import Namespace
from pathlib import Path
from random import randint
import logging

import torch

from prototree.eval import eval_model, single_leaf_eval
from prototree.models import ProtoTree
from prototree.node import InternalNode
from visualize.create.explanation.decision_flows import (
    save_decision_flow_visualizations,
)
from visualize.create.explanation.multi_patch import save_multi_patch_visualizations
from visualize.prepare.explanations import data_explanations
from visualize.prepare.matches import node_patch_matches
from prototree.projection import project_prototypes
from prototree.prune import prune_unconfident_leaves
from prototree.train import train_epoch, get_nonlinear_optimizer, NonlinearOptimParams
from visualize.create.patches import save_patch_visualizations
from util.args import get_args
from util.data import get_dataloaders
from visualize.create.tree import save_tree_visualization

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
    lr = args.lr
    lr_block = args.lr_block
    lr_net = args.lr_net
    momentum = args.momentum
    weight_decay = args.weight_decay

    # Training loop args
    disable_cuda = args.disable_cuda
    epochs = args.epochs
    evaluate_each_epoch = 5
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    freeze_epochs = args.freeze_epochs

    # prototree specifics
    leaf_pruning_multiplier = args.leaf_pruning_multiplier

    # Architecture args
    backbone = args.backbone
    pretrained = not args.disable_pretrained
    h_proto = args.H1
    w_proto = args.W1
    channels_proto = args.num_features
    depth = args.depth

    log.info(f"Training and testing ProtoTree with {args=}.")

    # PREPARE DATA
    device = get_device(disable_cuda)
    pin_memory = "cuda" in device.type
    train_loader, project_loader, test_loader = get_dataloaders(
        pin_memory=pin_memory,
        batch_size=batch_size,
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

    # PREPARE MODEL
    model = ProtoTree(
        h_proto=h_proto,
        w_proto=w_proto,
        channels_proto=channels_proto,
        num_classes=num_classes,
        depth=depth,
        leaf_pruning_threshold=leaf_pruning_threshold,
        leaf_opt_ewma_alpha=leaf_opt_ewma_alpha,
        backbone_net=backbone,
        pretrained=pretrained,
    )
    log.info(
        f"Max depth {depth}, so {model.tree_section.num_internal_nodes} internal nodes and {model.tree_section.num_leaves} leaves."
    )
    log.info(f"Running on {device=}")
    model = model.to(device)

    nonlinear_optim_params = NonlinearOptimParams(
        optim_type=optim_type,
        backbone=backbone,
        momentum=momentum,
        weight_decay=weight_decay,
        lr=lr,
        lr_block=lr_block,
        lr_backbone=lr_net,
        dataset=dataset,
    )
    # PREPARE OPTIMIZER AND SCHEDULER
    optimizer, params_to_freeze, params_to_train = get_nonlinear_optimizer(model, nonlinear_optim_params)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=gamma
    )

    # TRAINING HELPERS
    def should_evaluate(candidate_epoch: int):
        if evaluate_each_epoch > 0 and candidate_epoch == 0:
            return False
        return candidate_epoch % evaluate_each_epoch == 0 or candidate_epoch == epochs

    params_frozen = False

    def freeze():
        nonlocal params_frozen
        for param in params_to_freeze:
            param.requires_grad = False
        params_frozen = True

    def unfreeze():
        nonlocal params_frozen
        for param in params_to_freeze:
            param.requires_grad = True
        params_frozen = False

    if freeze_epochs > 0:
        log.info(f"Freezing network for {freeze_epochs} epochs.")
        freeze()

    # TRAIN
    log.info("Starting training.")
    for epoch in range(1, epochs + 1):
        if params_frozen and epoch > freeze_epochs:
            log.info(f"\nUnfreezing network at {epoch=}.")
            unfreeze()

        train_epoch(
            model,
            train_loader,
            optimizer,
            progress_desc=f"Training epoch {epoch}/{epochs}",
        )
        scheduler.step()

        model.log_state()

        if should_evaluate(epoch):
            eval_model(model, test_loader, desc=f"Testing after epoch: {epoch}")
    log.info(f"Finished training.")

    # EVALUATE AND ANALYSE TRAINED TREE
    model = model.eval()

    model.log_state()

    _prune_tree(model.tree_root, leaf_pruning_threshold)

    model.log_state()
    pruned_acc = eval_model(model, test_loader, desc="Pruned only")
    log.info(f"\nTest acc. after pruning only: {pruned_acc:.3f}")

    # PROJECT
    log.info(
        "Projecting prototypes to nearest training patch (with class restrictions)."
    )
    node_to_patch_matches = node_patch_matches(model, project_loader)
    project_prototypes(model, node_to_patch_matches)  # TODO: Assess the impact of this.
    model.log_state()

    pruned_and_proj_acc = eval_model(model, test_loader)
    log.info(f"\nTest acc. after pruning and projection: {pruned_and_proj_acc:.3f}")
    single_leaf_eval(model, test_loader, "Pruned and projected")

    def explanations_provider():
        return data_explanations(
            model, test_loader, class_names
        )  # This is lazy to enable iterator reuse.

    model.tree_root.print_tree()

    # SAVE VISUALIZATIONS
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches"
    save_patch_visualizations(node_to_patch_matches, patches_dir)
    save_tree_visualization(model, patches_dir, vis_dir / "tree", class_names)
    save_multi_patch_visualizations(explanations_provider(), vis_dir / "explanations")
    save_decision_flow_visualizations(
        explanations_provider(), patches_dir, vis_dir / "explanations"
    )

    return model


def _prune_tree(root: InternalNode, leaf_pruning_threshold: float):
    log.info(
        f"Before pruning: {root.num_internal_nodes} internal_nodes and {root.num_leaves} leaves"
    )
    num_nodes_before = len(root.descendant_internal_nodes)

    # all work happens here, the rest is just logging
    prune_unconfident_leaves(root, leaf_pruning_threshold)

    frac_nodes_pruned = 1 - len(root.descendant_internal_nodes) / num_nodes_before
    log.info(
        f"After pruning: {root.num_internal_nodes} internal_nodes and {root.num_leaves} leaves"
    )
    log.info(f"Fraction of nodes pruned: {frac_nodes_pruned}")


def create_proto_tree(
        h_proto: int,
        w_proto: int,
        channels_proto: int,
        num_classes: int,
        depth: int,
        backbone_net="resnet50_inat",
        pretrained=True,
):
    return tree


def get_device(disable_cuda=False):
    if not disable_cuda and torch.cuda.is_available():
        num_cudas = torch.cuda.device_count()
        device_str = f"cuda:{randint(0, num_cudas - 1)}"  # TODO: Do this properly.
    else:
        device_str = "cpu"
    return torch.device(device_str)


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
