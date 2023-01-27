from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prototree.eval import eval_fidelity, eval_tree
from prototree.models import ProtoTree
from prototree.node import InternalNode, log_leaves_properties
from prototree.project import replace_prototypes_by_projections
from prototree.prune import prune_unconfident_leaves
from prototree.train import train_epoch
from prototree.visualization import save_prototype_visualizations
from util.args import get_args, get_optimizer
from util.data import get_dataloaders
from util.log import Log
from util.net import BASE_ARCHITECTURE_TO_FEATURES


@torch.no_grad()
def apply_xavier(tree: ProtoTree):
    def _xavier_on_conv(m):
        if type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_normal_(
                m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
            )

    tree.add_on.apply(_xavier_on_conv)


def save_tree(
    tree: ProtoTree,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    name="latest",
):
    basedir = Path(checkpoint_dir) / name
    tree.eval()
    tree.save(basedir)
    tree.save_state(basedir)
    torch.save(optimizer.state_dict(), basedir / "optimizer_state.pth")
    torch.save(scheduler.state_dict(), basedir / "scheduler_state.pth")


# TODO: use ptl, make this obsolete
def get_log(log_dir: str):
    log = Log(log_dir)
    print("Log dir: ", log_dir, flush=True)
    log.create_log(
        "log_epoch_overview",
        "epoch",
        "test_acc",
        "mean_train_acc",
        "mean_train_crossentropy_loss_during_epoch",
    )
    log.create_log(
        "log_train_epochs_losses",
        "epoch",
        "batch",
        "loss",
        "batch_train_acc",
    )
    return log


# TODO: remove dependency on args everywhere
def run_tree(args: Namespace):
    # data and paths
    dataset = args.dataset
    log_dir = args.log_dir

    # training hardware
    milestones = args.milestones
    gamma = args.gamma

    # Optimizer args
    optim_type = args.optimizer
    # batch_size = args.batch_size
    batch_size = 32
    lr = args.lr
    lr_block = args.lr_block
    lr_net = args.lr_net
    lr_pi = args.lr_pi
    momentum = args.momentum
    weight_decay = args.weight_decay

    # Training loop args
    disable_cuda = False
    epochs = 10
    evaluate_each_epoch = 5
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    freeze_epochs = 0

    # prototree specifics
    pruning_threshold_percentage = 0.1
    pruning_threshold_leaves = 1 / 3 * (1 + pruning_threshold_percentage)

    # Architecture args
    backbone = args.backbone
    pretrained = True
    h_proto = 1
    w_proto = 1
    channels_proto = args.num_features
    depth = 6

    log = get_log(log_dir)

    # PREPARE DATA
    device = get_device(disable_cuda)
    pin_memory = "cuda" in device.type
    train_loader, project_loader, test_loader = get_dataloaders(
        pin_memory=pin_memory,
        batch_size=batch_size,
    )
    num_classes = len(test_loader.dataset.classes)
    log.log_message(f"Num classes: {num_classes}")

    # PREPARE MODEL
    tree = create_proto_tree(
        h_proto=h_proto,
        w_proto=w_proto,
        channels_proto=channels_proto,
        num_classes=num_classes,
        depth=depth,
        backbone=backbone,
        pretrained=pretrained,
    )
    print(
        f"Max depth {depth}, so {tree.num_internal_nodes} internal nodes and {tree.num_leaves} leaves"
    )
    print(f"Running on: {device}")
    tree = tree.to(device)

    # PREPARE OPTIMIZER AND SCHEDULER
    optimizer, params_to_freeze, params_to_train = get_optimizer(
        tree,
        optim_type,
        backbone,
        dataset,
        momentum,
        weight_decay,
        lr,
        lr_block,
        lr_pi,
        lr_net,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=gamma
    )

    # TRAINING HELPERS
    def should_evaluate(epoch: int):
        if evaluate_each_epoch > 0 and epoch == 0:
            return False
        return epoch % evaluate_each_epoch == 0 or epoch == epochs

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
        log.log_message(f"\nFreezing network for {freeze_epochs} epochs.")
        freeze()

    # TRAIN
    print("Starting training")
    for epoch in range(epochs):
        if params_frozen and epoch > freeze_epochs:
            log.log_message(f"\nUnfreezing network at {epoch=}.")
            unfreeze()

        train_epoch(
            tree,
            train_loader,
            optimizer,
            progress_desc=f"Train Epoch {epoch + 1}/{epochs}:",
        )
        scheduler.step()

        log_leaves_properties(
            tree.leaves,
            pruning_threshold_leaves,
        )

        if should_evaluate(epoch):
            eval_tree(tree, test_loader, desc=f"Testing after epoch: {epoch}")

    # EVALUATE AND ANALYSE TRAINED TREE
    print(f"Training Finished.")
    tree = tree.eval()

    log_leaves_properties(tree.leaves, pruning_threshold_leaves)

    _prune_tree(tree.tree_root, pruning_threshold_leaves, log)

    log_leaves_properties(tree.leaves, pruning_threshold_leaves)
    pruned_acc = eval_tree(tree, test_loader, desc="pruned")
    log.log_message(f"\nAccuracy with distributed routing: {pruned_acc:.3f}")

    # PROJECT
    print("Projecting prototypes to nearest training patch (with class restrictions)")
    node_to_patch_info = replace_prototypes_by_projections(tree, project_loader)
    log_leaves_properties(tree.leaves, pruning_threshold_leaves)
    test_acc = eval_tree(tree, test_loader, desc="pruned_and_projected")
    log.log_message(f"Test after pruning and projection: {test_acc:.3f}")

    perform_final_evaluation(tree, test_loader, log, eval_name="pruned_and_projected")
    tree.tree_root.print_tree()

    # SAVE VISUALIZATIONS
    viz_path = Path("data") / "visualizations"
    viz_path.mkdir(exist_ok=True, parents=True)
    print(f"Saving prototype visualizations to {viz_path}")
    save_prototype_visualizations(
        node_to_patch_info,
        viz_path,
    )


def _prune_tree(root: InternalNode, pruning_threshold_leaves: float, log: Log):
    log.log_message(
        f"Before pruning: {root.num_internal_nodes} internal_nodes and {root.num_leaves} leaves"
    )
    num_nodes_before = len(root.descendant_internal_nodes)

    # all work happens here, the rest is just logging
    prune_unconfident_leaves(root, pruning_threshold_leaves)

    frac_nodes_pruned = 1 - len(root.descendant_internal_nodes) / num_nodes_before
    log.log_message(
        f"After pruning: {root.num_internal_nodes} internal_nodes and {root.num_leaves} leaves"
    )
    log.log_message(f"Fraction of nodes pruned: {frac_nodes_pruned}")


def perform_final_evaluation(
    projected_pruned_tree: ProtoTree,
    test_loader: DataLoader,
    log: Log,
    eval_name="Final evaluation",
):
    for sampling_strategy in ["sample_max", "greedy"]:
        eval_tree(
            projected_pruned_tree,
            test_loader,
            sampling_strategy=sampling_strategy,
            desc=eval_name,
        )
    fidelities = eval_fidelity(projected_pruned_tree, test_loader)
    for strategy, fidelity in fidelities.items():
        log.log_message(f"Fidelity of {strategy} routing: {fidelity:.2f}")


def create_proto_tree(
    h_proto: int,
    w_proto: int,
    channels_proto: int,
    num_classes: int,
    depth: int,
    backbone="resnet50_inat",
    pretrained=True,
):
    """

    :param h_proto: height of prototype
    :param w_proto: width of prototype
    :param channels_proto: number of input channels for the prototypes,
        coincides with the output channels of the net+add_on layers, prior to prototype layers.
    :param num_classes:
    :param depth: depth of tree, will result in 2^depth leaves and 2^depth-1 internal nodes
    :param backbone: name of backbone, e.g. resnet18
    :param pretrained:
    :return:
    """
    features_net = BASE_ARCHITECTURE_TO_FEATURES[backbone](pretrained=pretrained)
    tree = ProtoTree(
        num_classes=num_classes,
        depth=depth,
        channels_proto=channels_proto,
        h_proto=h_proto,
        w_proto=w_proto,
        feature_net=features_net,
    )
    apply_xavier(tree)
    return tree


def get_device(disable_cuda=False):
    if not disable_cuda and torch.cuda.is_available():
        device_str = f"cuda:{torch.cuda.current_device()}"
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
            print(
                "lovely_tensors not installed, not monkey patching. "
                "For more efficient debugging, we recommend installing it with `pip install lovely-tensors`."
            )
    run_tree(get_args())
