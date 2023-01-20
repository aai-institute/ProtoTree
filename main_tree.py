from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prototree.eval import eval_fidelity, eval_tree
from prototree.node import InternalNode
from prototree.project import replace_prototypes_by_projections
from prototree.prototree import ProtoTree
from prototree.prune import prune_unconfident_leaves
from prototree.train import train_epoch
from util.analyse import log_pruned_leaf_analysis
from util.args import get_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree_weights
from util.log import Log
from util.net import get_prototree_base_networks


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


def run_tree(args: Namespace, skip_visualization=True):
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
    epochs = 1
    evaluate_each_epoch = 20
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    freeze_epochs = 0

    # prototree specifics
    upsample_threshold = args.upsample_threshold
    # This option should always be true, at least for now
    pruning_threshold_percentage = 0.1
    pruning_threshold_leaves = 1 / 3 * (1 + pruning_threshold_percentage)

    # Net architecture args
    net = args.net
    pretrained = True
    h_prototype = 2
    w_prototype = 2
    depth = args.depth
    out_channels = args.num_features

    log = get_log(log_dir)

    train_loader, project_loader, test_loader = get_dataloaders(
        dataset=dataset, disable_cuda=disable_cuda, batch_size=batch_size
    )
    num_classes = len(test_loader.dataset.classes)
    log.log_message(f"Num classes (k): {num_classes}")
    tree = create_proto_tree(
        h_prototype, w_prototype, num_classes, depth, net, out_channels, pretrained
    )
    print(
        f"Max depth {depth}, so {tree.num_internal_nodes} internal nodes and {tree.num_leaves} leaves"
    )

    device = get_device(disable_cuda)
    print(f"Running on: {device}")
    tree = tree.to(device)

    optimizer, params_to_freeze, params_to_train = get_optimizer(
        tree,
        optim_type,
        net,
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

    # tree.save(f"{log.checkpoint_dir}/tree_init")

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

    print("Starting training")
    for epoch in range(epochs):
        if params_frozen and epoch > freeze_epochs:
            log.log_message(f"\nUnfreezing network at {epoch=}.")
            unfreeze()

        train_single_epoch(tree, epoch, log, optimizer, train_loader)
        scheduler.step()

        log_pruned_leaf_analysis(
            tree.leaves,
            pruning_threshold_leaves,
            log,
        )

        if should_evaluate(epoch):
            test_acc = eval_tree(
                tree, test_loader, eval_name=f"Testing after epoch: {epoch}"
            )
            # if test_acc > best_test_acc:
            #     best_test_acc = test_acc
            #     tree.save(f"{log.checkpoint_dir}/best_test_model")
    # only evaluate and for some reason also save, I disabled it for now
    # else:  # tree was loaded and not trained, so evaluate only
    #     raise NotImplementedError("This is not implemented yet")
    # eval_info = eval(tree, test_loader, epoch, device, log)
    # test_acc = eval_info["test_accuracy"]
    # save_tree(
    #     tree, optimizer, scheduler, log.checkpoint_dir, "best_test_model"
    # )

    # EVALUATE AND ANALYSE TRAINED TREE
    print(f"Training Finished.")

    log_pruned_leaf_analysis(tree.leaves, pruning_threshold_leaves, log)

    # TODO: see todo in the function, IMPORTANT
    _prune_tree(tree.tree_root, pruning_threshold_leaves, log)

    log_pruned_leaf_analysis(tree.leaves, pruning_threshold_leaves, log)
    pruned_acc = eval_tree(tree, test_loader, eval_name="pruned")
    log.log_message(f"\nAccuracy with distributed routing: {pruned_acc:.3f}")

    # PROJECT
    print("Projecting prototypes to nearest training patch (with class restrictions)")
    replace_prototypes_by_projections(tree, project_loader)
    log_pruned_leaf_analysis(tree.leaves, pruning_threshold_leaves, log)
    test_acc = eval_tree(tree, test_loader, eval_name="pruned_and_projected")
    log.log_message(f"Test after pruning and projection: {test_acc:.3f}")

    perform_final_evaluation(tree, test_loader, log, eval_name="pruned_and_projected")
    tree.tree_root.print_tree()


#     # Upsample prototype for visualization
#     upsample(
#         tree,
#         upsample_threshold,
#         project_info,
#         project_loader,
#         name,
#         log,
#         log_dir,
#         dir_for_saving_images,
#     )
#     generate_tree_visualization(
#         tree,
#         name,
#         tuple(range(num_classes)),
#         log_dir,
#         dir_for_saving_images,
#     )


def _prune_tree(root: InternalNode, pruning_threshold_leaves: float, log: Log):
    log.log_message("\nPruning...")
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


# TODO: this only logs stuff and doesn't return anything...
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
            eval_name=eval_name,
        )
    fidelities = eval_fidelity(projected_pruned_tree, test_loader)
    for strategy, fidelity in fidelities.items():
        log.log_message(f"Fidelity of {strategy} routing: {fidelity:.2f}")


def create_proto_tree(
    H1: int,
    W1: int,
    num_classes: int,
    depth: int,
    net: str,
    out_channels: int,
    pretrained=True,
):
    """

    :param H1: height of prototype
    :param W1: width of prototype
    :param num_classes:
    :param depth:
    :param net:
    :param out_channels: number of output channels of the net+add_on layers, prior to prototype layers.
        This coincides with the number of input channels for the prototypes
    :param pretrained:
    :return:
    """
    features_net, add_on_layers = get_prototree_base_networks(
        out_channels, net=net, pretrained=pretrained
    )
    tree = ProtoTree(
        num_classes=num_classes,
        prototype_channels=out_channels,
        depth=depth,
        feature_net=features_net,
        add_on_layers=add_on_layers,
        h_prototype=H1,
        w_prototype=W1,
    )
    init_tree_weights(tree)
    return tree


def train_single_epoch(
    tree: ProtoTree,
    epoch: int,
    log: Log,
    optimizer,
    trainloader: DataLoader,
):
    train_acc = train_epoch(
        tree,
        trainloader,
        optimizer,
        epoch,
        log,
        "log_train_epochs",
    )
    return train_acc


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
    run_tree(get_args(), skip_visualization=True)
