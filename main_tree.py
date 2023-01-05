from argparse import Namespace
from copy import deepcopy
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from prototree.project import project_with_class_constraints
from prototree.prototree import ProtoTree
from prototree.prune import prune_unconfident_leaves
from prototree.test import eval_fidelity, eval_tree
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.upsample import upsample
from util.analyse import (
    add_epoch_statistic_to_leaf_labels_dict_and_log_pruned_leaf_analysis,
    average_distance_nearest_image,
    log_avg_path_length,
)
from util.args import get_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree_weights
from util.log import Log
from util.net import get_prototree_base_networks
from util.visualize import generate_tree_visualization


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
    # Create a logger
    log = Log(log_dir)
    print("Log dir: ", log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log(
        "log_epoch_overview",
        "epoch",
        "test_acc",
        "mean_train_acc",
        "mean_train_crossentropy_loss_during_epoch",
    )

    log_loss = "log_train_epochs_losses"
    log.create_log(log_loss, "epoch", "batch", "loss", "batch_train_acc")
    return log


def run_tree(args: Namespace, skip_visualization=True):
    # data and paths
    dataset = args.dataset
    log_dir = args.log_dir
    dir_for_saving_images = args.dir_for_saving_images

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
    # epochs = args.epochs
    disable_cuda = args.disable_cuda
    epochs = 2
    evaluate_each_epoch = 20
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU,
    # which may cause surprising memory errors after the training was already running for a while
    # freeze_epochs = args.freeze_epochs
    freeze_epochs = 0

    # prototree specifics
    upsample_threshold = args.upsample_threshold
    kontschieder_train = args.kontschieder_train
    kontschieder_normalization = args.kontschieder_normalization
    # This option should always be true, at least for now
    # disable_derivative_free_leaf_optim = args.disable_derivative_free_leaf_optim
    # TODO: this cannot come from args, needs to be adjusted to num of classes!!
    # pruning_threshold_leaves = args.pruning_threshold_leaves
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
    print(f"Running on: {device}, moving tree to device")
    tree = tree.to(device)
    print("Creating optimizer")

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
    print("Creating scheduler")
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
    leaf_labels = dict()
    test_acc = 0.0
    print("Starting training")
    for epoch in range(epochs):
        if params_frozen and epoch > freeze_epochs:
            log.log_message(f"\nUnfreezing network at {epoch=}.")
            unfreeze()

        leaf_labels, train_info = train_single_epoch(
            num_classes,
            epoch,
            kontschieder_normalization,
            kontschieder_train,
            leaf_labels,
            log,
            net,
            optimizer,
            train_loader,
            tree,
            pruning_threshold_leaves,
        )
        scheduler.step()

        if should_evaluate(epoch):
            eval_info = eval_tree(
                tree, test_loader, log, eval_name=f"Testing after epoch: {epoch}"
            )
            test_acc = eval_info["test_accuracy"]
            # if test_acc > best_test_acc:
            #     best_test_acc = test_acc
            #     tree.save(f"{log.checkpoint_dir}/best_test_model")
        log.log_values(
            "log_epoch_overview",
            epoch,
            test_acc,
            train_info["train_accuracy"],
            train_info["loss"],
        )
    # only evaluate and for some reason also save, I disabled it for now
    # else:  # tree was loaded and not trained, so evaluate only
    #     raise NotImplementedError("This is not implemented yet")
    # eval_info = eval(tree, test_loader, epoch, device, log)
    # test_acc = eval_info["test_accuracy"]
    # save_tree(
    #     tree, optimizer, scheduler, log.checkpoint_dir, "best_test_model"
    # )
    # log.log_values(
    #     "log_epoch_overview", epoch, eval_info["test_accuracy"], "n.a.", "n.a."
    # )

    # EVALUATE AND ANALYSE TRAINED TREE
    print(f"Training Finished.")
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_pruned_leaf_analysis(
        tree, epochs - 1, num_classes, leaf_labels, pruning_threshold_leaves, log
    )
    # TODO: this logs a long array, removing it for now
    # log_leaf_distributions_analysis(tree, log)

    # TODO: see todo in the function, IMPORTANT
    pruned_tree = _get_pruned_tree(tree, pruning_threshold_leaves, log)

    # todo: pass actual class labels?
    leaf_labels, pruned_test_acc = analyse_tree(
        tree,
        range(num_classes),
        epochs - 1,
        leaf_labels,
        log,
        pruning_threshold_leaves,
        test_loader,
        eval_name="pruned",
    )
    # save_tree(pruned_tree, optimizer, scheduler, log.checkpoint_dir, name="pruned")
    #
    # # find "real image" prototypes through projection
    projected_pruned_tree, project_info = project_with_class_constraints(
        deepcopy(pruned_tree), project_loader, log
    )
    name = "pruned_and_projected"
    # save_tree(projected_pruned_tree, optimizer, scheduler, log.checkpoint_dir, name=name)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, projected_pruned_tree, log)
    add_epoch_statistic_to_leaf_labels_dict_and_log_pruned_leaf_analysis(
        projected_pruned_tree,
        # TODO: wassup with +3?
        epoch + 3,
        num_classes,
        leaf_labels,
        pruning_threshold_leaves,
        log,
    )
    eval_info = eval_tree(projected_pruned_tree, test_loader, log, eval_name=name)
    log.log_message(f"Test after pruning and projection: {eval_info['test_accuracy']}")

    perform_final_evaluation(projected_pruned_tree, test_loader, log, eval_name=name)

    if not skip_visualization:
        # Upsample prototype for visualization
        upsample(
            projected_pruned_tree,
            upsample_threshold,
            project_info,
            project_loader,
            name,
            log,
            log_dir,
            dir_for_saving_images,
        )
        generate_tree_visualization(
            projected_pruned_tree,
            name,
            tuple(range(num_classes)),
            log_dir,
            dir_for_saving_images,
        )


def _get_pruned_tree(tree: ProtoTree, pruning_threshold_leaves: float, log: Log):
    # TODO: this doesn't actually prune anything, it just modifies the parents relations which
    #  in no way affects inference and forward calls. What is done later on is that the pruning threshold
    #  is passed again to an analysis function which drops a part of the predicted label distributions...
    log.log_message("\nPruning...")
    log.log_message(
        f"Before pruning: {tree.num_internal_nodes} internal_nodes and {tree.num_leaves} leaves"
    )
    num_prototypes_before = tree.num_internal_nodes

    # all work happens here, the rest is just logging
    pruned_tree = deepcopy(tree)
    prune_unconfident_leaves(pruned_tree.tree_root, pruning_threshold_leaves)

    frac_nodes_pruned = 1 - pruned_tree.num_internal_nodes / num_prototypes_before
    log.log_message(
        f"After pruning: {tree.num_internal_nodes} internal_nodes and {tree.num_leaves} leaves"
    )
    log.log_message(f"Fraction of prototypes pruned: {frac_nodes_pruned}")
    return pruned_tree


# TODO: this mainly logs stuff and doesn't return anything...
def perform_final_evaluation(
    projected_pruned_tree: ProtoTree,
    test_loader: DataLoader,
    log: Log,
    eval_name="Final evaluation",
):
    for sampling_strategy in ["sample_max", "greedy"]:
        eval_info = eval_tree(
            projected_pruned_tree,
            test_loader,
            log,
            sampling_strategy=sampling_strategy,
            eval_name=eval_name,
        )
        log_avg_path_length(projected_pruned_tree, eval_info, log)
    eval_fidelity(projected_pruned_tree, test_loader, log)


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


# TODO: rename, split up
def analyse_tree(
    tree,
    classes,
    epoch,
    leaf_labels,
    log,
    pruning_threshold_leaves,
    testloader,
    eval_name="Eval",
):
    """
    Does a whole bunch of logging to some secret log file

    :param tree:
    :param classes:
    :param epoch:
    :param leaf_labels:
    :param log:
    :param pruning_threshold_leaves:
    :param testloader:
    :param eval_name:
    :return:
    """
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_pruned_leaf_analysis(
        tree, epoch + 2, len(classes), leaf_labels, pruning_threshold_leaves, log
    )
    # TODO: this just logs a long array, really not necessary
    # log_leaf_distributions_analysis(tree, log)
    eval_info = eval_tree(tree, testloader, log, eval_name=eval_name)
    pruned_test_acc = eval_info["test_accuracy"]
    return leaf_labels, pruned_test_acc


# TODO: just kill me now...
def train_single_epoch(
    num_classes,
    epoch,
    kontschieder_normalization,
    kontschieder_train,
    leaf_labels,
    log,
    net,
    optimizer,
    trainloader,
    tree,
    pruning_threshold_leaves,
    disable_derivative_free_leaf_optim=False,
):
    epoch_trainer = get_single_epoch_trainer(
        kontschieder_normalization, kontschieder_train
    )

    train_info = epoch_trainer(
        tree,
        trainloader,
        optimizer,
        epoch,
        disable_derivative_free_leaf_optim,
        log,
        "log_train_epochs",
    )

    # TODO: does this have to happen before scheduler step?
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_pruned_leaf_analysis(
        tree,
        epoch,
        num_classes,
        leaf_labels,
        pruning_threshold_leaves,
        log,
    )
    return leaf_labels, train_info


def get_single_epoch_trainer(
    kontschieder_normalization: bool, kontschieder_train: bool
):
    if kontschieder_train:
        return partial(
            train_epoch_kontschieder,
            kontschieder_normalization=kontschieder_normalization,
        )
    return train_epoch


def get_device(disable_cuda):
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
