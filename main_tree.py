from argparse import Namespace
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Sequence

import lovely_tensors

from prototree.project import project_with_class_constraints
from prototree.prune import prune
from prototree.test import eval_fidelity, eval_tree
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.upsample import upsample
from util.analyse import *
from util.args import get_args, get_optimizer, save_args
from util.data import get_dataloaders
from util.init import init_tree_weights, load_things_into_tree
from util.net import freeze, get_prototree_base_networks
from util.visualize import gen_vis

lovely_tensors.monkey_patch()


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
    epochs = 10
    evaluate_each_epoch = 20
    # NOTE: after this, part of the net becomes unfrozen and loaded to GPU, which may cause memory errors
    # freeze_epochs = args.freeze_epochs
    freeze_epochs = 0

    # prototree specifics
    upsample_threshold = args.upsample_threshold
    kontschieder_train = args.kontschieder_train
    kontschieder_normalization = args.kontschieder_normalization
    disable_derivative_free_leaf_optim = args.disable_derivative_free_leaf_optim
    # TODO: this cannot come from args, needs to be adjusted to num of classes!!
    # pruning_threshold_leaves = args.pruning_threshold_leaves
    pruning_threshold_percentage = 0.1
    pruning_threshold_leaves = 1 / 3 * (1 + pruning_threshold_percentage)

    # Net architecture args
    net = args.net
    pretrained = True
    H1 = args.H1
    W1 = args.W1
    depth = args.depth
    out_channels = args.num_features

    log = get_log(log_dir)

    train_loader, project_loader, test_loader = get_dataloaders(
        dataset=dataset, disable_cuda=disable_cuda, batch_size=batch_size
    )
    num_classes = len(test_loader.dataset.classes)
    log.log_message(f"Num classes (k): {len(num_classes)}")
    tree = create_proto_tree(H1, W1, num_classes, depth, net, out_channels, pretrained)

    device = get_device(disable_cuda)
    print(f"Running on {device=}")
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
        disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=gamma
    )

    # tree.save(f"{log.checkpoint_dir}/tree_init")
    log.log_message(
        "Max depth %s, so %s internal nodes and %s leaves"
        % (depth, tree.num_internal_nodes, tree.num_leaves)
    )
    analyse_output_shape(tree, train_loader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.0
    best_test_acc = 0.0
    test_acc = 0.0
    # train and evaluate
    # if epoch < epochs + 1:
    def time_to_evaluate(epoch: int):
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
    for epoch in range(epochs + 1):
        if params_frozen and epoch > freeze_epochs:
            log.log_message(f"\nUnfreezing network at {epoch=}.")
            unfreeze()

        leaf_labels, train_info = train_single_epoch(
            num_classes,
            disable_derivative_free_leaf_optim,
            epoch,
            freeze_epochs,
            kontschieder_normalization,
            kontschieder_train,
            leaf_labels,
            log,
            net,
            optimizer,
            params_to_freeze,
            train_loader,
            tree,
            pruning_threshold_leaves,
        )
        scheduler.step()

        if time_to_evaluate(epoch):
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
    log.log_message(
        "Training Finished. Best training accuracy was %s, best test accuracy was %s\n"
        % (str(best_train_acc), str(best_test_acc))
    )
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_leaf_analysis(
        tree, epoch + 1, num_classes, leaf_labels, pruning_threshold_leaves, log
    )
    # TODO: this logs a long array, removing it for now
    # log_leaf_distributions_analysis(tree, log)

    # prune
    pruned_tree = deepcopy(tree)
    prune(pruned_tree, pruning_threshold_leaves, log)
    # todo: pass actual class labels?
    leaf_labels, pruned_test_acc = analyse_tree(
        tree,
        range(num_classes),
        epoch,
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
    add_epoch_statistic_to_leaf_labels_dict_and_log_leaf_analysis(
        projected_pruned_tree,
        epoch + 3,
        num_classes,
        leaf_labels,
        pruning_threshold_leaves,
        log,
    )
    log_leaf_distributions_analysis(projected_pruned_tree, log)
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
        # visualize tree
        gen_vis(projected_pruned_tree, name, classes, log_dir, dir_for_saving_images)


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
        get_avg_path_length(projected_pruned_tree, eval_info, log)
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
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_prototree_base_networks(
        out_channels, net=net, pretrained=pretrained
    )
    tree = ProtoTree(
        num_classes=num_classes,
        out_channels=out_channels,
        depth=depth,
        feature_net=features_net,
        add_on_layers=add_on_layers,
        H1=H1,
        W1=W1,
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
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_leaf_analysis(
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
    log_optimization_info(
        epoch,
        log,
        net,
        optimizer,
        disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
    )
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
    leaf_labels = add_epoch_statistic_to_leaf_labels_dict_and_log_leaf_analysis(
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


def log_optimization_info(
    epoch, log, net, optimizer, disable_derivative_free_leaf_optim=False
):
    log.log_message("\nEpoch %s" % str(epoch))
    log_learning_rates(
        optimizer,
        net,
        log,
        disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
    )


def get_device(disable_cuda):
    if not disable_cuda and torch.cuda.is_available():
        device_str = f"cuda:{torch.cuda.current_device()}"
    else:
        device_str = "cpu"
    return torch.device(device_str)


if __name__ == "__main__":
    run_tree(get_args(), skip_visualization=True)
