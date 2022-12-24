from argparse import Namespace
from copy import deepcopy
from functools import partial
from pathlib import Path

from prototree.project import project_with_class_constraints
from prototree.prune import prune
from prototree.test import eval_fidelity, eval_tree
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.upsample import upsample
from util.analyse import *
from util.args import get_args, get_optimizer, save_args
from util.data import get_dataloaders
from util.init import init_tree
from util.net import freeze, get_network
from util.visualize import gen_vis


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


def run_tree(args: Namespace):
    dataset = args.dataset
    disable_cuda = args.disable_cuda
    batch_size = args.batch_size
    log_dir = args.log_dir
    milestones = args.milestones
    gamma = args.gamma
    pruning_threshold_leaves = args.pruning_threshold_leaves
    freeze_epochs = args.freeze_epochs
    net = args.net

    # Optimizer args
    lr = args.lr
    lr_block = args.lr_block
    lr_net = args.lr_net
    lr_pi = args.lr_pi
    momentum = args.momentum
    weight_decay = args.weight_decay
    disable_derivative_free_leaf_optim = args.disable_derivative_free_leaf_optim

    # Training args
    epochs = args.epochs
    kontschieder_train = args.kontschieder_train
    kontschieder_normalization = args.kontschieder_normalization

    log = get_log(log_dir)
    save_args(args, log.metadata_dir)

    device = get_device(disable_cuda, log)

    trainloader, projectloader, testloader, classes, num_channels = get_dataloaders(
        dataset=dataset, disable_cuda=disable_cuda, batch_size=batch_size
    )

    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    tree = ProtoTree(
        num_classes=len(classes),
        feature_net=features_net,
        add_on_layers=add_on_layers,
    )
    tree = tree.to(device=device)

    optimizer, params_to_freeze, params_to_train = get_optimizer(
        tree,
        net,
        dataset,
        momentum,
        weight_decay,
        lr_block,
        lr,
        lr_block,
        lr_pi,
        lr_net,
        disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=milestones, gamma=gamma
    )
    # TODO: this may also load the tree from a checkpoint, hence the epoch. Split this up.
    #  It currently mutates the scheduler!!
    # TODO: I removed the loading part, the paths to load from would previously be read off the args
    tree, epoch = init_tree(tree, optimizer, scheduler=scheduler)

    tree.save(f"{log.checkpoint_dir}/tree_init")
    log.log_message(
        "Max depth %s, so %s internal nodes and %s leaves"
        % (args.depth, tree.num_descendants, tree.num_leaves)
    )
    analyse_output_shape(tree, trainloader, log, device)

    leaf_labels = dict()
    best_train_acc = 0.0
    best_test_acc = 0.0
    test_acc = 0.0
    # train and evaluate
    if epoch < epochs + 1:
        for epoch in range(epoch, args.epochs + 1):
            best_train_acc, leaf_labels, train_info = train_single_epoch(
                best_train_acc,
                classes,
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
                scheduler,
                trainloader,
                tree,
                pruning_threshold_leaves,
            )

            # Evaluate tree
            if epoch % 10 == 0 or epoch == epochs:
                eval_info = eval_tree(tree, testloader, epoch, device, log)
                test_acc = eval_info["test_accuracy"]
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    tree.save(f"{log.checkpoint_dir}/best_test_model")
            log.log_values(
                "log_epoch_overview",
                epoch,
                test_acc,
                train_info["train_accuracy"],
                train_info["loss"],
            )
            scheduler.step()
    # only evaluate and for some reason also save, I disabled it for now
    else:  # tree was loaded and not trained, so evaluate only
        raise NotImplementedError("This is not implemented yet")
        # eval_info = eval(tree, testloader, epoch, device, log)
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
    leaf_labels = analyse_leaves(
        tree, epoch + 1, len(classes), leaf_labels, args.pruning_threshold_leaves, log
    )
    log_leaf_distributions_analysis(tree, log)

    # prune
    pruned_tree = deepcopy(tree)
    prune(pruned_tree, pruning_threshold_leaves, log)
    leaf_labels, pruned_test_acc = analyse_tree(
        tree,
        classes,
        epoch,
        leaf_labels,
        log,
        pruning_threshold_leaves,
        testloader,
        eval_name="pruned",
    )
    save_tree(pruned_tree, optimizer, scheduler, log.checkpoint_dir, name="pruned")

    # find "real image" prototypes through projection
    # TODO: don't overwrite tree...
    project_info, tree = project_with_class_constraints(
        deepcopy(pruned_tree), projectloader, device, args, log
    )
    name = "pruned_and_projected"
    save_tree(tree, optimizer, scheduler, log.checkpoint_dir, name=name)
    pruned_projected_tree = deepcopy(tree)
    # Analyse and evaluate pruned tree with projected prototypes
    average_distance_nearest_image(project_info, tree, log)
    analyse_leaves(
        tree, epoch + 3, len(classes), leaf_labels, pruning_threshold_leaves, log
    )
    log_leaf_distributions_analysis(tree, log)
    eval_info = eval_tree(tree, testloader, name, device, log)
    pruned_projected_test_acc = eval_info["test_accuracy"]
    eval_info_samplemax = eval_tree(tree, testloader, name, device, log, "sample_max")
    get_avg_path_length(tree, eval_info_samplemax, log)
    eval_info_greedy = eval_tree(tree, testloader, name, device, log, "greedy")
    get_avg_path_length(tree, eval_info_greedy, log)
    fidelity_info = eval_fidelity(tree, testloader, device, log)

    # Upsample prototype for visualization
    project_info = upsample(tree, project_info, projectloader, name, args, log)
    # visualize tree
    gen_vis(tree, name, args, classes)

    # TODO: simplify this, it is actually never used...
    return (
        tree.to("cpu"),
        pruned_tree.to("cpu"),
        pruned_projected_tree.to("cpu"),
        test_acc,
        pruned_test_acc,
        pruned_projected_test_acc,
        project_info,
        eval_info_samplemax,
        eval_info_greedy,
        fidelity_info,
    )


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
    leaf_labels = analyse_leaves(
        tree, epoch + 2, len(classes), leaf_labels, pruning_threshold_leaves, log
    )
    log_leaf_distributions_analysis(tree, log)
    eval_info = eval_tree(tree, testloader, log, eval_name=eval_name)
    pruned_test_acc = eval_info["test_accuracy"]
    return leaf_labels, pruned_test_acc


# TODO: just kill me now...
def train_single_epoch(
    best_train_acc,
    classes,
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
    scheduler,
    trainloader,
    tree,
    pruning_threshold_leaves,
):
    log.log_message("\nEpoch %s" % str(epoch))
    log_learning_rates(
        optimizer,
        net,
        log,
        disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
    )
    # Train tree
    epoch_train_method = (
        partial(
            train_epoch_kontschieder,
            kontschieder_normalization=kontschieder_normalization,
        )
        if kontschieder_train
        else train_epoch
    )
    # Freeze (part of) network for some epochs if indicated
    freeze(epoch, params_to_freeze, log, freeze_epochs)
    train_info = epoch_train_method(
        tree,
        trainloader,
        optimizer,
        epoch,
        disable_derivative_free_leaf_optim,
        log,
        "log_train_epochs",
    )
    # TODO: do we need that much saving?
    save_tree(
        tree, optimizer, scheduler, checkpoint_dir=log.checkpoint_dir, name="latest"
    )
    if epoch % 10 == 0:
        save_tree(
            tree,
            optimizer,
            scheduler,
            checkpoint_dir=log.checkpoint_dir,
            name=f"epoch_{epoch}",
        )
    train_accuracy = train_info["train_accuracy"]
    if train_accuracy > best_train_acc:
        save_tree(
            tree,
            optimizer,
            scheduler,
            checkpoint_dir=log.checkpoint_dir,
            name="best_train",
        )
    leaf_labels = analyse_leaves(
        tree,
        epoch,
        len(classes),
        leaf_labels,
        pruning_threshold_leaves,
        log,
    )
    return train_accuracy, leaf_labels, train_info


def get_device(disable_cuda, log):
    if not disable_cuda and torch.cuda.is_available():
        device_str = f"cuda:{torch.cuda.current_device()}"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    log.log_message("Device used: " + str(device))
    return device


if __name__ == "__main__":
    run_tree(get_args())
