import logging

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.prototree import ProtoTree
from util.log import Log

logger = logging.getLogger(__name__)


def train_epoch(
    tree: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    disable_derivative_free_leaf_optim: bool = False,
    log: Log = None,
    log_prefix: str = "log_train_epochs",
    progress_prefix: str = "Train Epoch",
) -> dict:
    # TODO: combine this block with the leaf distribution optimization below into a single, separate function
    tree.eval()
    nr_batches = float(len(train_loader))
    # with torch.no_grad():
    #     _old_dist_params = {
    #         leaf: leaf.dist_params.detach().clone() for leaf in tree.leaves
    #     }

    # Show progress on progress bar
    train_iter = tqdm(
        train_loader,
        desc=f"{progress_prefix} {epoch}",
    )
    tree.train()

    # Iterate through the data set to update leaves, prototypes and network
    eye = torch.eye(tree.num_classes, device=tree.device)
    total_loss = 0.0
    total_acc = 0.0
    for i, (xs, ys) in enumerate(train_iter):
        optimizer.zero_grad()
        # TODO: to we need to send them to the device? pin_memory=True should be enough
        xs, ys = xs.to(tree.device), ys.to(tree.device)

        y_pred_proba, info = tree.forward(xs)
        logits = y_pred_proba if tree.log_probabilities else torch.log(y_pred_proba)

        loss = F.nll_loss(logits, ys)
        loss.backward()
        optimizer.step()

        # TODO: remove option of this being false. Also, remove option of configuring log_probabilities
        if not disable_derivative_free_leaf_optim:
            # Update leaves with derivative-free algorithm
            # Make sure the tree is in eval mode
            target = eye[ys]  # shape (batchsize, num_classes), one-hot encoding of ys
            tree.eval()  # TODO: is this necessary?
            with torch.no_grad():
                for leaf in tree.leaves:
                    # shape (batchsize, 1)
                    p_arrival = info["p_arrival"][leaf.index].unsqueeze(1)
                    # shape (num_classes)
                    dist = leaf.distribution()

                    if tree.log_probabilities:
                        log_update = torch.logsumexp(
                            p_arrival + dist + torch.log(target) - y_pred_proba,
                            dim=0,
                        )
                        update = torch.exp(log_update)
                    else:
                        update = torch.sum(
                            p_arrival * dist * target / y_pred_proba, dim=0
                        )

                    # TODO: is this still normalized?
                    leaf.dist_params *= 1 - 1 / nr_batches
                    leaf.dist_params += update
                    # dist_params values can get slightly negative because of floating point issues, so set to zero.
                    # TODO: previously this line had no effect, I reinstated it. I hope it doesn't break stuff...
                    F.relu(leaf.dist_params, inplace=True)

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(y_pred_proba, dim=1)

        # TODO: use some accuracy evaluation and logging from a library (talking about reinventing the wheel...)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        # TODO: is this needed?
        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}"
        )
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

        # TODO: move this outside, remove dependency on epoch and probably also all log stuff
        if log is not None:
            log.log_values(f"{log_prefix}_losses", epoch, i + 1, loss.item(), acc)

    return {
        "loss": total_loss / (nr_batches + 1),
        "train_accuracy": total_acc / (nr_batches + 1),
    }


# TODO: remove the massive duplication
def train_epoch_kontschieder(
    tree: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device,
    log: Log = None,
    log_prefix: str = "log_train_epochs",
    progress_prefix: str = "Train Epoch",
    disable_derivative_free_leaf_optim: bool = False,
    kontschieder_normalization=False,
) -> dict:
    """

    :param tree:
    :param train_loader:
    :param optimizer:
    :param epoch:
    :param disable_derivative_free_leaf_optim:
    :param device:
    :param log:
    :param log_prefix:
    :param progress_prefix:
    :param kontschieder_normalization:
    :return: dict with two entries - loss and train_accuracy
    """

    tree = tree.to(device)

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

    # Create a log if required
    log_loss = f"{log_prefix}_losses"
    if log is not None and epoch == 1:
        log.create_log(log_loss, "epoch", "batch", "loss", "batch_train_acc")

    # Reset the gradients
    optimizer.zero_grad()

    if disable_derivative_free_leaf_optim:
        print(
            "WARNING: kontschieder arguments will be ignored when training leaves with gradient descent"
        )
    else:
        if kontschieder_normalization:
            # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
            for _ in range(10):
                # Train leaves with derivative-free algorithm using normalization factor
                train_leaves_epoch(tree, train_loader, epoch, device)
        else:
            # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
            train_leaves_epoch(tree, train_loader, epoch, device)
    # Train prototypes and network.
    # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
    # Show progress on progress bar
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + " %s" % epoch,
        ncols=0,
    )
    # Make sure the model is in train mode
    tree.train()
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Reset the gradients
        optimizer.zero_grad()
        # Perform a forward pass through the network
        ys_pred, _ = tree.forward(xs)
        # Compute the loss
        if tree.log_probabilities:
            loss = F.nll_loss(ys_pred, ys)
        else:
            loss = F.nll_loss(torch.log(ys_pred), ys)
        # Compute the gradient
        loss.backward()
        # Update model parameters
        optimizer.step()

        # Count the number of correct classifications
        ys_pred = torch.argmax(ys_pred, dim=1)

        correct = torch.sum(torch.eq(ys_pred, ys))
        acc = correct.item() / float(len(xs))

        train_iter.set_postfix_str(
            f"Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}"
        )
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info["loss"] = total_loss / float(i + 1)
    train_info["train_accuracy"] = total_acc / float(i + 1)
    return train_info


# Updates leaves with derivative-free algorithm
def train_leaves_epoch(
    tree: ProtoTree,
    train_loader: DataLoader,
    epoch: int,
    device,
    progress_prefix: str = "Train Leafs Epoch",
) -> dict:

    # Make sure the tree is in eval mode for updating leafs
    tree.eval()

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in tree.leaves:
            _old_dist_params[leaf] = leaf.dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(tree.num_classes).to(device)

        # Show progress on progress bar
        train_iter = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=progress_prefix + " %s" % epoch,
            ncols=0,
        )

        # Iterate through the data set
        update_sum = dict()

        # Create empty tensor for each leaf that will be filled with new values
        for leaf in tree.leaves:
            update_sum[leaf] = torch.zeros_like(leaf.dist_params)

        for i, (xs, ys) in train_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Train leafs without gradient descent
            out, info = tree.forward(xs)
            target = eye[ys]  # shape (batchsize, num_classes)
            for leaf in tree.leaves:
                if tree.log_probabilities:
                    # log version
                    update = torch.exp(
                        torch.logsumexp(
                            info["p_arrival"][leaf.index]
                            + leaf.distribution()
                            + torch.log(target)
                            - out,
                            dim=0,
                        )
                    )
                else:
                    update = torch.sum(
                        (info["p_arrival"][leaf.index] * leaf.distribution() * target)
                        / out,
                        dim=0,
                    )
                update_sum[leaf] += update

        for leaf in tree.leaves:
            leaf.dist_params -= leaf.dist_params  # set current dist params to zero
            leaf.dist_params += update_sum[leaf]  # give dist params new value
