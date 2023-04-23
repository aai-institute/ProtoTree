import logging

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree

log = logging.getLogger(__name__)


def train_epoch(
    model: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    progress_desc: str = "Train Epoch",
) -> dict:
    n_batches = len(train_loader)

    tqdm_loader = tqdm(
        train_loader,
        desc=progress_desc,
    )
    tqdm_loader.update()  # Stops earlier logging appearing after tqdm starts showing progress.
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    for batch_num, (x, y) in enumerate(tqdm_loader):
        model.train()
        optimizer.zero_grad()
        x, y = x.to(model.device), y.to(model.device)
        logits, node_to_prob, predicting_leaves = model.forward(x)
        loss = F.nll_loss(logits, y)
        loss.backward()
        optimizer.step()

        smoothing_factor = 1 - 1 / n_batches
        model.tree_section.update_leaf_distributions(
            y, logits.detach(), node_to_prob, smoothing_factor
        )

        model.eval()
        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / len(x)
        tqdm_loader.set_postfix_str(f"batch: loss={loss.item():.5f}, {acc=:.5f}")
        total_loss += loss.item()
        total_acc += acc

        if (
            batch_num == n_batches - 1
        ):  # TODO: Hack due to https://github.com/tqdm/tqdm/issues/1369
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            tqdm_loader.set_postfix_str(
                f"average: loss={avg_loss:.5f}, acc={avg_acc:.5f}"
            )

    return {
        "loss": avg_loss,
        "train_accuracy": avg_acc,
    }
