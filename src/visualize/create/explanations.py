import pydot
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree, LeafRationalization


@torch.no_grad()
def save_prediction_visualizations(
        tree: ProtoTree,
        loader: DataLoader,
        class_names: tuple
):
    tqdm_loader = tqdm(loader, desc="Data loader", ncols=0)
    prediction_counter = 0
    for batch_num, (x, y) in enumerate(tqdm_loader):
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_probs, predicting_leaves, leaf_rationalizations = tree.explain(x)
        for leaf_rationalization, true_label in zip(leaf_rationalizations, y):
            prediction_counter += 1
            pydot_graph = _gen_pydot_dag(leaf_rationalization, true_label, class_names, prediction_counter)


def _gen_pydot_dag(
        leaf_rationalization: LeafRationalization,
        true_label: int,
        class_names: tuple,
        prediction_counter: int
) -> pydot.Dot:
    pydot_dag = pydot.Dot(
        "Prediction graph",
        graph_type="digraph",
    )
