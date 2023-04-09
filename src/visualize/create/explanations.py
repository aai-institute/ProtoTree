import os

import pydot
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree, LeafRationalization


@torch.no_grad()
def save_prediction_visualizations(
        tree: ProtoTree,
        loader: DataLoader,
        class_names: tuple,
        save_dir: os.PathLike,
        img_size=(224, 224)
):
    tqdm_loader = tqdm(loader, desc="Data loader", ncols=0)
    prediction_counter = 0
    for batch_num, (x, y) in enumerate(tqdm_loader):
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_probs, predicting_leaves, leaf_rationalizations = tree.explain(x)
        for leaf_rationalization, true_label in zip(leaf_rationalizations, y):
            prediction_counter += 1
            pydot_graph = _decision_flow_pydot(leaf_rationalization, true_label, class_names, prediction_counter)


def _decision_flow_pydot(
        leaf_rationalization: LeafRationalization,
        true_label: int,
        class_names: tuple,
        prediction_counter: int
) -> pydot.Dot:
    flow_dag = pydot.Dot(
        "Decision flow for prediction of 1 image.",
        graph_type="digraph",
    )

    for ancestor_similarity in leaf_rationalization.ancestor_similarities:
        ancestor_similarity.transformed_image
