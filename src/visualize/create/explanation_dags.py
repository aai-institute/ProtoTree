import os
from collections.abc import Callable
from typing import Iterator, Union

import numpy as np
import pydot
import torch
from PIL.Image import Image
from tqdm import tqdm

from prototree.models import LeafRationalization
from prototree.node import InternalNode, Leaf, Node
from util.data import save_img
from util.image import get_inverse_base_transform, get_latent_to_pixel
from visualize.create.patches import closest_patch_imgs


@torch.no_grad()
def save_explanation_visualizations(
    explanations: Iterator[tuple[LeafRationalization, int, tuple]],
    patches_dir: os.PathLike,
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    inverse_transform = get_inverse_base_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    tqdm_explanations = tqdm(explanations, desc="Visualizing explanations", ncols=0)
    for explanation_counter, (leaf_explanation, true_label, class_names) in enumerate(
        tqdm_explanations
    ):
        explanation_dir = explanations_dir / f"img_{explanation_counter}"
        _save_explanation(
            leaf_explanation,
            true_label,
            class_names,
            inverse_transform,
            latent_to_pixel,
            patches_dir,
            explanation_dir,
        )


def _save_explanation(
    leaf_rationalization: LeafRationalization,
    true_label: int,
    class_names: tuple,
    inverse_transform: Callable[[torch.Tensor], Image],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    patches_dir: os.PathLike,
    explanation_dir: os.PathLike,
):
    dag = pydot.Dot(
        "Decision flow for explanation of 1 image.",
        graph_type="digraph",
    )
    proto_pydot_nodes, bbox_pydot_nodes = [], []
    decision_pydot_edges, bbox_pydot_edges = [], []

    for ancestor_similarity, went_right in zip(
        leaf_rationalization.ancestor_similarities,
        leaf_rationalization.ancestors_went_right,
    ):
        (
            im_closest_patch,
            im_original,
            im_with_bbox,
            im_with_heatmap,
        ) = closest_patch_imgs(ancestor_similarity, inverse_transform, latent_to_pixel)

        proto_node = ancestor_similarity.internal_node
        proto_file = patches_dir / f"{proto_node.index}_closest_patch.png"
        proto_pydot_node = pydot.Node(
            _proto_node_name(proto_node),
            image=f'"{proto_file}"',
            label="",
            shape="plaintext",
        )
        proto_pydot_nodes.append(proto_pydot_node)

        if went_right:
            bbox_file = explanation_dir / f"level_{proto_node.depth}_bounding_box.png"
            save_img(im_with_bbox, bbox_file)
            bbox_pydot_node = pydot.Node(
                _bbox_node_name(proto_node),
                image=f'"{bbox_file}"',
                label="",
                shape="plaintext",
            )
            bbox_pydot_nodes.append(bbox_pydot_node)

            decision_edge = pydot.Edge(
                _proto_node_name(proto_node),
                _proto_node_name(proto_node.right),
                label="Present",
            )

            bbox_pydot_edge = pydot.Edge(
                _proto_node_name(proto_node),
                _bbox_node_name(proto_node),
            )
            bbox_pydot_edges.append(bbox_pydot_edge)
        else:
            decision_edge = pydot.Edge(
                _proto_node_name(proto_node),
                _proto_node_name(proto_node.left),
                label="Absent",
            )
        decision_pydot_edges.append(decision_edge)

    pydot_nodes = proto_pydot_nodes + bbox_pydot_nodes
    pydot_edges = decision_pydot_edges + bbox_pydot_edges
    for pydot_node in pydot_nodes:
        dag.add_node(pydot_node)
    for pydot_edge in pydot_edges:
        dag.add_edge(pydot_edge)

    dot_file = explanation_dir / "explanation.dot"
    dag.write_dot(dot_file)

    png_file = explanation_dir / "explanation.png"
    dag.write_png(png_file)


def _proto_node_name(node: Union[InternalNode, Leaf]) -> str:
    match node:
        case InternalNode():
            return f"proto_{node.index}"
        case Leaf():
            return f"leaf_{node.index}"
        case other:
            raise ValueError(f"Unknown node {other}.")


def _bbox_node_name(node: Node) -> str:
    return f"bbox_{node.index}"
