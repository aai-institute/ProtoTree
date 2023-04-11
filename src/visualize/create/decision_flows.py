import os
from collections.abc import Callable
from typing import Iterator

import numpy as np
import pydot
import torch
from tqdm import tqdm

from prototree.models import LeafRationalization
from prototree.node import Node
from util.data import save_img
from util.image import get_latent_to_pixel, get_inverse_arr_transform
from visualize.create.dot import _node_name, gen_leaf, FONT
from visualize.create.patches import closest_patch_imgs


@torch.no_grad()
def save_decision_flow_visualizations(
    explanations: Iterator[tuple[LeafRationalization, int, tuple]],
    patches_dir: os.PathLike,
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    decision_flows_dir = explanations_dir / "decision_flows"
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    tqdm_explanations = tqdm(explanations, desc="Visualizing decision flows", ncols=0)
    for explanation_counter, (leaf_explanation, true_label, class_names) in enumerate(
        tqdm_explanations
    ):
        decision_flow_dir = decision_flows_dir / f"img_{explanation_counter}"
        _save_decision_flows(
            leaf_explanation,
            true_label,
            class_names,
            inv_transform,
            latent_to_pixel,
            patches_dir,
            decision_flow_dir,
        )


def _save_decision_flows(
    leaf_rationalization: LeafRationalization,
    true_label: int,
    class_names: tuple,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    patches_dir: os.PathLike,
    explanation_dir: os.PathLike,
):
    dag = pydot.Dot(
        "Decision flow for explanation of 1 image.",
        graph_type="digraph",
        rankdir="LR",
    )
    proto_pydot_nodes, bbox_pydot_nodes = [], []
    decision_pydot_edges, bbox_pydot_edges = [], []

    leaf_pydot_node = gen_leaf(leaf_rationalization.leaf, class_names)

    for ancestor_similarity, went_right in zip(
        leaf_rationalization.ancestor_similarities,
        leaf_rationalization.ancestors_went_right,
    ):
        (_, _, im_with_bbox, _) = closest_patch_imgs(
            ancestor_similarity, inv_transform, latent_to_pixel
        )  # Other return values are unused for now, but we could easily change this.

        proto_node = ancestor_similarity.internal_node
        proto_file = patches_dir / f"{proto_node.index}_closest_patch.png"
        proto_pydot_node = pydot.Node(
            _node_name(proto_node),
            image=f'"{proto_file}"',
            imagescale=True,
            fixedsize=True,
            height=1.5,
            width=1.5,
            label="",
            shape="plaintext",
        )
        proto_pydot_nodes.append(proto_pydot_node)

        similarity = ancestor_similarity.highest_patch_similarity
        if went_right:
            bbox_file = explanation_dir / f"level_{proto_node.depth}_bounding_box.png"
            save_img(im_with_bbox, bbox_file)
            bbox_pydot_node = pydot.Node(
                _bbox_node_name(proto_node),
                image=f'"{bbox_file}"',
                imagescale=True,
                fixedsize=True,
                height=2,
                width=2,
                label="",
                shape="plaintext",
            )
            bbox_pydot_nodes.append(bbox_pydot_node)

            decision_edge = pydot.Edge(
                _node_name(proto_node),
                _node_name(proto_node.right),
                label=f"Present\nSimilarity={similarity:.5f}",
                weight=100,
            )

            bbox_pydot_edge = pydot.Edge(
                _node_name(proto_node),
                _bbox_node_name(proto_node),
                style="dashed",
                dir="none",
                tailport="s",
                headport="n",
            )
            bbox_pydot_edges.append(bbox_pydot_edge)
        else:
            decision_edge = pydot.Edge(
                _node_name(proto_node),
                _node_name(proto_node.left),
                label=f"Absent\nSimilarity={similarity:.5f}",
            )
        decision_pydot_edges.append(decision_edge)

    original_im = inv_transform(
        leaf_rationalization.ancestor_similarities[0].transformed_image
    )
    original_file = explanation_dir / "original.png"
    save_img(original_im, original_file)

    true_name = class_names[true_label]
    original_im_node = pydot.Node(
        "original",
        image=f'"{original_file}"',
        imagescale=True,
        fixedsize=True,
        height=2,
        width=2,
        label="",
        shape="plaintext",
    )
    original_label_node = pydot.Node(
        "original_label",
        label=f"Test image\n{true_name}",
        lp="c",
        fontname=FONT,
        shape="plaintext",
    )
    original_label_edge = pydot.Edge(
        "original_label", "original", weight=100, style="invis"
    )
    original_to_proto_edge = pydot.Edge(
        "original",
        _node_name(leaf_rationalization.ancestor_similarities[0].internal_node),
        weight=100,
    )
    original_nodes = [original_label_node, original_im_node]
    original_edges = [original_label_edge, original_to_proto_edge]

    pydot_nodes = (
        proto_pydot_nodes + bbox_pydot_nodes + [leaf_pydot_node] + original_nodes
    )
    pydot_edges = decision_pydot_edges + bbox_pydot_edges + original_edges
    for pydot_node in pydot_nodes:
        dag.add_node(pydot_node)
    for pydot_edge in pydot_edges:
        dag.add_edge(pydot_edge)

    dot_file = explanation_dir / "explanation.dot"
    dag.write_dot(dot_file)

    png_file = explanation_dir / "explanation.png"
    dag.write_png(png_file)


def _bbox_node_name(node: Node) -> str:
    return f"bbox_{node.index}"
