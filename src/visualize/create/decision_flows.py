import os
from collections.abc import Callable
from typing import Iterator

import numpy as np
import pydot
import torch
from tqdm import tqdm

from prototree.img_similarity import ImageProtoSimilarity
from prototree.models import LeafRationalization
from prototree.node import Node
from util.data import save_img
from util.image import get_latent_to_pixel, get_inverse_arr_transform
from visualize.create.dot import _node_name, gen_leaf, FONT, graph_with_components
from visualize.create.patches import closest_patch_imgs


@torch.no_grad()
def save_decision_flow_visualizations(
    explanations: Iterator[tuple[LeafRationalization, str, tuple]],
    patches_dir: os.PathLike,
    explanations_dir: os.PathLike,
    img_size=(224, 224),
):
    """
    Saves visualizations of each explanation as a DOT file and png.
    TODO: Note that this currently relies on the patch visualizations being run first, we should probably change this,
     or change the API to enforce it.
    """
    decision_flows_dir = explanations_dir / "decision_flows"
    decision_flows_dir.mkdir(parents=True, exist_ok=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    tqdm_explanations = tqdm(explanations, desc="Visualizing decision flows", ncols=0)
    for explanation_counter, (leaf_explanation, true_class, class_names) in enumerate(
        tqdm_explanations
    ):
        decision_flow_dir = decision_flows_dir / f"img_{explanation_counter}"
        flow_dag = _decision_flow_dag(
            leaf_explanation,
            true_class,
            class_names,
            inv_transform,
            latent_to_pixel,
            patches_dir,
            decision_flow_dir,
        )
        _save_pydot(flow_dag, decision_flow_dir)


def _save_pydot(flow_dag: pydot.Dot, decision_flow_dir: os.PathLike):
    dot_file = decision_flow_dir / "explanation.dot"
    flow_dag.write_dot(dot_file)

    png_file = decision_flow_dir / "explanation.png"
    flow_dag.write_png(png_file)


def _decision_flow_dag(
    leaf_rationalization: LeafRationalization,
    true_class: str,
    class_names: tuple,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    patches_dir: os.PathLike,
    decision_flow_dir: os.PathLike,
):
    proto_subgraphs, decision_pydot_edges = [], []
    for ancestor_similarity, went_right in zip(
        leaf_rationalization.ancestor_similarities,
        leaf_rationalization.ancestors_went_right,
    ):
        proto_subgraph, decision_edge = _proto_node_components(
            ancestor_similarity,
            went_right,
            inv_transform,
            latent_to_pixel,
            patches_dir,
            decision_flow_dir,
        )
        decision_pydot_edges.append(decision_edge)
        proto_subgraphs.append(proto_subgraph)

    original_nodes, original_edges = _original_im_components(
        inv_transform,
        leaf_rationalization.ancestor_similarities[0],
        true_class,
        decision_flow_dir,
    )
    leaf_pydot_node = gen_leaf(leaf_rationalization.leaf, class_names)

    pydot_nodes = [leaf_pydot_node] + original_nodes
    pydot_edges = decision_pydot_edges + original_edges
    return _assemble_flow_dag(pydot_nodes, proto_subgraphs, pydot_edges)


def _assemble_flow_dag(
    nodes: list[pydot.Node], subgraphs: list[pydot.Subgraph], edges: list[pydot.Edge]
) -> pydot.Dot:
    flow_dag = pydot.Dot(
        "Decision flow for explanation of an image.",
        graph_type="digraph",
        rankdir="LR",
    )
    return graph_with_components(flow_dag, nodes, subgraphs, edges)


def _proto_node_components(
    ancestor_similarity: ImageProtoSimilarity,
    went_right: bool,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    patches_dir: os.PathLike,
    decision_flow_dir: os.PathLike,
) -> tuple[pydot.Subgraph, pydot.Edge]:
    """
    Produces the components of the graph that correspond to the decision-making with the prototype at a single node in
    the tree. This consists of a subgraph of {prototype visualization, (optional) bounding box for the matching patch on
    the image, (optional) edge connecting the two images}, and an edge leading to the next node in the tree.
    """
    (_, _, im_with_bbox, _) = closest_patch_imgs(
        ancestor_similarity, inv_transform, latent_to_pixel
    )  # Other return values are unused for now, but we could easily change this.
    proto_node = ancestor_similarity.internal_node
    proto_file = patches_dir / f"{proto_node.index}_closest_patch.png"

    proto_subgraph = pydot.Subgraph(f"proto_subgraph_{proto_node.depth}", rank="same")

    proto_pydot_node = _img_pydot_node(_node_name(proto_node), proto_file, 1.5)
    proto_subgraph.add_node(proto_pydot_node)

    similarity = ancestor_similarity.highest_patch_similarity
    if went_right:
        bbox_file = decision_flow_dir / f"level_{proto_node.depth}_bounding_box.png"
        save_img(im_with_bbox, bbox_file)
        bbox_pydot_node = _img_pydot_node(_bbox_node_name(proto_node), bbox_file, 2.0)

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
            minlen=2,
        )

        proto_subgraph.add_node(bbox_pydot_node)
        proto_subgraph.add_edge(bbox_pydot_edge)
    else:
        decision_edge = pydot.Edge(
            _node_name(proto_node),
            _node_name(proto_node.left),
            label=f"Absent\nSimilarity={similarity:.5f}",
        )
    return proto_subgraph, decision_edge


def _original_im_components(
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    root_similarity: ImageProtoSimilarity,
    true_class: str,
    decision_flow_dir: os.PathLike,
) -> tuple[list[pydot.Node], list[pydot.Edge]]:
    original_im = inv_transform(root_similarity.transformed_image)
    original_file = decision_flow_dir / "original.png"
    save_img(original_im, original_file)

    original_im_node = _img_pydot_node("original", original_file, 2.0)
    original_label_node = pydot.Node(
        "original_label",
        label=f"Test image\n{true_class}",
        lp="c",
        fontname=FONT,
        shape="plaintext",
    )
    original_label_edge = pydot.Edge(
        "original_label", "original", weight=100, style="invis", minlen=0.5
    )
    original_to_proto_edge = pydot.Edge(
        "original",
        _node_name(root_similarity.internal_node),
        weight=100,
    )
    original_nodes = [original_label_node, original_im_node]
    original_edges = [original_label_edge, original_to_proto_edge]

    return original_nodes, original_edges


def _img_pydot_node(node_name: str, im_file: os.PathLike, size: float) -> pydot.Node:
    """
    Creates a pydot node which resizes the im_file to the specified size..
    """
    return pydot.Node(
        node_name,
        image=f'"{im_file}"',
        imagescale=True,
        fixedsize=True,
        height=size,
        width=size,
        label="",
        shape="plaintext",
    )


def _bbox_node_name(node: Node) -> str:
    return f"bbox_{node.index}"
