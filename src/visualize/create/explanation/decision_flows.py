import logging
import os
from collections.abc import Callable
from typing import Iterator
from pathlib import Path
import pandas as pd
import numpy as np
import pydot
import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from src.core.img_similarity import ImageProtoSimilarity
from src.core.models import ProtoTree
from src.core.node import Node, InternalNode
from src.util.data import save_img
from src.util.image import get_latent_to_pixel, get_inverse_arr_transform
from src.visualize.create.dot import _node_name, gen_leaf, graph_with_components
from src.visualize.create.explanation.common import _original_im_components, _img_pydot_node
from src.visualize.create.patches import closest_patch_imgs

log = logging.getLogger(__name__)


@torch.no_grad()
def save_decision_flow_visualizations(
    explanations: Iterator[tuple[ProtoTree.LeafRationalization, str, tuple]],
    protos_info: dict[str, dict], 
    explanations_dir: os.PathLike,
    patches_dir: os.PathLike,
    scores: pd.DataFrame = None, 
    img_size=(224, 224),
):
    """
    Saves visualizations of each explanation as a DOT file and png.
    """
    decision_flows_dir = explanations_dir / "decision_flows"
    decision_flows_dir.mkdir(parents=True, exist_ok=True)
    inv_transform = get_inverse_arr_transform(img_size)
    latent_to_pixel = get_latent_to_pixel(img_size)

    log.info(
        f"Saving decision flow visualizations of the explanations to {decision_flows_dir}."
    )
    tqdm_explanations = tqdm(
        explanations,
        desc="Saving decision flow visualizations of the explanations",
        ncols=0,
    )
    # Leaf explanation contains a list of 9 ancestors sim [node, imagesimilarity (with proto_id)] objects
    
    for explanation_counter, sample in enumerate(
        tqdm_explanations
    ):
        leaf_explanation, true_class, class_names = sample[0], sample[1], sample[2]
        
        if scores is not None:
            img_path = sample[3]
            decision_flow_dir = decision_flows_dir / os.path.basename(img_path).split(".")[0]
            img_scores = scores.loc[scores["image"] == img_path]
        else:
            decision_flow_dir = decision_flows_dir / f"img_{explanation_counter}" #img_path
            img_scores = None
            
        flow_dag = _decision_flow_dag(
            leaf_explanation,
            true_class,
            class_names,
            inv_transform,
            latent_to_pixel,
            protos_info,
            decision_flow_dir,
            patches_dir,
            img_scores
        )
        _save_pydot(flow_dag, decision_flow_dir)


def _save_pydot(flow_dag: pydot.Dot, decision_flow_dir: os.PathLike):
    dot_file = decision_flow_dir / "explanation.dot"
    flow_dag.write_dot(dot_file)

    png_file = decision_flow_dir / "explanation.png"
    flow_dag.write_png(png_file)


def _decision_flow_dag(
    leaf_rationalization: ProtoTree.LeafRationalization,
    true_class: str,
    class_names: tuple,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    protos_info: dict[str, dict],
    decision_flow_dir: os.PathLike,
    patches_dir: os.PathLike,
    proto_scores: pd.DataFrame = None
) -> pydot.Dot:
    # TODO: There's a lot of parameters to this function (and some of the others further down this file). We could "fix"
    #  this by making a class to hold several parameters, but it's not clear to me what the right class would be, or
    #  whether that would actually be a useful abstraction (i.e. we probably don't just want to put lots of unrelated
    #  items into a class to hide the number of arguments, particularly if we end up unpacking them from the class
    #  almost immediately). Perhaps the number of parameters indicates we need a design rethink for this part of the
    #  code.
    """
    Produces the pydot graph for the decision flow. This function (and others it calls) save required images with
    side effects.
    """

    proto_subgraphs, decision_pydot_edges = [], []
    for ancestor_sim, proto_present in zip(
        leaf_rationalization.ancestor_sims,
        leaf_rationalization.proto_presents() 
    ):
        proto_subgraph, decision_edge = _proto_node_components(
            ancestor_sim,
            proto_present,
            inv_transform,
            latent_to_pixel,
            protos_info,
            decision_flow_dir,
            patches_dir, 
            proto_scores
        )
        decision_pydot_edges.append(decision_edge)
        proto_subgraphs.append(proto_subgraph)

    first_ancestor_sim = leaf_rationalization.ancestor_sims[0]
    original_nodes, original_edges = _original_im_components(
        inv_transform,
        first_ancestor_sim.similarity.transformed_image,
        _node_name(first_ancestor_sim.node),
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
    ancestor_sim: ProtoTree.LeafRationalization.NodeSimilarity,
    proto_present: bool,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    protos_info: dict[str, dict],
    decision_flow_dir: os.PathLike,
    patches_dir: os.PathLike, 
    protos_scores: pd.DataFrame = None
) -> tuple[pydot.Subgraph, pydot.Edge]:
    """
    Produces the components of the graph that correspond to the decision-making with the prototype at a single node in
    the tree. This consists of a subgraph of {prototype visualization, (optional) bounding box for the matching patch on
    the image, (optional) edge connecting the two images}, and an edge leading to the next node in the tree.
    """
    proto_img = plt.imread(protos_info[str(ancestor_sim.node.index)]["path"])
    patch_bbox = protos_info[str(ancestor_sim.node.index)]["bbox"]
    proto_patch = proto_img[patch_bbox[1]:patch_bbox[3], patch_bbox[0]:patch_bbox[2]] # check PIL bbox as needs to be done x,y,w,h, or x1,y1,x2,y2img_orig[bb[1]:bb[3], bb[0]:bb[2]]
    
    n = 2 if protos_scores is not None else 1
    fig, axs = plt.subplots(1, n)
    axs[0].imshow(proto_patch)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
     
    if protos_scores is not None:   
        proto_scores = round(protos_scores.loc[protos_scores["prototype"] == ancestor_sim.node.index], 4)
        proto_scores.plot(x="modification", y="delta", ax=axs[1], kind="bar", legend=False)
        axs[1].bar_label(axs[1].containers[0])
        axs[1].set_yticks([])
        axs[1].set_facecolor('0.9')

    proto_file = patches_dir / f"{ancestor_sim.node.index}_closest_patch.png"
    plt.savefig(proto_file) 
    
    proto_subgraph = pydot.Subgraph(
        f"proto_subgraph_{ancestor_sim.node.depth}", rank="same"
    )

    proto_pydot_node = _img_pydot_node(_node_name(ancestor_sim.node), proto_file, 1.5)
    proto_subgraph.add_node(proto_pydot_node)
    if proto_present:
        bbox_pydot_node, bbox_pydot_edge = _bbox_components(
            ancestor_sim.similarity,
            ancestor_sim.node,
            inv_transform,
            latent_to_pixel,
            decision_flow_dir,
            proto_file
        )
        proto_subgraph.add_node(bbox_pydot_node)
        proto_subgraph.add_edge(bbox_pydot_edge)

    decision_edge = _decision_edge(
        ancestor_sim.node,
        proto_present,
        ancestor_sim.similarity.highest_patch_similarity,
    )
    return proto_subgraph, decision_edge


def _bbox_components(
    ancestor_sim: ImageProtoSimilarity,
    proto_node: InternalNode,
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    latent_to_pixel: Callable[[np.ndarray], np.ndarray],
    decision_flow_dir: os.PathLike,
    proto_file: str,
):
    """
    Produces the bounding box image node and dotted line edge that appear if the prototype patch is present in the
    image.
    """
    (_, _, im_with_bbox, _) = closest_patch_imgs(
        ancestor_sim, inv_transform, latent_to_pixel
    )  # Other return values are unused for now, but we could easily change this.

    bbox_file = decision_flow_dir / f"level_{proto_node.depth}_bounding_box.png"
    save_img(im_with_bbox, bbox_file)
    bbox_pydot_node = _img_pydot_node(_bbox_node_name(proto_node), bbox_file, 2.0)
    
    bbox_pydot_edge = pydot.Edge(
        _node_name(proto_node),
        _bbox_node_name(proto_node), #proto_pydot_node, #
        style="dashed",
        dir="none",
        tailport="s",
        headport="n",
        minlen=2,
    )

    return bbox_pydot_node, bbox_pydot_edge 


def _decision_edge(
    proto_node: InternalNode, proto_present: bool, similarity: float
) -> pydot.Edge:
    """
    This function produces the main arrow edges that show the progress down the tree as we reach each InternalNode.
    """
    if proto_present:
        proto_presence, next_node = "Present", proto_node.right
    else:
        proto_presence, next_node = "Absent", proto_node.left

    return pydot.Edge(
        _node_name(proto_node),
        _node_name(next_node),
        label=f"{proto_presence}\nSimilarity={similarity:.5f}",
    )


def _bbox_node_name(node: Node) -> str:
    return f"bbox_{node.index}"
