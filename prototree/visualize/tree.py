import copy
import logging
import math
import os

import numpy as np
import pydot
import torch
from PIL import Image, ImageOps

from prototree.models import ProtoTree
from prototree.node import InternalNode, Leaf, Node

log = logging.getLogger(__name__)

# TODO: See if there's a non-clunky (i.e. without defining for the base class) way to do single dispatch on `Node`s in
#  this file.

# TODO: Less hardcoding (particularly of numbers), both here and elsewhere in the file.
FONT = "Helvetica"
EDGE_ATTRS = dict(fontsize=10, tailport="s", headport="n", fontname=FONT)
SINGLE_NODE_IMG_SIZE = (100, 100)
INTERNAL_NODE_IMG_GAP = 4


@torch.no_grad()
def save_tree_visualization(
        tree: ProtoTree,
        patches_path: os.PathLike,
        save_path: os.PathLike,
        class_names: tuple,
):
    """
    Saves visualization as a DOT file and png.
    """
    node_vis_path = save_path / "node_vis"
    node_vis_path.mkdir(parents=True, exist_ok=True)

    pydot_tree = _gen_pydot_tree(tree.tree_root, patches_path, node_vis_path, class_names)

    dot_file = save_path / "tree.dot"
    log.info(f"Saving tree DOT to {dot_file}, this file is just for debugging/further processing, and is not directly "
             f"used in image output generation.")
    pydot_tree.write_dot(dot_file)

    png_file = save_path / "treevis.png"
    log.info(f"Saving rendered tree to {png_file}")
    pydot_tree.write_png(png_file)


def _gen_pydot_tree(
        root: Node,
        patches_path: os.PathLike,
        node_vis_path: os.PathLike,
        class_names: tuple,
) -> pydot.Dot:
    pydot_tree = pydot.Dot("prototree", graph_type="digraph", bgcolor="white", margin=0.0, ranksep=0.03, nodesep=0.05,
                           splines=False)

    pydot_nodes = _gen_pydot_nodes(root, patches_path, node_vis_path, class_names)
    pydot_edges = _gen_pydot_edges(root)
    for pydot_node in pydot_nodes:
        pydot_tree.add_node(pydot_node)
    for pydot_edge in pydot_edges:
        pydot_tree.add_edge(pydot_edge)

    return pydot_tree


def _gen_pydot_nodes(
        subtree_root: Node,
        patches_path: os.PathLike,
        node_vis_path: os.PathLike,
        class_names: tuple,
) -> list[pydot.Node]:
    img = _gen_node_rgb(subtree_root, patches_path)
    # TODO: Perhaps we should extract some pure functions here.
    img_file = os.path.abspath(node_vis_path / f"node_{subtree_root.index}_vis.jpg")
    img.save(img_file)

    if isinstance(subtree_root, InternalNode):
        pydot_node = pydot.Node(subtree_root.index, image=f'"{img_file}"', xlabel=f'"{subtree_root.index}"', fontsize=6,
                                labelfontcolor="gray50", fontname=FONT, shape="box")
        l_descendants = _gen_pydot_nodes(subtree_root.left, patches_path, node_vis_path, class_names)
        r_descendants = _gen_pydot_nodes(subtree_root.right, patches_path, node_vis_path, class_names)
        return [pydot_node] + l_descendants + r_descendants
    if isinstance(subtree_root, Leaf):
        leaf_probs = copy.deepcopy(torch.exp(subtree_root.logits()).detach().numpy())
        max_prob_inds = np.argmax(leaf_probs, keepdims=True)
        predicted_classes = ",".join([class_names[i] for i in max_prob_inds])

        pydot_node = pydot.Node(subtree_root.index, image=f'"{img_file}"', label=predicted_classes, imagepos="tc",
                                imagescale="height", labelloc="b", fontsize=10, penwidth=0, fontname=FONT)
        return [pydot_node]

    raise ValueError(f"Unknown node {subtree_root}.")


def _gen_pydot_edges(subtree_root: Node) -> list[pydot.Edge]:
    if isinstance(subtree_root, InternalNode):
        l_descendants = _gen_pydot_edges(subtree_root.left)
        r_descendants = _gen_pydot_edges(subtree_root.right)
        l_edge = pydot.Edge(subtree_root.index, subtree_root.left.index, label="Absent", **EDGE_ATTRS)
        r_edge = pydot.Edge(subtree_root.index, subtree_root.right.index, label="Present", **EDGE_ATTRS)
        return [l_edge, r_edge] + l_descendants + r_descendants
    if isinstance(subtree_root, Leaf):
        return []

    raise ValueError(f"Unknown node {subtree_root}.")


def _gen_node_rgb(node: Node, patches_path: os.PathLike) -> Image:
    if isinstance(node, Leaf):
        img = _gen_leaf_img(node)
    elif isinstance(node, InternalNode):
        img = _gen_internal_node_img(node, patches_path)
    else:
        raise ValueError(f"Unknown node {node}.")

    return img.convert("RGB")


def _gen_leaf_img(node: Leaf) -> Image:
    pixel_depth = 255
    height = 24
    footer_height = 10

    node_probs = node.probs().detach().numpy()
    distribution = (np.ones(node_probs.shape) - node_probs) * pixel_depth
    num_classes = len(distribution)

    width = min(36, num_classes)
    # Correcting potential off-by-one errors.
    scaler = math.ceil(width / num_classes)
    width = scaler * num_classes

    img = Image.new("F", (width, height))
    # TODO: using img.load is discouraged, improve
    pixels = img.load()

    # TODO: Vectorize if this turns out to be a performance bottleneck.
    for i in range(width):
        for j in range(height - footer_height):
            pixels[i, j] = distribution[int(i / scaler)]
        # separate footer by black line
        pixels[i, height - footer_height] = 0
        for j in range(height - footer_height - 1, height):
            # Set the bottom part of the node white so that the class label is readable.
            pixels[i, j] = pixel_depth

    return ImageOps.contain(img, (SINGLE_NODE_IMG_SIZE[0], height))


def _gen_internal_node_img(node: InternalNode, patches_path: os.PathLike) -> Image:
    internal_node_id = node.index
    # TODO: move hardcoded strings to config
    patch_path = os.path.join(patches_path, f"{internal_node_id}_closest_patch.png")
    bb_path = os.path.join(patches_path, f"{internal_node_id}_bounding_box_closest_patch.png")
    patch_img_orig = Image.open(patch_path)
    bb_img_orig = Image.open(bb_path)

    bb_img = ImageOps.contain(bb_img_orig, SINGLE_NODE_IMG_SIZE)
    patch_img = ImageOps.contain(patch_img_orig, SINGLE_NODE_IMG_SIZE)
    wbb, hbb = bb_img.size
    w, h = patch_img.size

    total_w = w + INTERNAL_NODE_IMG_GAP + wbb
    total_h = max(h, hbb)

    together = Image.new(patch_img.mode, (total_w, total_h), color=(255, 255, 255))
    together.paste(patch_img, (0, 0))
    together.paste(bb_img, (w + INTERNAL_NODE_IMG_GAP, 0))

    return together
