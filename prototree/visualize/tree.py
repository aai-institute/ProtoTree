import copy
import logging
import math
import os

import numpy as np
import pydot
import torch
from PIL import Image

from prototree.models import ProtoTree
from prototree.node import InternalNode, Leaf, Node

log = logging.getLogger(__name__)

FONT = "Helvetica"
EDGE_ATTRS = dict(fontsize=10, tailport="s", headport="n", fontname=FONT)


@torch.no_grad()
def save_tree_visualization(
        tree: ProtoTree,
        patches_path: os.PathLike,
        save_path: os.PathLike,
        class_names: tuple,
):
    """
    Saves visualization as a dotfile (and as pdf, if supported)
    """
    node_vis_path = save_path / "node_vis"
    node_vis_path.mkdir(parents=True, exist_ok=True)

    pydot_tree = _gen_pydot_tree(tree.tree_root, patches_path, node_vis_path, class_names)

    dot_file = save_path / "tree.dot"
    log.info(f"Saving tree dot to {dot_file}, this file is just for debugging/further processing, and is not directly "
             f"used in image output generation.")

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

    if isinstance(subtree_root, Leaf):
        ws = copy.deepcopy(torch.exp(subtree_root.logits()).detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = class_names[t]
        predicted_classes = (
            ",".join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        )
        predicted_classes = predicted_classes.replace("_", " ")

        pydot_node = pydot.Node(subtree_root.index, image=f'"{img_file}"', label=predicted_classes, imagepos="tc",
                                imagescale="height", labelloc="b", fontsize=10, penwidth=0, fontname=FONT)
    else:
        pydot_node = pydot.Node(subtree_root.index, image=f'"{img_file}"', xlabel=f'"{subtree_root.index}"', fontsize=6,
                                labelfontcolor="gray50", fontname=FONT, shape="box")

    if isinstance(subtree_root, InternalNode):
        l_descendants = _gen_pydot_nodes(subtree_root.left, patches_path, node_vis_path, class_names)
        r_descendants = _gen_pydot_nodes(subtree_root.right, patches_path, node_vis_path, class_names)
        subtree_pydot_nodes = [pydot_node] + l_descendants + r_descendants
    elif isinstance(subtree_root, Leaf):
        subtree_pydot_nodes = [pydot_node]
    else:
        raise ValueError(f"Unknown node {subtree_root}.")

    return subtree_pydot_nodes


def _gen_pydot_edges(subtree_root: Node) -> list[pydot.Edge]:
    if isinstance(subtree_root, InternalNode):
        l_descendants = _gen_pydot_edges(subtree_root.left)
        r_descendants = _gen_pydot_edges(subtree_root.right)
        l_edge = pydot.Edge(subtree_root.index, subtree_root.left.index, label="Absent", **EDGE_ATTRS)
        r_edge = pydot.Edge(subtree_root.index, subtree_root.right.index, label="Present", **EDGE_ATTRS)
        return [l_edge, r_edge] + l_descendants + r_descendants
    if isinstance(subtree_root, Leaf):
        return []


def _gen_node_rgb(node: Node, patches_path: os.PathLike):
    if isinstance(node, Leaf):
        img = _gen_leaf_img(node)
    elif isinstance(node, InternalNode):
        img = _gen_internal_node_img(node, patches_path)
    else:
        raise ValueError(f"Unknown node {node}.")

    return img.convert("RGB")


def _gen_leaf_img(node: Leaf):
    pixel_depth = 255
    height = 24
    footer_height = 10
    max_width = 100

    distribution = copy.deepcopy(node.y_proba().detach().numpy())
    distribution = (np.ones(distribution.shape) - distribution) * pixel_depth
    num_classes = len(distribution)

    width = min(36, num_classes)
    scaler = math.ceil(width / num_classes)
    # correcting potential off-by-one errors
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
            # set bottom part of node white such that class label is readable
            pixels[i, j] = pixel_depth

    if width > max_width:
        img = img.resize((max_width, height))
    return img


def _gen_internal_node_img(node: InternalNode, patches_path: os.PathLike):
    internal_node_id = node.index
    # TODO: move hardcoded strings to config
    patch_img = Image.open(
        os.path.join(patches_path, f"{internal_node_id}_closest_patch.png")
    )
    bb_img = Image.open(
        os.path.join(
            patches_path, f"{internal_node_id}_bounding_box_closest_patch.png"
        )
    )
    w, h = patch_img.size
    wbb, hbb = bb_img.size

    # TODO: duplication
    if wbb > 100 or hbb > 100:
        cs = 100 / wbb, 100 / hbb
        min_cs = min(cs)
        bb_img = bb_img.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb_img.size

    if w > 100 or h > 100:
        cs = 100 / w, 100 / h
        min_cs = min(cs)
        patch_img = patch_img.resize(size=(int(min_cs * w), int(min_cs * h)))
        w, h = patch_img.size

    between = 4
    total_w = w + wbb + between
    total_h = max(h, hbb)

    together = Image.new(patch_img.mode, (total_w, total_h), color=(255, 255, 255))
    together.paste(patch_img, (0, 0))
    together.paste(bb_img, (w + between, 0))

    return together
