import copy
import logging
import math
import os
from subprocess import check_call

import numpy as np
import torch
from PIL import Image

from prototree.models import ProtoTree
from prototree.node import InternalNode, Leaf, Node

log = logging.getLogger(__name__)


# TODO: use pydot
@torch.no_grad()
def save_tree_visualization(
    tree: ProtoTree,
    patches_path: os.PathLike,
    save_path: os.PathLike,
    class_names: tuple,
):
    """
    Saves visualization as a dotfile (and as pdf, if supported)

    :param tree:
    :param patches_path:
    :param save_path:
    :param class_names:
    :return:
    """
    node_vis_path = save_path / "node_vis"
    node_vis_path.mkdir(parents=True, exist_ok=True)

    s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
    s += 'node [shape=rect, label=""];\n'
    s += _gen_dot_nodes(tree.tree_root, patches_path, node_vis_path, class_names)
    s += _gen_dot_edges(tree.tree_root, class_names)[0]
    s += "}\n"

    dot_file = save_path / "tree.dot"
    log.info(f"Saving tree visualization to {dot_file}")
    with open(dot_file, "w") as f:
        f.write(s)

    # Save as pdf using graphviz
    try:
        pdf_file = save_path / "treevis.pdf"
        check_call(["dot", "-Tpdf", "-Gmargin=0", dot_file, "-o", pdf_file])
        log.info(f"Saved tree visualization as pdf to {pdf_file}")
    except FileNotFoundError:
        log.error(
            f"Could not find graphviz, skipping generation of pdf. "
            f"Please install it and make sure it is available on the PATH to generate pdfs. "
            f"See https://graphviz.org/ for instructions.."
        )


def _gen_dot_nodes(
    node: Node,
    patches_path: os.PathLike,
    node_vis_path: os.PathLike,
    class_names: tuple,
):
    img = _gen_node_rgb(node, patches_path)
    if isinstance(node, Leaf):
        ws = copy.deepcopy(torch.exp(node.logits()).detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = class_names[t]
        str_targets = (
            ",".join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        )
        str_targets = str_targets.replace("_", " ")

    # TODO: Perhaps we should extract some pure functions here.
    fname = node_vis_path / f"node_{node.index}_vis.jpg"
    img.save(fname)

    if isinstance(node, Leaf):
        s = (
            f'{node.index}[imagepos="tc" imagescale=height image="{fname}" '
            f'label="{str_targets}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'
        )
    else:
        s = (
            f'{node.index}[image="{fname}" xlabel="{node.index}" '
            f"fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n"
        )
    if isinstance(node, InternalNode):
        return (
            s
            + _gen_dot_nodes(node.left, patches_path, node_vis_path, class_names)
            + _gen_dot_nodes(node.right, patches_path, node_vis_path, class_names)
        )
    if isinstance(node, Leaf):
        return s


def _gen_dot_edges(node: Node, class_names: tuple):
    if isinstance(node, InternalNode):
        edge_l, targets_l = _gen_dot_edges(node.left, class_names)
        edge_r, targets_r = _gen_dot_edges(node.right, class_names)
        s = (
            f'{node.index} -> {node.left.index} [label="Absent" fontsize=10 tailport="s" headport="n" '
            f"fontname=Helvetica];\n {node.index} -> "
            f'{node.right.index} [label="Present" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n'
        )
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
    if isinstance(node, Leaf):
        ws = copy.deepcopy(torch.exp(node.logits()).detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = class_names[t]
        return "", class_targets


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
