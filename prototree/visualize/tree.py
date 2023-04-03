import copy
import logging
import math
import os
from pathlib import Path
from subprocess import check_call

import numpy as np
import torch
from PIL import Image

from prototree.models import ProtoTree
from prototree.node import InternalNode, Leaf, Node

log = logging.getLogger(__name__)


# TODO: use pydot
@torch.no_grad()
def generate_tree_visualization(
    tree: ProtoTree,
    classes: tuple,
    folder_name: str,
    log_dir: os.PathLike,
    dir_for_saving_images: os.PathLike,
):
    """
    Saves visualization as a dotfile (and as pdf, if supported)

    :param tree:
    :param classes:
    :param folder_name:
    :param log_dir:
    :param dir_for_saving_images:
    :return:
    """
    log_dir = Path(log_dir)
    destination_folder = log_dir / folder_name
    upsample_dir = log_dir / dir_for_saving_images / folder_name
    node_vis_dir = destination_folder / "node_vis"

    node_vis_dir.mkdir(parents=True, exist_ok=True)
    upsample_dir.mkdir(parents=True, exist_ok=True)

    s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
    s += 'node [shape=rect, label=""];\n'
    s += _gen_dot_nodes(tree.tree_root, destination_folder, upsample_dir, classes)
    s += _gen_dot_edges(tree.tree_root, classes)[0]
    s += "}\n"

    dot_file = destination_folder / "tree.dot"
    log.info(f"Saving tree visualization to {dot_file}")
    with open(dot_file, "w") as f:
        f.write(s)

    # Save as pdf using graphviz
    try:
        pdf_file = destination_folder / "treevis.pdf"
        check_call(["dot", "-Tpdf", "-Gmargin=0", dot_file, "-o", pdf_file])
        log.info(f"Saved tree visualization as pdf to {pdf_file}")
    except FileNotFoundError:
        log.error(
            f"Could not find graphviz, skipping generation of pdf. "
            f"Please install it and make sure it is available on the PATH to generate pdfs. "
            f"See https://graphviz.org/ for instructions.."
        )


def _node_vis(node: Node, upsample_dir: str):
    if isinstance(node, Leaf):
        return _leaf_vis(node)
    if isinstance(node, InternalNode):
        return _internal_node_vis(node, upsample_dir)


def _leaf_vis(node: Leaf):
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


def _internal_node_vis(node: InternalNode, upsample_dir: str):
    internal_node_id = node.index
    # TODO: move hardcoded strings to config
    img = Image.open(
        os.path.join(upsample_dir, f"{internal_node_id}_nearest_patch_of_image.png")
    )
    bb = Image.open(
        os.path.join(
            upsample_dir, f"{internal_node_id}_bounding_box_nearest_patch_of_image.png"
        )
    )
    # TODO: was map ever used?
    # map = Image.open(
    #     os.path.join(upsample_dir, f"{internal_node_id}_heatmap_original_image.png")
    # )
    w, h = img.size
    wbb, hbb = bb.size

    # TODO: duplication
    if wbb > 100 or hbb > 100:
        cs = 100 / wbb, 100 / hbb
        min_cs = min(cs)
        bb = bb.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb.size

    if w > 100 or h > 100:
        cs = 100 / w, 100 / h
        min_cs = min(cs)
        img = img.resize(size=(int(min_cs * w), int(min_cs * h)))
        w, h = img.size

    between = 4
    total_w = w + wbb + between
    total_h = max(h, hbb)

    together = Image.new(img.mode, (total_w, total_h), color=(255, 255, 255))
    together.paste(img, (0, 0))
    together.paste(bb, (w + between, 0))

    return together


def _gen_dot_nodes(
    node: Node,
    destination_folder: os.PathLike,
    upsample_dir: os.PathLike,
    classes: tuple,
):
    img = _node_vis(node, upsample_dir).convert("RGB")
    if isinstance(node, Leaf):
        ws = copy.deepcopy(torch.exp(node.logits()).detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        str_targets = (
            ",".join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        )
        str_targets = str_targets.replace("_", " ")
    filename = "{}/node_vis/node_{}_vis.jpg".format(destination_folder, node.index)
    img.save(filename)
    if isinstance(node, Leaf):
        s = (
            f'{node.index}[imagepos="tc" imagescale=height image="{filename}" '
            f'label="{str_targets}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'
        )
    else:
        s = (
            f'{node.index}[image="{filename}" xlabel="{node.index}" '
            f"fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n"
        )
    if isinstance(node, InternalNode):
        return (
            s
            + _gen_dot_nodes(node.left, destination_folder, upsample_dir, classes)
            + _gen_dot_nodes(node.right, destination_folder, upsample_dir, classes)
        )
    if isinstance(node, Leaf):
        return s


def _gen_dot_edges(node: Node, classes: tuple):
    if isinstance(node, InternalNode):
        edge_l, targets_l = _gen_dot_edges(node.left, classes)
        edge_r, targets_r = _gen_dot_edges(node.right, classes)
        # TODO: unused vars
        str_targets_l = (
            ",".join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
        )
        str_targets_r = (
            ",".join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
        )
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
            class_targets[i] = classes[t]
        return "", class_targets
