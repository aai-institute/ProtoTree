import argparse
import copy
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prototree.node import InternalNode, Leaf, Node
from prototree.prototree import ProtoTree


def gen_vis(
    tree: ProtoTree,
    folder_name: str,
    classes: tuple,
    log_dir,
    dir_for_saving_images,
):
    log_dir = Path(log_dir)
    destination_folder = log_dir / folder_name
    upsample_dir = log_dir / dir_for_saving_images / folder_name
    node_vis_dir = destination_folder / "node_vis"

    node_vis_dir.mkdir(parents=True, exist_ok=True)
    upsample_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        s += 'node [shape=rect, label=""];\n'
        s += _gen_dot_nodes(tree.root, destination_folder, upsample_dir, classes)
        s += _gen_dot_edges(tree.root, classes)[0]
        s += "}\n"

    with open(os.path.join(destination_folder, "treevis.dot"), "w") as f:
        f.write(s)

    # TODO: this requires dot to be installed. We probably don't want a pdf anyway
    # from_p = os.path.join(destination_folder, "treevis.dot")
    # to_pdf = os.path.join(destination_folder, "treevis.pdf")
    # check_call(f"dot -Tpdf -Gmargin=0 {from_p} -o {to_pdf}", shell=True)


def _node_vis(node: Node, upsample_dir: str):
    if isinstance(node, Leaf):
        return _leaf_vis(node)
    if isinstance(node, InternalNode):
        return _internal_node_vis(node, upsample_dir)


def _leaf_vis(node: Leaf):
    if node.log_probabilities:
        ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        ws = copy.deepcopy(node.distribution().cpu().detach().numpy())

    ws = np.ones(ws.shape) - ws
    ws *= 255

    height = 24

    if ws.shape[0] < 36:
        img_size = 36
    else:
        img_size = ws.shape[0]
    scaler = math.ceil(img_size / ws.shape[0])

    img = Image.new("F", (ws.shape[0] * scaler, height))
    pixels = img.load()

    for i in range(scaler * ws.shape[0]):
        for j in range(height - 10):
            pixels[i, j] = ws[int(i / scaler)]
        for j in range(height - 10, height - 9):
            pixels[i, j] = 0  # set bottom line of leaf distribution black
        for j in range(height - 9, height):
            pixels[
                i, j
            ] = 255  # set bottom part of node white such that class label is readable

    if scaler * ws.shape[0] > 100:
        img = img.resize((100, height))
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

    # TODO: this is profoundly broken
    if wbb < 100 and hbb < 100:
        cs = wbb, hbb
    else:
        cs = 100 / wbb, 100 / hbb
        min_cs = min(cs)
        bb = bb.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb.size

    if w < 100 and h < 100:
        cs = w, h
    else:
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
    node: Node, destination_folder: str, upsample_dir: str, classes: tuple
):
    img = _node_vis(node, upsample_dir).convert("RGB")
    if isinstance(node, Leaf):
        if node.log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
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
        if node.log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        return "", class_targets
