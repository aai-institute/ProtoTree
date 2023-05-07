import logging
import os

import pydot
import torch
from PIL import Image, ImageOps

from proto.models import ProtoTree
from proto.node import InternalNode, Leaf, Node
from visualize.create.dot import (
    FONT,
    _node_name,
    gen_leaf,
    graph_with_components,
)

log = logging.getLogger(__name__)

# TODO: Less hardcoding (particularly of numbers), both here and elsewhere in the file.
EDGE_ATTRS = dict(fontsize=10, tailport="s", headport="n", fontname=FONT)
SINGLE_NODE_IMG_SIZE = (100, 100)
INTERNAL_NODE_IMG_GAP = 4


@torch.no_grad()
def save_tree_visualization(
    model: ProtoTree,
    patches_dir: os.PathLike,
    tree_dir: os.PathLike,
    class_names: tuple,
):
    """
    Saves visualization as a DOT file and png.
    TODO: Note that this currently relies on the patch visualizations being run first. We should probably change this,
     or change the API to enforce it.
    """
    node_imgs_dir = tree_dir / "node_imgs"
    node_imgs_dir.mkdir(parents=True, exist_ok=True)

    pydot_tree = _pydot_tree(
        model.tree_section.root,
        model.proto_patch_matches,
        patches_dir,
        node_imgs_dir,
        class_names,
    )
    _save_pydot(pydot_tree, tree_dir)


def _save_pydot(pydot_tree: pydot.Dot, tree_dir: os.PathLike):
    dot_file = tree_dir / "tree.dot"
    log.info(
        f"Saving tree DOT to {dot_file}, this file is just for debugging/further processing, and is not directly "
        f"used in image output generation."
    )
    pydot_tree.write_dot(dot_file)

    png_file = tree_dir / "treevis.png"
    log.info(f"Saving rendered tree to {png_file}")
    pydot_tree.write_png(png_file)


def _pydot_tree(
    root: Node, patches_dir: os.PathLike, node_imgs_dir: os.PathLike, class_names: tuple
) -> pydot.Dot:
    pydot_tree = pydot.Dot(
        "prototree",
        graph_type="digraph",
        bgcolor="white",
        margin=0.0,
        ranksep=0.03,
        nodesep=0.05,
        splines=False,
    )

    pydot_nodes = _pydot_nodes(root, patches_dir, node_imgs_dir, class_names)
    pydot_edges = _pydot_edges(root)
    return graph_with_components(pydot_tree, pydot_nodes, [], pydot_edges)


def _pydot_nodes(
    subtree_root: Node,
    patches_dir: os.PathLike,
    node_imgs_dir: os.PathLike,
    class_names: tuple,
) -> list[pydot.Node]:
    # TODO: How do we get a Julia-style by-default error when we can't dispatch? Or even better, some sort of typing
    #  error like in Rust, Scala etc? (same for everywhere else we match on node type) Specifying a union/enum manually
    #  and then using NoReturn seems to defeat the point of abstracting into a superclass.
    match subtree_root:
        case InternalNode() as internal_node:
            return _pydot_nodes_internal(
                internal_node, patches_dir, node_imgs_dir, class_names
            )
        case Leaf() as leaf:
            return _pydot_nodes_leaf(leaf, class_names)
        case other:
            raise ValueError(f"Unrecognized node {other}.")


def _pydot_nodes_internal(
    subtree_root: InternalNode,
    patches_dir: os.PathLike,
    node_imgs_dir: os.PathLike,
    class_names: tuple,
) -> list[pydot.Node]:
    img = _gen_internal_node_img(subtree_root, patches_dir)
    # TODO: Perhaps we should extract some pure functions here.
    img_file = node_imgs_dir / f"node_{subtree_root.index}_vis.jpg"
    img.save(img_file)

    pydot_node = pydot.Node(
        _node_name(subtree_root),
        image=f'"{img_file}"',
        xlabel=f'"{subtree_root.index}"',
        fontsize=6,
        labelfontcolor="gray50",
        fontname=FONT,
        shape="box",
    )
    l_descendants = _pydot_nodes(
        subtree_root.left, patches_dir, node_imgs_dir, class_names
    )
    r_descendants = _pydot_nodes(
        subtree_root.right, patches_dir, node_imgs_dir, class_names
    )
    return [pydot_node] + l_descendants + r_descendants


def _pydot_nodes_leaf(
    leaf: Leaf,
    class_names: tuple,
) -> list[pydot.Node]:
    pydot_node = gen_leaf(leaf, class_names)
    return [pydot_node]


def _pydot_edges(subtree_root: Node) -> list[pydot.Edge]:
    match subtree_root:
        case InternalNode() as internal_node:
            return _pydot_edges_internal(internal_node)
        case Leaf():
            return []
        case other:
            raise ValueError(f"Unrecognized node {other}.")


def _pydot_edges_internal(subtree_root: InternalNode) -> list[pydot.Edge]:
    l_descendants = _pydot_edges(subtree_root.left)
    r_descendants = _pydot_edges(subtree_root.right)
    l_edge = pydot.Edge(
        _node_name(subtree_root),
        _node_name(subtree_root.left),
        label="Absent",
        **EDGE_ATTRS,
    )
    r_edge = pydot.Edge(
        _node_name(subtree_root),
        _node_name(subtree_root.right),
        label="Present",
        **EDGE_ATTRS,
    )
    return [l_edge, r_edge] + l_descendants + r_descendants


def _gen_internal_node_img(node: InternalNode, patches_dir: os.PathLike) -> Image:
    internal_node_id = node.index
    # TODO: move hardcoded strings to config
    patch_path = os.path.join(patches_dir, f"{internal_node_id}_closest_patch.png")
    bb_path = os.path.join(
        patches_dir, f"{internal_node_id}_bounding_box_closest_patch.png"
    )
    patch_img_orig = Image.open(patch_path)
    bb_img_orig = Image.open(bb_path)

    bb_img = ImageOps.contain(bb_img_orig, SINGLE_NODE_IMG_SIZE)
    patch_img = ImageOps.contain(patch_img_orig, SINGLE_NODE_IMG_SIZE)
    wbb, hbb = bb_img.size
    w, h = patch_img.size

    together_w, together_h = w + INTERNAL_NODE_IMG_GAP + wbb, max(h, hbb)
    together = Image.new(
        patch_img.mode, (together_w, together_h), color=(255, 255, 255)
    )
    together.paste(patch_img, (0, 0))
    together.paste(bb_img, (w + INTERNAL_NODE_IMG_GAP, 0))
    return together.convert("RGB")
