import logging
import os

import pydot
import torch
from PIL import Image, ImageOps, ImageDraw
import pandas as pd
import matplotlib.pyplot as plt

from src.core.models import ProtoTree
from src.core.node import InternalNode, Leaf, Node
from src.visualize.create.dot import (
    FONT,
    _node_name,
    gen_leaf,
    graph_with_components,
)

log = logging.getLogger(__name__)

EDGE_ATTRS = dict(fontsize=10, tailport="s", headport="n", fontname=FONT)
SINGLE_NODE_IMG_SIZE = (100, 100)
INTERNAL_NODE_IMG_GAP = 4


@torch.no_grad()
def save_tree_visualization(
    model: ProtoTree,
    prototypes_info: dict, #patches_dir: os.PathLike,
    tree_dir: os.PathLike,
    class_names: tuple,
    global_scores: pd.DataFrame = None,
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
        model.tree_section.node_to_proto_idx,
        prototypes_info,
        global_scores,
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
    root: Node,
    node_to_proto_idx: dict[Node, int],
    prototypes_info: dict[str, dict],  
    global_scores: pd.DataFrame,
    node_imgs_dir: os.PathLike,
    class_names: tuple,
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

    pydot_nodes = _pydot_nodes(
        root, node_to_proto_idx, prototypes_info, global_scores, node_imgs_dir, class_names
    )
    pydot_edges = _pydot_edges(root)
    return graph_with_components(pydot_tree, pydot_nodes, [], pydot_edges)


def _pydot_nodes(
    subtree_root: Node,
    node_to_proto_idx: dict[Node, int],
    prototypes_info: dict[str, dict],
    global_scores: pd.DataFrame,
    node_imgs_dir: os.PathLike,
    class_names: tuple,
) -> list[pydot.Node]:
    # maybe leave it here in the code?
    
    # TODO: How do we get a Julia-style by-default error when we can't dispatch? Or even better, some sort of typing
    #  error like in Rust, Scala etc? (same for everywhere else we match on node type) Specifying a union/enum manually
    #  and then using NoReturn seems to defeat the point of abstracting into a superclass.
    match subtree_root:
        case InternalNode() as internal_node:
            return _pydot_nodes_internal(
                internal_node,
                node_to_proto_idx,
                prototypes_info,
                global_scores,
                node_imgs_dir,
                class_names,
            )
        case Leaf() as leaf:
            return _pydot_nodes_leaf(leaf, class_names)
        case other:
            raise ValueError(f"Unrecognized node {other}.")


def _pydot_nodes_internal(
    subtree_root: InternalNode,
    node_to_proto_idx: dict[Node, int],
    prototypes_info: dict[str, dict],  
    global_scores: pd.DataFrame,
    node_imgs_dir: os.PathLike,
    class_names: tuple,
) -> list[pydot.Node]:
    subtree_root_proto_idx = node_to_proto_idx[subtree_root]
    img = _gen_internal_node_img(subtree_root_proto_idx, prototypes_info, global_scores)
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
        subtree_root.left, node_to_proto_idx, prototypes_info, global_scores, node_imgs_dir, class_names
    )
    r_descendants = _pydot_nodes(
        subtree_root.right, node_to_proto_idx, prototypes_info, global_scores, node_imgs_dir, class_names
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


# def _gen_internal_node_img(proto_idx: int, patches_dir: os.PathLike) -> Image:
#     patch_path = os.path.join(patches_dir, f"{proto_idx}_closest_patch.png")
#     bb_path = os.path.join(patches_dir, f"{proto_idx}_bounding_box_closest_patch.png")
#     patch_img_orig = Image.open(patch_path)
#     bb_img_orig = Image.open(bb_path)

#     bb_img = ImageOps.contain(bb_img_orig, SINGLE_NODE_IMG_SIZE)
#     patch_img = ImageOps.contain(patch_img_orig, SINGLE_NODE_IMG_SIZE)
#     wbb, hbb = bb_img.size
#     w, h = patch_img.size

#     together_w, together_h = w + INTERNAL_NODE_IMG_GAP + wbb, max(h, hbb)
#     together = Image.new(
#         patch_img.mode, (together_w, together_h), color=(255, 255, 255)
#     )
#     together.paste(patch_img, (0, 0))
#     together.paste(bb_img, (w + INTERNAL_NODE_IMG_GAP, 0))
#     return together.convert("RGB")

def _gen_internal_node_img(proto_idx: int, prototypes_info: dict, global_scores_info: pd.DataFrame) -> Image:
    
    proto_info = prototypes_info[str(proto_idx)]
    
    img_orig = Image.open(proto_info["path"])
    bb = proto_info["bbox"]
    patch_img_orig = img_orig.crop(bb) # TODOs gio check PIL bbox as needs to be done x,y,w,h, or x1,y1,x2,y2img_orig[bb[1]:bb[3], bb[0]:bb[2]]
    
    draw = ImageDraw.Draw(img_orig)
    draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], outline="yellow", width=5)
    
    bb_img = ImageOps.contain(img_orig, SINGLE_NODE_IMG_SIZE)
    patch_img = ImageOps.contain(patch_img_orig, SINGLE_NODE_IMG_SIZE)
    
    wbb, hbb = bb_img.size
    w, h = patch_img.size
    
    # Get global score image  
    if global_scores_info is not None:
        proto_scores = global_scores_info.loc[global_scores_info["prototype"] == proto_idx].set_index("prototype")
        proto_scores = round(proto_scores, 4)
        ax = proto_scores.plot.bar()
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        ax.yaxis.set_visible(False)
        ax.set_facecolor('0.9')
        plt.savefig('tmp.png')
        img_scores_orig = Image.open("tmp.png")
        os.remove("tmp.png")
    
        img_scores = ImageOps.contain(img_scores_orig, SINGLE_NODE_IMG_SIZE)
        wscore, hscore = img_scores.size

    if global_scores_info is None:
        together_w, together_h = w + INTERNAL_NODE_IMG_GAP + wbb + INTERNAL_NODE_IMG_GAP, max(h, hbb)
    else:
        together_w, together_h = w + INTERNAL_NODE_IMG_GAP + wbb + INTERNAL_NODE_IMG_GAP + wscore, max(h, hbb, hscore)
    
    together = Image.new(
        patch_img.mode, (together_w, together_h), color=(255, 255, 255)
    )
    together.paste(patch_img, (0, 0))
    together.paste(bb_img, (w + INTERNAL_NODE_IMG_GAP, 0))
    
    if global_scores_info is not None:
        together.paste(img_scores, (w + INTERNAL_NODE_IMG_GAP + wbb + INTERNAL_NODE_IMG_GAP, 0))
    return together.convert("RGB")