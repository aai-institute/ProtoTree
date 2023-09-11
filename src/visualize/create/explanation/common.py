import os
from typing import Callable

import numpy as np
import pydot
import torch

from src.util.data import save_img
from src.visualize.create.dot import FONT


def _original_im_components(
    inv_transform: Callable[[torch.Tensor], np.ndarray],
    transformed_image: torch.Tensor,
    connecting_node_name: str,
    true_class: str,
    decision_flow_dir: os.PathLike,
) -> tuple[list[pydot.Node], list[pydot.Edge]]:
    """
    Produces the nodes and edges for displaying the original image and its class, and connecting it to the next node in
    the graph.

    TODO(Hack): There's currently a separate node that holds the class label (and other text) for the original image,
     so this function produces 2 nodes in total. This is because it's difficult to control the position of the text if
     we use an xlabel on the image node, e.g. the dot engine doesn't respect the xlp attribute. We might want to try
     other engines (e.g. Neato) instead.
    """
    original_im = inv_transform(transformed_image)
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
        connecting_node_name,
        weight=100,
    )
    original_nodes = [original_label_node, original_im_node]
    original_edges = [original_label_edge, original_to_proto_edge]

    return original_nodes, original_edges


def _img_pydot_node(node_name: str, im_file: os.PathLike, size: float) -> pydot.Node:
    """
    Creates a pydot node which resizes the im_file to the specified size.
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
