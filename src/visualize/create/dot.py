import numpy as np
import pydot
import torch

from prototree.node import Leaf, InternalNode, Node

FONT = "Helvetica"  # TODO: Config?


@torch.no_grad()
def gen_leaf(leaf: Leaf, class_names: tuple) -> pydot.Node:
    top_classes_label = _leaf_label(leaf, class_names)

    return pydot.Node(
        _node_name(leaf),
        label=top_classes_label,
        labelfontcolor="gray50",
        fontname=FONT,
        shape="box",
    )


def _leaf_label(leaf: Leaf, class_names: tuple) -> str:
    """
    Generates a label for the leaf, consisting of the predicted classes and associated probability.
    """
    leaf_probs = leaf.y_probs().detach()
    max_prob_inds = np.argmax(leaf_probs, keepdims=True)
    max_prob = leaf_probs[max_prob_inds[0]]
    return f"p = {max_prob:.5f}:\n" + ",\n".join(
        [class_names[i] for i in max_prob_inds]
    )


def _node_name(node: Node) -> str:
    match node:
        case InternalNode():
            return f"internal_{node.index}"
        case Leaf():
            return f"leaf_{node.index}"
        case other:
            raise ValueError(f"Unknown node {other}.")


def graph_with_components(
    graph: pydot.Dot,
    nodes: list[pydot.Node],
    subgraphs: list[pydot.Subgraph],
    edges: list[pydot.Edge],
) -> pydot.Dot:
    for pydot_node in nodes:
        graph.add_node(pydot_node)
    for pydot_subgraph in subgraphs:
        graph.add_subgraph(pydot_subgraph)
    for pydot_edge in edges:
        graph.add_edge(pydot_edge)

    return graph
