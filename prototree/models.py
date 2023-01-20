import pickle
from os import PathLike
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from prototree.node import InternalNode, Leaf, Node, NodeProbabilities, create_tree
from prototree.types import SamplingStrategy
from util.func import min_pool2d
from util.l2conv import L2Conv2D
from util.net import default_add_on_layers


class PrototypeBase(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        prototype_shape: tuple[int, int, int],
        feature_net: torch.nn.Module,
        add_on_layers: Optional[Union[nn.Module, Literal["default"]]] = "default",
    ):
        """

        :param prototype_shape: shape of the prototypes. (channels, height, width)
        :param feature_net: usually a pretrained network that extracts features from the input images
        :param add_on_layers: used to connect the feature net with the prototypes.
        """
        super().__init__()
        self.prototype_layer = L2Conv2D(
            num_prototypes,
            *prototype_shape,
        )
        self._init_prototype_layer()
        if isinstance(add_on_layers, nn.Module):
            self.add_on = add_on_layers
        elif add_on_layers is None:
            self.add_on = nn.Identity()
        elif add_on_layers == "default":
            self.add_on = default_add_on_layers(feature_net, prototype_shape[0])
        self.net = feature_net

    def _init_prototype_layer(self):
        # TODO: wassup with the constants?
        torch.nn.init.normal_(self.prototype_layer.prototype_tensors, mean=0.5, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the minimal distances between the prototypes and the input.
        The output has the shape (batch_size, num_prototypes)
        """
        x = self.extract_features(x)
        x = self.prototype_layer(x)
        return min_pool2d(x, kernel_size=x.shape[-2:]).squeeze()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def num_prototypes(self):
        return self.prototype_layer.num_prototypes

    @property
    def prototype_channels(self):
        return self.prototype_layer.input_channels

    @property
    def prototype_shape(self):
        return self.prototype_layer.prototype_shape

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feature net and add_on layers to the input.
        """
        x = self.net(x)
        x = self.add_on(x)
        return x

    # TODO: remove all saving things - delegate to downstream code
    def save(self, basedir: PathLike, file_name: str = "model.pth"):
        self.eval()
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self, basedir / file_name)

    def save_state(
        self,
        basedir: PathLike,
        state_file_name="model_state.pth",
        pickle_file_name="tree.pkl",
    ):
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basedir / state_file_name)
        with open(basedir / pickle_file_name, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + "/model.pth")


class ProtoPNet(PrototypeBase):
    def __init__(
        self,
        num_classes: int,
        num_prototypes: int,
        prototype_shape: tuple[int, int, int],
        feature_net: nn.Module,
    ):
        super().__init__(num_prototypes, prototype_shape, feature_net)
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self.classifier(x)

    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the probabilities of the classes for the input.
        The output has the shape (batch_size, num_classes)
        """
        x = self.forward(x)
        return torch.softmax(x, dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.predict_proba(x), dim=-1)


class ProtoTree(PrototypeBase):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        prototype_channels: int,
        w_prototype: int,
        h_prototype: int,
        feature_net: torch.nn.Module,
        add_on_layers: Optional[Union[nn.Module, Literal["default"]]] = "default",
    ):
        # the number of internal nodes
        num_prototypes = 2**depth - 1
        super().__init__(
            num_prototypes=num_prototypes,
            prototype_shape=(prototype_channels, w_prototype, h_prototype),
            feature_net=feature_net,
            add_on_layers=add_on_layers,
        )
        self.num_classes = num_classes
        self.tree_root = create_tree(depth, num_classes)
        self.node_to_proto_idx = {
            node: idx
            for idx, node in enumerate(self.tree_root.descendant_internal_nodes)
        }

    def _get_node_to_log_p_right(self, similarities: torch.Tensor):
        return {node: -similarities[:, i] for node, i in self.node_to_proto_idx.items()}

    def get_node_to_log_p_right(self, x: torch.Tensor) -> dict[Node, torch.Tensor]:
        """
        Extracts the mapping `node->log_p_right` from the prototype similarities.
        """
        similarities = super().forward(x)
        return self._get_node_to_log_p_right(similarities)

    @staticmethod
    def _node_to_probs_from_p_right(
        node_to_log_p_right: dict[Node, torch.Tensor], root: InternalNode
    ):
        """
        This method was separated out from get_node_to_probs to
        highlight that it doesn't directly depend on the input x and could
        in principle be moved to the forward of the tree or somewhere else.

        :param node_to_log_p_right: computed from the similarities, see get_node_to_log_p_right
        :param root: a root of a tree from which node_to_log_p_right was computed
        :return:
        """
        root_log_p_right = node_to_log_p_right[root]

        root_probs = NodeProbabilities(
            log_p_arrival=torch.zeros_like(
                root_log_p_right, device=root_log_p_right.device
            ),
            log_p_right=root_log_p_right,
        )
        result = {root: root_probs}

        def add_strict_descendants(node: Union[InternalNode, Leaf]):
            """
            Adds the log probability of arriving at all strict descendants of node to the result dict.
            """
            if node.is_leaf:
                return
            n_probs = result[node]
            log_p_right_right = node_to_log_p_right.get(node.right)
            log_p_right_left = node_to_log_p_right.get(node.left)

            log_p_arrival_left = n_probs.log_p_arrival_left
            log_p_arrival_right = n_probs.log_p_arrival_right
            result[node.left] = NodeProbabilities(
                log_p_arrival=log_p_arrival_left,
                log_p_right=log_p_right_left,
            )
            result[node.right] = NodeProbabilities(
                log_p_arrival=log_p_arrival_right,
                log_p_right=log_p_right_right,
            )

            add_strict_descendants(node.left)
            add_strict_descendants(node.right)

        add_strict_descendants(root)
        return result

    def get_node_to_probs(self, x: torch.Tensor) -> dict[Node, NodeProbabilities]:
        """
        Computes the log probabilities (left, right, arrival) for all nodes for the input x.

        :param x: input images of shape (batch_size, channels, height, width)
        :return: dictionary mapping each node to a dataclass containing tensors of shape (batch_size,)
        """
        node_to_log_p_right = self.get_node_to_log_p_right(x)
        return self._node_to_probs_from_p_right(node_to_log_p_right, self.tree_root)

    def forward(
        self,
        x: torch.Tensor,
        sampling_strategy: Literal[
            "distributed", "sample_max", "greedy"
        ] = "distributed",
    ) -> tuple[torch.Tensor, dict[Node, NodeProbabilities], Optional[List[Leaf]]]:
        """
        If sampling_strategy is `distributed`, all leaves contribute to each prediction,
        and predicting_leaves is None.
        For other sampling strategies, only one leaf is used per sample, which results in
        an interpretable prediction. Then predicting_leaves is a list of leaves of length
        `batch_size`.

        :param x: tensor of shape (batch_size, n_channels, w, h)
        :param sampling_strategy:

        :return: tensor of predicted logits of shape (bs, k), node_probabilities, predicting_leaves.
        """

        node_to_probs = self.get_node_to_probs(x)

        if sampling_strategy == "distributed":
            predicting_leaves = None
            logits = self.tree_root.forward(node_to_probs)
        else:
            if self.training:
                raise ValueError(
                    f"Only distributed sampling_strategy is supported during training but got: {sampling_strategy=}"
                )
            predicting_leaves = get_predicting_leaves(
                self.tree_root, node_to_probs, sampling_strategy
            )
            logits = [leaf.logits().unsqueeze(0) for leaf in predicting_leaves]
            logits = torch.cat(logits, dim=0)
        return logits, node_to_probs, predicting_leaves

    def predict(
        self,
        x: torch.Tensor,
        sampling_strategy: SamplingStrategy = "sample_max",
    ) -> torch.Tensor:
        logits = self.forward(x, sampling_strategy)[0]
        return logits.argmax(dim=1)

    def predict_proba(
        self,
        x: torch.Tensor,
        strategy: SamplingStrategy = "sample_max",
    ) -> torch.Tensor:
        logits = self.forward(x, strategy)[0]
        return logits.softmax(dim=1)

    @property
    def all_nodes(self) -> set:
        return self.tree_root.descendants

    @property
    def internal_nodes(self):
        return self.tree_root.descendant_internal_nodes

    @property
    def leaves(self):
        return self.tree_root.leaves

    @property
    def num_internal_nodes(self):
        return self.tree_root.num_internal_nodes

    @property
    def num_leaves(self):
        return self.tree_root.num_leaves


def get_predicting_leaves(
    root: InternalNode,
    node_to_probs: dict[Node, NodeProbabilities],
    sampling_strategy: Literal["sample_max", "greedy"],
):
    if sampling_strategy == "sample_max":
        return get_max_p_arrival_leaves(root.leaves, node_to_probs)
    if sampling_strategy == "greedy":
        return get_predicting_leaves_greedily(root, node_to_probs)
    raise ValueError(f"Unknown {sampling_strategy=}")


def get_predicting_leaves_greedily(
    root: InternalNode, node_to_probs: dict[Node, NodeProbabilities]
) -> List[Leaf]:

    neg_log_2 = -np.log(2)

    def get_leaf_for_sample(sample_idx: int):
        # walk through greedily from root to leaf using probabilities for the selected sample
        cur_node = root
        while cur_node.is_internal:
            cur_probs = node_to_probs[cur_node]
            log_p_left = cur_probs.log_p_left[sample_idx].item()
            if log_p_left > neg_log_2:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return cur_node

    batch_size = node_to_probs[root].batch_size
    return [get_leaf_for_sample(i) for i in range(batch_size)]


def get_max_p_arrival_leaves(
    leaves: List[Leaf], node_to_probs: dict[Node, NodeProbabilities]
) -> List[Leaf]:
    """
    Selects one leaf for each entry of the batch covered in node_to_probs.

    :param leaves:
    :param node_to_probs: see `ProtoTree.get_node_to_probs`
    :return: list of leaves of length `node_to_probs.batch_size`
    """
    log_p_arrivals = [node_to_probs[leaf].log_p_arrival.unsqueeze(1) for leaf in leaves]
    log_p_arrivals = torch.cat(log_p_arrivals, dim=1)  # shape: (bs, n_leaves)
    predicting_leaf_idx = torch.argmax(log_p_arrivals, dim=1).long()  # shape: (bs,)
    return [leaves[i.item()] for i in predicting_leaf_idx]
