import pickle
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from prototree.node import InternalNode, Leaf, Node, create_tree
from util.func import min_pool2d
from util.l2conv import L2Conv2D


# TODO: inherit from Node?
class ProtoTree(nn.Module):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        prototype_channels: int,
        w_prototype: int,
        h_prototype: int,
        feature_net: torch.nn.Module,
        log_probabilities: bool = True,
        kontschieder_normalization: bool = False,
        add_on_layers: nn.Module = None,
        disable_derivative_free_leaf_optim=False,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.root = create_tree(
            depth,
            num_classes,
            log_probabilities=log_probabilities,
            kontschieder_normalization=kontschieder_normalization,
            disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
        )

        # this thing contains one prototype per node. The nodes themselves actually don't contain any model parameters
        # During the forward of the nodes, the outputs of this layer are passed as a somehow obscure datastracture
        # from which each node reads off information...
        # TODO: fix this messy pattern or at least make the data flow very explicit. Do nodes need to be modules?
        #  They have no parameters to fit!
        self.prototype_layer = L2Conv2D(
            self.num_internal_nodes, prototype_channels, w_prototype, h_prototype
        )
        # TODO: why is this not part of the prototype layer?
        self.init_prototype_layer()

        # Keep a dict that stores a reference to each node's parent
        # Key: node -> Value: the node's parent
        # The root of the tree is mapped to None
        # TODO: move this to the tree
        self.node2parent: dict[Node, Optional[InternalNode]] = {}
        self._set_parents()

        self.net = feature_net
        self.add_on = add_on_layers if add_on_layers is not None else nn.Identity()

        # Flag that indicates whether probabilities or log probabilities are computed
        self.log_probabilities = log_probabilities

        # Map each decision node to an output of the feature net
        # TODO: that's not really true, and why do we need this at all? Can't we use node id?
        #   This is used for retrieving outputs from some indexed list...
        self.out_map: dict[Node, int] = {
            n: i for i, n in zip(range(2**depth - 1), self.internal_nodes)
        }

    @property
    def num_prototypes(self):
        return self.prototype_layer.num_prototypes

    @property
    def prototype_channels(self):
        return self.prototype_layer.input_channels

    def init_prototype_layer(self):
        # TODO: wassup with the constants?
        torch.nn.init.normal_(self.prototype_layer.prototype_tensors, mean=0.5, std=0.1)

    @property
    def device(self):
        return next(self.parameters()).device

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

    # TODO: convoluted, multiple returns in the middle of code, lots of duplication. Simplify this!
    def forward(
        self,
        x: torch.Tensor,
        sampling_strategy: Literal[
            "distributed", "sample_max", "greedy"
        ] = "distributed",
    ) -> tuple[torch.Tensor, dict]:
        """

        :param x:
        :param sampling_strategy:
        :return: tensor of predicted prob. distributions/logits of shape (bs, k), info_dict
        """
        distances = self.extract_prototype_distances(x)
        # Compute minimal distance for each prototype to any patch by min pooling over the patches
        # Shape: (batch_size, num_prototypes)
        min_distances = min_pool2d(
            distances, kernel_size=distances.shape[-2:]
        ).squeeze()

        similarities = -min_distances
        if not self.log_probabilities:
            similarities = torch.exp(similarities)

        # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        # TODO: here is something important, tightly coupled with out_map construction
        #   This turns it into a list. Do we need chunk? Can't we just use it directly?
        #   The variable should be renamed to something more meaningful.
        similarities = similarities.chunk(similarities.size(1), dim=1)
        # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
        # the tree. Use a copy, as the original should not be modified

        y_pred_proba, node_attr_dict = self.root.forward(
            x, similarities=similarities, out_map=self.out_map
        )

        # Store the output probabilities of all decision nodes in the tree
        info = dict()
        info["p_arrival"] = {
            n.index: node_attr_dict[n, "p_arrival"] for n in self.all_nodes
        }
        info["p_right"] = {
            n.index: node_attr_dict[n, "p_right"] for n in self.internal_nodes
        }

        # TODO: split this off into separate method. Note: for training only distributed should be used
        #  So this probably generally should be moved.
        # Generate the output based on the chosen sampling strategy
        if sampling_strategy == "distributed":
            return y_pred_proba, info
        if sampling_strategy == "sample_max":  # Sample max
            batch_size = x.size(0)

            # Prepare data for selection of most probable distributions
            leaves = list(self.leaves)
            # All shaped (bs, 1)
            p_arrivals = [
                node_attr_dict[l, "p_arrival"].view(batch_size, 1) for l in leaves
            ]
            p_arrivals = torch.cat(tuple(p_arrivals), dim=1)  # shape: (bs, L)
            # All shaped (bs, 1, k)
            distributions = [
                node_attr_dict[l, "distribution"].view(batch_size, 1, self.num_classes)
                for l in leaves
            ]
            distributions = torch.cat(tuple(distributions), dim=1)  # shape: (bs, L, k)

            # Let L denote the number of leaves in this tree
            # Select indices (in the 'leaves' variable) of leaves with the highest path probability
            ix = torch.argmax(p_arrivals, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with the highest path probability
            dists = []
            for j, i in zip(range(distributions.shape[0]), ix):
                dists += [distributions[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)

            # Store the indices of the leaves with the highest path probability
            info["out_leaf_ix"] = [leaves[i.item()].index for i in ix]

            return dists, info

        if sampling_strategy == "greedy":  # Greedy
            # At every decision node, the child with the highest probability will be chosen
            batch_size = x.size(0)
            # Set the threshold for when either child is more likely
            threshold = 0.5 if not self.log_probabilities else np.log(0.5)
            # Keep track of the routes taken for each of the items in the batch
            routing = [[] for _ in range(batch_size)]
            # Traverse the tree for all items
            # Keep track of all nodes encountered
            for i in range(batch_size):
                node = self.root
                while node in self.internal_nodes:
                    routing[i] += [node]
                    if node_attr_dict[node, "p_right"][i].item() > threshold:
                        node = node.right
                    else:
                        node = node.left
                routing[i] += [node]

            # Obtain output distributions of each leaf
            # Each selected leaf is at the end of a path stored in the `routing` variable
            dists = [node_attr_dict[path[-1], "distribution"][0] for path in routing]
            # Concatenate the dists in a new batch dimension
            dists = torch.cat([dist.unsqueeze(0) for dist in dists], dim=0).to(
                device=x.device
            )

            # Store info
            info["out_leaf_ix"] = [path[-1].index for path in routing]

            return dists, info
        raise ValueError("Sampling strategy not recognized!")

    def extract_prototype_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns a tensor of shape (batch_size, num_prototypes, n_patches_w, n_patches_h) obtained
        from computing  the squared L2 distances for patches of the prototype shape from the input using all prototypes.
        Here n_patches_w = W - w + 1 and n_patches_h = H - h + 1 where (w, h) are the width and height of the
        prototypes and (W, H) are the width and height of the output of `extract_features` applied to x.
        There are in total n_patches_w * n_patches_h patches of the prototype shape in the input.
        """

        features = self.extract_features(x)
        distances = self.prototype_layer(features)
        return distances

    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def size(self) -> int:
        return self.root.size

    @property
    def all_nodes(self) -> set:
        return self.root.descendants

    @property
    def node_by_index(self) -> dict:
        return self.root.node_by_index

    # TODO: this shouldn't be a property (among many other properties)
    @property
    def node_depths(self) -> dict[Node, int]:
        node2depth = {}

        def assign_depths(node, cur_depth):
            node2depth[node] = cur_depth
            for child in node.child_nodes:
                assign_depths(child, cur_depth + 1)

        assign_depths(self.root, 0)
        return node2depth

    @property
    def internal_nodes(self) -> set:
        return self.root.descendant_internal_nodes

    @property
    def leaves(self) -> set:
        return self.root.leaves

    @property
    def num_internal_nodes(self) -> int:
        return self.root.num_internal_nodes

    @property
    def num_leaves(self) -> int:
        return self.root.num_leaves

    def save(self, basedir: Union[str, Path], file_name: str = "model.pth"):
        self.eval()
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self, basedir / file_name)

    def save_state(
        self,
        basedir: str,
        state_file_name="model_state.pth",
        pickle_file_name="tree.pkl",
    ):
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basedir / state_file_name)
        # TODO: what's up with this?
        with open(basedir / pickle_file_name, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + "/model.pth")

    # TODO: move whole parent logic to tree (i.e. to node and root node in particular) instead of this class
    def _set_parents(self) -> None:
        self.node2parent.clear()
        self.node2parent[self.root] = None

        def _set_parents_recursively(node: Node):
            if isinstance(node, InternalNode):
                self.node2parent[node.right] = node
                self.node2parent[node.left] = node
                _set_parents_recursively(node.right)
                _set_parents_recursively(node.left)
                return
            if isinstance(node, Leaf):
                return  # Nothing to do here!

        _set_parents_recursively(self.root)

    # TODO: make a tree abstraction, move it there
    def path_to(self, node: Node):
        assert node in self.leaves or node in self.internal_nodes
        path = [node]
        while isinstance(self.node2parent[node], Node):
            node = self.node2parent[node]
            path = [node] + path
        return path
