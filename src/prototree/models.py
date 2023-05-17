import pickle
from math import isnan
from os import PathLike
from pathlib import Path
from statistics import mean
from typing import List, Literal, Optional, Union, Any

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from pydantic import dataclasses, root_validator
from torch import Tensor
from torch.nn import functional as F

from prototree.base import ProtoBase
from prototree.img_similarity import img_proto_similarity, ImageProtoSimilarity
from prototree.node import InternalNode, Leaf, Node, NodeProbabilities, create_tree
from prototree.optim import (
    get_nonlinear_scheduler,
    NonlinearSchedulerParams,
    freezable_step,
)
from prototree.types import SamplingStrat, SingleLeafStrat
from util.indexing import select_not
from util.l2conv import L2Conv2D
from util.net import default_add_on_layers, NAME_TO_NET

MATCH_UPDATE_PERIOD = 125


class ProtoModel(nn.Module):
    # Temporarily preserved class for the old inheritance based ProtoTree.

    # TODO: "Composition over Inheritance" probably applies here for the backbone and prototypes. As added motivation,
    #  it looks like the way this is built right now violates the Liskov substitution principle (e.g. Mypy picks up
    #  incompatible signatures for methods like `forward`).

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
        # TODO: The parameters in this hardcoded initialization seem to (very, very) roughly correspond to a random
        #  average-looking latent patch (which is a good start point for the prototypes), but it would be nice if we had
        #  a more principled way of choosing the initialization.
        # NOTE: The paper means std=0.1 when it says N(0.5, 0.1), not var=0.1.
        torch.nn.init.normal_(self.prototype_layer.prototype_tensors, mean=0.5, std=0.1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feature net and add_on layers to the input. The output
        has the shape (batch_size, num_channels, height, width), where num_channels is the
        number of channels of the prototypes.
        """
        x = self.net(x)
        x = self.add_on(x)
        return x

    def patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the patches for a given input tensor. This is the same as extract_features, except the output is reshaped to
        be (batch_size, d, n_patches_w, n_patches_h, w_proto, h_proto).
        """
        w_proto, h_proto = self.prototype_shape[:2]
        features = self.extract_features(x)
        return features.unfold(2, w_proto, 1).unfold(3, h_proto, 1)

    def distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the minimal distances between the prototypes and the input.
        The output has the shape (batch_size, num_prototypes, n_patches_w, n_patches_h)
        """
        x = self.extract_features(x)
        return self.prototype_layer(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the minimal distances between the prototypes and the input.
        The input has the shape (batch_size, num_channels, H, W).
        The output has the shape (batch_size, num_prototypes).
        """
        x = self.distances(x)
        return torch.amin(x, dim=(2, 3))

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


class ProtoPNet(pl.LightningModule):
    # TODO: We could abstract this and ProtoTree into a superclass. However, perhaps we should wait for the rule of 3 to
    #  help us choose the right abstraction.
    def __init__(
        self,
        h_proto: int,
        w_proto: int,
        channels_proto: int,
        num_classes: int,
        prototypes_per_class: int,
        project_epochs: set[int],
        nonlinear_scheduler_params: NonlinearSchedulerParams,
        backbone_name="resnet50_inat",
        pretrained=True,
        cluster_coeff=0.8,
        sep_coeff=-0.08,
    ):
        super().__init__()
        # TODO: Dependency injection?

        num_prototypes = num_classes * prototypes_per_class
        backbone = NAME_TO_NET[backbone_name](pretrained=pretrained)
        self.proto_base = ProtoBase(
            num_prototypes=num_prototypes,
            prototype_shape=(channels_proto, w_proto, h_proto),
            backbone=backbone,
        )
        self.class_proto_lookup = torch.reshape(
            torch.arange(0, num_prototypes),
            (num_classes, prototypes_per_class),
        )
        self.cluster_coeff, self.sep_coeff = cluster_coeff, sep_coeff

        # TODO: The paper specifies no bias, why?
        self.classifier = nn.Linear(num_prototypes, num_classes, bias=False)

        self.project_epochs = project_epochs
        self.nonlinear_scheduler_params = nonlinear_scheduler_params
        self.automatic_optimization = False

        self.train_step_outputs, self.val_step_outputs = [], []
        self.proto_patch_matches: dict[int, ImageProtoSimilarity] = {}

    def training_step(self, batch, batch_idx):
        nonlinear_optim = self.optimizers()
        nonlinear_scheduler = self.lr_schedulers()

        x, y = batch

        if batch_idx == 0:
            freezable_step(
                nonlinear_scheduler,
                self.trainer.current_epoch,
                nonlinear_optim.params_to_freeze,
            )

        all_dists = self.proto_base.forward(x)
        unnormed_logits = self.classifier(all_dists)
        logits = F.log_softmax(unnormed_logits, dim=1)
        nll_loss = F.nll_loss(logits, y)
        loss = nll_loss + self._proto_costs(all_dists, y)
        if isnan(loss.item()):
            raise ValueError("Loss is NaN, cannot proceed any further.")

        nonlinear_optim.zero_grad()
        self.manual_backward(loss)
        nonlinear_optim.step()

        # TODO: Hack because update_proto_patch_matches is inefficient.
        if batch_idx % MATCH_UPDATE_PERIOD == MATCH_UPDATE_PERIOD - 1:
            # It's useful to compute this for visualizations, even if we're not projecting.
            self.proto_base.update_proto_patch_matches(self.proto_patch_matches, x, y)

        y_pred = logits.argmax(dim=1)
        acc = (y_pred == y).sum().item() / len(y)
        self.train_step_outputs.append((acc, nll_loss.item(), loss.item()))
        self.log("Train acc", acc, prog_bar=True)
        self.log("Train NLL loss", nll_loss, prog_bar=True)
        self.log("Train loss", loss, prog_bar=True)

    def _proto_costs(self, all_dists: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.class_proto_lookup = self.class_proto_lookup.to(device=self.device)
        proto_in_class_indices = self.class_proto_lookup[y, :]
        proto_out_class_indices = select_not(self.class_proto_lookup, y)
        min_in_class_dists = torch.gather(all_dists, 1, proto_in_class_indices)
        min_out_class_dists = torch.gather(all_dists, 1, proto_out_class_indices)
        min_in_class_dist = torch.amin(min_in_class_dists, dim=1)
        min_out_class_dist = torch.amin(min_out_class_dists, dim=1)

        cluster_cost, sep_cost = torch.mean(min_in_class_dist), torch.mean(
            min_out_class_dist
        )
        return self.cluster_coeff * cluster_cost + self.sep_coeff * sep_cost

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.predict(x)
        acc = (y_pred == y).sum().item() / len(y)
        self.val_step_outputs.append(acc)
        self.log("Val acc", acc, prog_bar=True)

    def on_train_epoch_end(self):
        if self.trainer.current_epoch in self.project_epochs:
            self.proto_base.project_prototypes(self.proto_patch_matches)

        avg_acc = mean([item[0] for item in self.train_step_outputs])
        avg_nll_loss = mean([item[1] for item in self.train_step_outputs])
        avg_loss = mean([item[2] for item in self.train_step_outputs])
        self.log("Train avg acc", avg_acc, prog_bar=True)
        self.log("Train avg NLL loss", avg_nll_loss, prog_bar=True)
        self.log("Train avg loss", avg_loss, prog_bar=True)
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        avg_acc = mean(self.val_step_outputs)
        self.log("Val avg acc", avg_acc, prog_bar=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        [optimizer], [scheduler] = get_nonlinear_scheduler(
            self, self.nonlinear_scheduler_params
        )
        optimizer.param_groups.append(
            {"params": list(self.classifier.parameters())} | optimizer.defaults
        )
        return [optimizer], [scheduler]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proto_base.forward(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def predict_probs(
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
        return torch.argmax(self.predict_probs(x), dim=-1)


class ProtoTree(ProtoModel):
    @dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
    class LeafRationalization:
        @dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
        class NodeSimilarity:
            similarity: ImageProtoSimilarity
            node: InternalNode

        # Note: Can only correspond to all the leaf's ancestors in order starting from the root.
        ancestor_sims: list[NodeSimilarity]
        leaf: Leaf

        @root_validator()  # Ignore PyCharm, this makes the method a classmethod.
        def validate_ancestor_sims(cls, vals: dict[str, Any]):
            ancestor_sims: list[ProtoTree.NodeSimilarity] = vals.get("ancestor_sims")
            leaf: Leaf = vals.get("leaf")

            assert ancestor_sims, "ancestor_sims must not be empty"
            assert [
                sim.node for sim in ancestor_sims
            ] == leaf.ancestors, "sims must be of the leaf ancestors"

            return vals

        def proto_presents(self) -> list[bool]:
            """
            Returns a list of bools the same length as ancestor_sims, where each item indicates whether the
            prototype for that node was present. Equivalently, the booleans indicate whether the next node on the way to
            the leaf is a right child.
            """
            non_root_ancestors: list[InternalNode] = [
                sim.node for sim in self.ancestor_sims
            ][1:]
            ancestor_children: list[Node] = non_root_ancestors + [self.leaf]
            return [
                ancestor_child.is_right_child for ancestor_child in ancestor_children
            ]

    def __init__(
        self,
        num_classes: int,
        depth: int,
        channels_proto: int,
        h_proto: int,
        w_proto: int,
        feature_net: torch.nn.Module,
        add_on_layers: Optional[Union[nn.Module, Literal["default"]]] = "default",
    ):
        # the number of internal nodes
        num_prototypes = 2**depth - 1
        super().__init__(
            num_prototypes=num_prototypes,
            prototype_shape=(channels_proto, w_proto, h_proto),
            feature_net=feature_net,
            add_on_layers=add_on_layers,
        )
        self.num_classes = num_classes
        self.tree_root = create_tree(depth, num_classes)
        self.node_to_proto_idx = {
            node: idx
            for idx, node in enumerate(self.tree_root.descendant_internal_nodes)
        }

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for leaf in self.tree_root.leaves:
            leaf.to(*args, **kwargs)
        return self

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
        strategy: SamplingStrat = "distributed",
    ) -> tuple[torch.Tensor, dict[Node, NodeProbabilities], Optional[List[Leaf]]]:
        """
        Produces predictions for input images.

        If sampling_strategy is `distributed`, all leaves contribute to each prediction, and predicting_leaves is None.
        For other sampling strategies, only one leaf is used per sample, which results in an interpretable prediction;
        in this case, predicting_leaves is a list of leaves of length `batch_size`.

        :param x: tensor of shape (batch_size, n_channels, w, h)
        :param strategy:

        :return: Tuple[tensor of predicted logits of shape (bs, k), node_probabilities, and optionally
         predicting_leaves if a single leaf sampling strategy is used]
        """

        node_to_probs = self.get_node_to_probs(x)

        # TODO: Find a better approach for this branching logic (https://martinfowler.com/bliki/FlagArgument.html).
        match self.training, strategy:
            case _, "distributed":
                predicting_leaves = None
                logits = self.tree_root.forward(node_to_probs)
            case False, "sample_max" | "greedy":
                predicting_leaves = get_predicting_leaves(
                    self.tree_root, node_to_probs, strategy
                )
                logits = [leaf.y_logits().unsqueeze(0) for leaf in predicting_leaves]
                logits = torch.cat(logits, dim=0)
            case _:
                raise ValueError(
                    f"Invalid train/test and sampling strategy combination: {self.training=}, {strategy=}"
                )

        return logits, node_to_probs, predicting_leaves

    def explain(
        self,
        x: torch.Tensor,
        strategy: SingleLeafStrat = "sample_max",
    ) -> tuple[
        Tensor,
        dict[Node, NodeProbabilities],
        list[Leaf],
        list[LeafRationalization],
    ]:
        # TODO: This public method works by calling two other methods on the same class. This is perhaps a little bit
        #  unusual and/or clunky, and could be making testing harder. However, it's not currently clear to me if this is
        #  a serious problem, or what the right design for this would be.
        """
        Produces predictions for input images, and rationalizations for why the model made those predictions. This is
        done by chaining self.forward and self.rationalize. See those methods for details regarding the predictions and
        rationalizations.

        :param x: tensor of shape (batch_size, n_channels, w, h)
        :param strategy: This has to be a single leaf sampling strategy.

        :return: Tuple[predicted logits of shape (bs, k), node_probabilities, predicting_leaves, leaf_explanations].
         Since this method is only ever called with a single leaf sampling strategy, both predicting_leaves and
         leaf_explanations will always be not None if this method succeeds.
        """
        logits, node_to_probs, predicting_leaves = self.forward(x, strategy=strategy)
        leaf_explanations = self.rationalize(x, predicting_leaves)
        return logits, node_to_probs, predicting_leaves, leaf_explanations

    @torch.no_grad()
    def rationalize(
        self, x: torch.Tensor, predicting_leaves: list[Leaf]
    ) -> list[LeafRationalization]:
        # TODO: Lots of overlap with img_similarity.patch_match_candidates, so there's potential for extracting out
        #  commonality. However, we also need to beware of premature abstraction.
        """
        Takes in batch_size images and leaves. For each (image, leaf), the model tries to rationalize why that leaf is
        the correct prediction for that image. This rationalization comprises the most similar patch to the image for
        each ancestral node of the leaf (alongside some related information).

        Note that this method accepts arbitrary leaves, there's no requirement that the rationalizations be for leaves
        corresponding to correct labels for the images, or even that the leaves were predicted by the tree. This is
        deliberate, since it can help us assess the interpretability of the model on incorrect and/or random predictions
        and help avoid things like confirmation bias and cherry-picking.

        Args:
            x: Images tensor
            predicting_leaves: List of leaves

        Returns:
            List of rationalizations
        """
        patches, dists = self.patches(x), self.distances(
            x
        )  # Common subexpression elimination possible, if necessary.

        rationalizations = []
        for x_i, predicting_leaf, dists_i, patches_i in zip(
            x, predicting_leaves, dists, patches
        ):
            leaf_ancestors = predicting_leaf.ancestors
            ancestor_sims: list[ImageProtoSimilarity] = []
            for leaf_ancestor in leaf_ancestors:
                node_proto_idx = self.node_to_proto_idx[leaf_ancestor]

                node_distances = dists_i[node_proto_idx, :, :]
                similarity = img_proto_similarity(
                    leaf_ancestor, x_i, node_distances, patches_i
                )
                ancestor_sims.append(similarity)

            rationalization = ProtoTree.LeafRationalization(
                ancestor_sims,
                predicting_leaf,
            )
            rationalizations.append(rationalization)

        return rationalizations

    def predict(
        self,
        x: torch.Tensor,
        strategy: SamplingStrat = "sample_max",
    ) -> torch.Tensor:
        logits = self.forward(x, strategy)[0]
        return logits.argmax(dim=1)

    def predict_probs(
        self,
        x: torch.Tensor,
        strategy: SamplingStrat = "sample_max",
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
    strategy: SingleLeafStrat,
) -> List[Leaf]:
    """
    Selects one leaf for each entry of the batch covered in node_to_probs.
    """
    match strategy:
        case "sample_max":
            return _get_max_p_arrival_leaves(root.leaves, node_to_probs)
        case "greedy":
            return _get_predicting_leaves_greedily(root, node_to_probs)
        case other:
            raise ValueError(f"Unknown sampling strategy {other}")


def _get_predicting_leaves_greedily(
    root: InternalNode, node_to_probs: dict[Node, NodeProbabilities]
) -> List[Leaf]:
    """
    Selects one leaf for each entry of the batch covered in node_to_probs.
    """
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


def _get_max_p_arrival_leaves(
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
