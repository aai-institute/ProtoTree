import logging
from dataclasses import dataclass
from math import isnan
from statistics import mean
from typing import List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from pydantic import validator, dataclasses
from torch import Tensor
from torch.nn import functional as F

from prototree.base import ProtoBase
from prototree.img_similarity import img_proto_similarity, ImageProtoSimilarity
from prototree.node import InternalNode, Leaf, Node, NodeProbabilities, create_tree
from prototree.prune import prune_unconfident_leaves
from prototree.optim import (
    NonlinearSchedulerParams,
    get_nonlinear_scheduler,
    freezable_step,
)
from prototree.types import SamplingStrat, SingleLeafStrat
from util.indexing import select_not
from util.net import NAME_TO_NET

log = logging.getLogger(__name__)


MATCH_UPDATE_PERIOD = 125


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


@dataclass
class NodeSimilarity:
    similarity: ImageProtoSimilarity
    node: InternalNode


@dataclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class LeafRationalization:
    ancestor_sims: list[NodeSimilarity]
    leaf: Leaf

    @validator("ancestor_sims")
    def validate_ancestor_sims_nonempty(cls, v):
        assert v, "ancestor_sims must not be empty"

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
        return [ancestor_child.is_right_child for ancestor_child in ancestor_children]


class ProtoTree(pl.LightningModule):
    # TODO: We could abstract this and ProtoPNet into a superclass. However, perhaps we should wait for the rule of 3 to
    #  help us choose the right abstraction.
    def __init__(
        self,
        h_proto: int,
        w_proto: int,
        channels_proto: int,
        num_classes: int,
        depth: int,
        leaf_pruning_threshold: float,
        leaf_opt_ewma_alpha: float,
        project_epochs: set[int],
        nonlinear_scheduler_params: NonlinearSchedulerParams,
        gradient_leaf_opt: bool = False,
        backbone_name="resnet50_inat",
        pretrained=True,
    ):
        """
        :param h_proto: height of prototype
        :param w_proto: width of prototype
        :param channels_proto: number of input channels for the prototypes,
            coincides with the output channels of the net+add_on layers, prior to prototype layers.
        :param num_classes:
        :param depth: depth of tree, will result in 2^depth leaves and 2^depth-1 internal nodes
        :param backbone_name: name of backbone, e.g. resnet18
        :param pretrained:
        """
        super().__init__()

        # TODO: Use dependency injection here?
        backbone = NAME_TO_NET[backbone_name](pretrained=pretrained)
        num_prototypes = 2**depth - 1
        self.proto_base = ProtoBase(
            num_prototypes=num_prototypes,
            prototype_shape=(channels_proto, w_proto, h_proto),
            backbone=backbone,
        )
        self.tree_section = TreeSection(
            num_classes=num_classes,
            depth=depth,
            leaf_pruning_threshold=leaf_pruning_threshold,
            leaf_opt_ewma_alpha=leaf_opt_ewma_alpha,
            gradient_leaf_opt=gradient_leaf_opt,
        )
        self.gradient_leaf_opt = gradient_leaf_opt

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

        logits, node_to_prob, predicting_leaves = self.forward(x)
        loss = F.nll_loss(logits, y)
        if isnan(loss.item()):
            raise ValueError("Loss is NaN, cannot proceed any further.")
        nonlinear_optim.zero_grad()
        self.manual_backward(loss)
        nonlinear_optim.step()

        if not self.gradient_leaf_opt:
            self.tree_section.deriv_free_leaves_update(y, logits.detach(), node_to_prob)

        # TODO: Hack because update_proto_patch_matches is inefficient.
        if batch_idx % MATCH_UPDATE_PERIOD == MATCH_UPDATE_PERIOD - 1:
            # It's useful to compute this for visualizations, even if we're not projecting.
            self.proto_base.update_proto_patch_matches(self.proto_patch_matches, x, y)

        y_pred = logits.argmax(dim=1)
        acc = (y_pred == y).sum().item() / len(y)
        self.train_step_outputs.append((acc, loss.item()))
        self.log("Train acc", acc, prog_bar=True)
        self.log("Train loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.predict(x, strategy="distributed")
        acc = (y_pred == y).sum().item() / len(y)
        self.val_step_outputs.append(acc)
        self.log("Val acc", acc, prog_bar=True)

    def on_train_epoch_end(self):
        if self.trainer.current_epoch in self.project_epochs:
            self.proto_base.project_prototypes(self.proto_patch_matches)

        avg_acc = mean([item[0] for item in self.train_step_outputs])
        avg_loss = mean([item[1] for item in self.train_step_outputs])
        self.log("Train avg acc", avg_acc, prog_bar=True)
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
        if self.gradient_leaf_opt:
            leaf_dists = [leaf.dist_params for leaf in self.tree_section.root.leaves]
            optimizer.param_groups.append({"params": leaf_dists} | optimizer.defaults)
        return [optimizer], [scheduler]

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

        similarities = self.proto_base.forward(x)
        node_to_probs = self.tree_section.get_node_to_probs(similarities)

        # TODO: Find a better approach for this branching logic (https://martinfowler.com/bliki/FlagArgument.html).
        match self.training, strategy:
            case _, "distributed":
                predicting_leaves = None
                logits = self.tree_section.root.forward(node_to_probs)
            case False, "sample_max" | "greedy":
                predicting_leaves = TreeSection.get_predicting_leaves(
                    self.tree_section.root, node_to_probs, strategy
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
        patches, dists = self.proto_base.patches_and_dists(x)

        rationalizations = []
        for x_i, predicting_leaf, dists_i, patches_i in zip(
            x, predicting_leaves, dists, patches
        ):
            leaf_ancestors = predicting_leaf.ancestors
            ancestor_sims: list[NodeSimilarity] = []
            for leaf_ancestor in leaf_ancestors:
                node_proto_idx = self.tree_section.node_to_proto_idx[leaf_ancestor]

                node_distances = dists_i[node_proto_idx, :, :]
                similarity = img_proto_similarity(
                    node_proto_idx, x_i, node_distances, patches_i
                )
                ancestor_sims.append(NodeSimilarity(similarity, leaf_ancestor))

            rationalization = LeafRationalization(
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
        logits = self.predict_logits(x, strategy=strategy)
        return logits.argmax(dim=1)

    def predict_logits(
        self,
        x: torch.Tensor,
        strategy: SamplingStrat = "sample_max",
    ) -> torch.Tensor:
        return self.forward(x, strategy=strategy)[0]

    def predict_probs(
        self,
        x: torch.Tensor,
        strategy: SamplingStrat = "sample_max",
    ) -> torch.Tensor:
        logits = self.predict_logits(x, strategy=strategy)
        return logits.softmax(dim=1)

    def prune(self, leaf_pruning_threshold: float):
        self.tree_section.prune(leaf_pruning_threshold)

    def print(self):
        # TODO: Make this return a string?
        self.tree_section.root.print_tree()

    def log_state(self):
        self.tree_section.log_leaves_properties()


class TreeSection(nn.Module):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        leaf_pruning_threshold: float,
        leaf_opt_ewma_alpha: float,
        gradient_leaf_opt: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.root = create_tree(depth, num_classes, gradient_leaf_opt=gradient_leaf_opt)
        self.node_to_proto_idx: dict[Node, int] = {
            node: idx for idx, node in enumerate(self.root.descendant_internal_nodes)
        }
        self.leaf_pruning_threshold = leaf_pruning_threshold
        self.leaf_opt_ewma_alpha = leaf_opt_ewma_alpha

        # Ensures proper device handling
        self._dist_param_registration = nn.ParameterList(
            leaf.dist_params for leaf in self.root.leaves
        )

    def get_node_to_log_p_right(
        self, similarities: torch.Tensor
    ) -> dict[Node, torch.Tensor]:
        return {node: -similarities[:, i] for node, i in self.node_to_proto_idx.items()}

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
            log_p_arrival=torch.zeros_like(root_log_p_right),
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

    def get_node_to_probs(
        self, similarities: torch.Tensor
    ) -> dict[Node, NodeProbabilities]:
        """
        Computes the log probabilities (left, right, arrival) for all nodes for the input x.

        :param similarities:
        :return: dictionary mapping each node to a dataclass containing tensors of shape (batch_size,)
        """
        node_to_log_p_right = self.get_node_to_log_p_right(similarities)
        return self._node_to_probs_from_p_right(node_to_log_p_right, self.root)

    @property
    def all_nodes(self) -> set:
        return self.root.descendants

    @property
    def internal_nodes(self):
        return self.root.descendant_internal_nodes

    @property
    def leaves(self):
        return self.root.leaves

    @property
    def num_internal_nodes(self):
        return self.root.num_internal_nodes

    @property
    def num_leaves(self):
        return self.root.num_leaves

    @staticmethod
    def get_predicting_leaves(
        root: InternalNode,
        node_to_probs: dict[Node, NodeProbabilities],
        sampling_strategy: SingleLeafStrat,
    ) -> List[Leaf]:
        """
        Selects one leaf for each entry of the batch covered in node_to_probs.
        """
        match sampling_strategy:
            case "sample_max":
                return TreeSection._get_max_p_arrival_leaves(root.leaves, node_to_probs)
            case "greedy":
                return TreeSection._get_predicting_leaves_greedily(root, node_to_probs)
            case other:
                raise ValueError(f"Unknown sampling strategy {other}")

    @staticmethod
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

    @staticmethod
    def _get_max_p_arrival_leaves(
        leaves: List[Leaf], node_to_probs: dict[Node, NodeProbabilities]
    ) -> List[Leaf]:
        """
        Selects one leaf for each entry of the batch covered in node_to_probs.

        :param leaves:
        :param node_to_probs: see `ProtoTree.get_node_to_probs`
        :return: list of leaves of length `node_to_probs.batch_size`
        """
        log_p_arrivals = [
            node_to_probs[leaf].log_p_arrival.unsqueeze(1) for leaf in leaves
        ]
        log_p_arrivals = torch.cat(log_p_arrivals, dim=1)  # shape: (bs, n_leaves)
        predicting_leaf_idx = torch.argmax(log_p_arrivals, dim=1).long()  # shape: (bs,)
        return [leaves[i.item()] for i in predicting_leaf_idx]

    @torch.no_grad()
    def deriv_free_leaves_update(
        self,
        y_true: torch.Tensor,
        logits: torch.Tensor,
        node_to_prob: dict[Node, NodeProbabilities],
    ):
        """
        :param y_true: shape (batch_size)
        :param logits: shape (batch_size, num_classes)
        :param node_to_prob:
        """
        batch_size, num_classes = logits.shape

        y_true_one_hot = F.one_hot(y_true, num_classes=num_classes)
        y_true_logits = torch.log(y_true_one_hot)

        for leaf in self.leaves:
            self._deriv_free_leaf_update(leaf, node_to_prob, logits, y_true_logits)

    def _deriv_free_leaf_update(
        self,
        leaf: Leaf,
        node_to_prob: dict[Node, NodeProbabilities],
        logits: torch.Tensor,
        y_true_logits: torch.Tensor,
    ):
        """
        :param leaf:
        :param node_to_prob:
        :param logits: of shape (batch_size, num_classes)
        :param y_true_logits: of shape (batch_size, num_classes)
        """
        # shape (batch_size, 1)
        log_p_arrival = node_to_prob[leaf].log_p_arrival.unsqueeze(1)
        # shape (num_classes). Not the same as logits, which has (batch_size, num_classes)
        leaf_logits = leaf.y_logits()

        # TODO: y_true_logits is mostly -Inf terms (the rest being 0s) that won't contribute to the total, and we are
        #  also summing together tensors of different shapes. We should be able to express this more clearly and
        #  efficiently by taking advantage of this sparsity.
        log_dist_update = torch.logsumexp(
            log_p_arrival + leaf_logits + y_true_logits - logits,
            dim=0,
        )

        dist_update = torch.exp(log_dist_update)

        # This exponentially weighted moving average is designed to ensure stability of the leaf class probability
        # distributions (leaf.dist_params), by lowpass filtering out noise from minibatching in the optimization.
        # TODO: Work out how best to initialize the EWMA to avoid a long "burn-in".
        leaf.dist_param_update_count += 1
        count_alpha = (
            1 / leaf.dist_param_update_count
        )  # Stops the first updates having too large an impact.
        alpha = max(count_alpha, self.leaf_opt_ewma_alpha)
        leaf.dist_params.mul_(1.0 - alpha)
        leaf.dist_params.add_(dist_update)

    def log_leaves_properties(self):
        """
        Logs information about which leaves have a sufficiently high confidence and whether there
        are classes not predicted by any leaf. Useful for debugging the training process.
        """
        n_leaves_above_threshold = 0
        classes_covered = set()
        for leaf in self.leaves:
            classes_covered.add(leaf.predicted_label())
            if leaf.conf_predicted_label() > self.leaf_pruning_threshold:
                n_leaves_above_threshold += 1

        log.info(
            f"Leaves with confidence > {self.leaf_pruning_threshold:.3f}: {n_leaves_above_threshold}"
        )

        num_classes = self.leaves[0].num_classes
        class_labels_without_leaf = set(range(num_classes)) - classes_covered
        if class_labels_without_leaf:
            log.info(f"Never predicted classes: {class_labels_without_leaf}")

    def prune(self, leaf_pruning_threshold: float):
        log.info(
            f"Before pruning: {self.root.num_internal_nodes} internal_nodes and {self.root.num_leaves} leaves"
        )
        num_nodes_before = len(self.root.descendant_internal_nodes)

        # all work happens here, the rest is just logging
        prune_unconfident_leaves(self.root, leaf_pruning_threshold)

        frac_nodes_pruned = (
            1 - len(self.root.descendant_internal_nodes) / num_nodes_before
        )
        log.info(
            f"After pruning: {self.root.num_internal_nodes} internal_nodes and {self.root.num_leaves} leaves"
        )
        log.info(f"Fraction of nodes pruned: {frac_nodes_pruned}")
