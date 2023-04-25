from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from prototree.img_similarity import img_proto_similarity, ImageProtoSimilarity
from prototree.node import InternalNode, Leaf, Node, NodeProbabilities, create_tree, log
from prototree.train import NonlinearSchedulerParams, get_nonlinear_scheduler
from prototree.types import SamplingStrat, SingleLeafStrat
from util.l2conv import L2Conv2D
from util.net import default_add_on_layers, NAME_TO_NET


class ProtoBase(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        prototype_shape: tuple[int, int, int],
        backbone: torch.nn.Module,
        add_on_layers: Optional[Union[nn.Module, Literal["default"]]] = "default",
    ):
        """
        :param prototype_shape: shape of the prototypes. (channels, height, width)
        :param backbone: usually a pretrained network that extracts features from the input images
        :param add_on_layers: used to connect the feature net with the prototypes.
        """
        super().__init__()

        # TODO: The parameters in this hardcoded initialization seem to (very, very) roughly correspond to a random
        #  average-looking latent patch (which is a good start point for the prototypes), but it would be nice if we had
        #  a more principled way of choosing the initialization.
        # NOTE: The paper means std=0.1 when it says N(0.5, 0.1), not var=0.1.
        self.prototype_layer = L2Conv2D(
            num_prototypes, *prototype_shape, initial_mean=0.5, initial_std=0.1
        )

        if isinstance(add_on_layers, nn.Module):
            self.add_on = add_on_layers
        elif add_on_layers is None:
            self.add_on = nn.Identity()
        elif add_on_layers == "default":
            self.add_on = default_add_on_layers(backbone, prototype_shape[0])
        self.backbone = backbone

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the feature net and add_on layers to the input. The output
        has the shape (batch_size, num_channels, height, width), where num_channels is the
        number of channels of the prototypes.
        """
        x = self.backbone(x)
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

    @torch.no_grad()
    def apply_xavier(self):
        def _xavier_on_conv(m):
            if type(m) == torch.nn.Conv2d:
                torch.nn.init.xavier_normal_(
                    m.weight, gain=torch.nn.init.calculate_gain("sigmoid")
                )

        self.add_on.apply(_xavier_on_conv)


class ProtoPNet(ProtoBase):
    def __init__(self, num_classes: int, num_prototypes: int, proto_base: ProtoBase):
        super().__init__()
        self.proto_base = proto_base

        # TODO: Use dependency injection for the second half of the model?
        self.classifier = nn.Linear(num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        return self.classifier(x)

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
class LeafRationalization:
    ancestor_similarities: list[ImageProtoSimilarity]
    leaf: Leaf

    @property
    def proto_presents(self) -> list[bool]:
        """
        Returns a list of bools the same length as ancestor_similarities, where each item indicates whether the
        prototype for that node was present. Equivalently, the booleans indicate whether the next node on the way to
        the leaf is a right child.
        """
        ancestor_children = [
            ancestor_similarity.internal_node
            for ancestor_similarity in self.ancestor_similarities[1:]
        ] + [self.leaf]
        return [ancestor_child.is_right_child for ancestor_child in ancestor_children]


class ProtoTree(pl.LightningModule):
    def __init__(
        self,
        h_proto: int,
        w_proto: int,
        channels_proto: int,
        num_classes: int,
        depth: int,
        leaf_pruning_threshold: float,
        leaf_opt_ewma_alpha: float,
        nonlinear_scheduler_params: NonlinearSchedulerParams,
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
        self.proto_base.apply_xavier()
        self.tree_section = TreeSection(
            num_classes=num_classes,
            depth=depth,
            leaf_pruning_threshold=leaf_pruning_threshold,
            leaf_opt_ewma_alpha=leaf_opt_ewma_alpha,
        )

        self.nonlinear_scheduler_params = nonlinear_scheduler_params
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        nonlinear_optim = self.optimizers()
        nonlinear_scheduler = self.lr_schedulers()

        x, y = batch

        if batch_idx == 0:
            current_epoch = self.trainer.current_epoch
            if current_epoch > 0:
                nonlinear_scheduler.step()

            # TODO: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html ?
            if nonlinear_scheduler.freeze_epochs > 0:
                if current_epoch == 0:
                    log.info(f"Freezing network for {nonlinear_scheduler.freeze_epochs} epochs.")
                    for param in nonlinear_optim.params_to_freeze:
                        param.requires_grad = False
                elif current_epoch == nonlinear_scheduler.freeze_epochs + 1:
                    log.info(f"Unfreezing network on epoch {current_epoch}.")
                    for param in nonlinear_optim.params_to_freeze:
                        param.requires_grad = True

        nonlinear_optim.zero_grad()
        logits, node_to_prob, predicting_leaves = self.forward(x)
        # TODO (critical bug): Becomes nan after a while, why has this only happened after refactoring to use PyTorch
        #  Lightning? It was working fine before this.
        loss = F.nll_loss(logits, y)
        loss.backward()
        nonlinear_optim.step()

        self.tree_section.update_leaf_distributions(y, logits.detach(), node_to_prob)

        log.info(f"{loss=}, epoch={self.trainer.current_epoch}, {batch_idx=}")

    def configure_optimizers(self):
        return get_nonlinear_scheduler(self, self.nonlinear_scheduler_params)

    def forward(
        self,
        x: torch.Tensor,
        sampling_strategy: SamplingStrat = "distributed",
    ) -> tuple[torch.Tensor, dict[Node, NodeProbabilities], Optional[List[Leaf]]]:
        """
        Produces predictions for input images.

        If sampling_strategy is `distributed`, all leaves contribute to each prediction, and predicting_leaves is None.
        For other sampling strategies, only one leaf is used per sample, which results in an interpretable prediction;
        in this case, predicting_leaves is a list of leaves of length `batch_size`.

        :param x: tensor of shape (batch_size, n_channels, w, h)
        :param sampling_strategy:

        :return: tensor of predicted logits of shape (bs, k), node_probabilities, predicting_leaves
        """

        similarities = self.proto_base.forward(x)
        node_to_probs = self.tree_section.get_node_to_probs(similarities)

        # TODO: Find a better approach for this branching logic (https://martinfowler.com/bliki/FlagArgument.html).
        match self.training, sampling_strategy:
            case _, "distributed":
                predicting_leaves = None
                logits = self.tree_section.tree_root.forward(node_to_probs)
            case False, "sample_max" | "greedy":
                predicting_leaves = TreeSection.get_predicting_leaves(
                    self.tree_section.tree_root, node_to_probs, sampling_strategy
                )
                logits = [leaf.y_logits().unsqueeze(0) for leaf in predicting_leaves]
                logits = torch.cat(logits, dim=0)
            case _:
                raise ValueError(
                    f"Invalid train/test and sampling strategy combination: {self.training=}, {sampling_strategy=}"
                )

        return logits, node_to_probs, predicting_leaves

    def explain(
        self,
        x: torch.Tensor,
        sampling_strategy: SingleLeafStrat = "sample_max",
    ) -> tuple[
        Tensor,
        dict[Node, NodeProbabilities],
        Optional[list[Leaf]],
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
        :param sampling_strategy:

        :return: predicted logits of shape (bs, k), node_probabilities, predicting_leaves, leaf_explanations
        """
        logits, node_to_probs, predicting_leaves = self.forward(
            x, sampling_strategy=sampling_strategy
        )
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
        patches, dists = self.proto_base.patches(x), self.proto_base.distances(
            x
        )  # Common subexpression elimination possible, if necessary.

        rationalizations = []
        for x_i, predicting_leaf, dists_i, patches_i in zip(
            x, predicting_leaves, dists, patches
        ):
            leaf_ancestors = predicting_leaf.ancestors
            ancestor_similarities: list[ImageProtoSimilarity] = []
            for leaf_ancestor in leaf_ancestors:
                node_proto_idx = self.tree_section.node_to_proto_idx[leaf_ancestor]

                node_distances = dists_i[node_proto_idx, :, :]
                similarity = img_proto_similarity(
                    leaf_ancestor, x_i, node_distances, patches_i
                )
                ancestor_similarities.append(similarity)

            rationalization = LeafRationalization(
                ancestor_similarities,
                predicting_leaf,
            )
            rationalizations.append(rationalization)

        return rationalizations

    def predict(
        self,
        x: torch.Tensor,
        sampling_strategy: SamplingStrat = "sample_max",
    ) -> torch.Tensor:
        logits = self.forward(x, sampling_strategy)[0]
        return logits.argmax(dim=1)

    def predict_probs(
        self,
        x: torch.Tensor,
        strategy: SamplingStrat = "sample_max",
    ) -> torch.Tensor:
        logits = self.forward(x, strategy)[0]
        return logits.softmax(dim=1)

    def log_state(self):
        self.tree_section.log_leaves_properties()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.proto_base.to(*args, **kwargs)
        self.tree_section.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return self.proto_base.device


class TreeSection(nn.Module):
    def __init__(
        self,
        num_classes: int,
        depth: int,
        leaf_pruning_threshold: float,
        leaf_opt_ewma_alpha: float,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.tree_root = create_tree(depth, num_classes)
        self.node_to_proto_idx = {
            node: idx
            for idx, node in enumerate(self.tree_root.descendant_internal_nodes)
        }
        self.leaf_pruning_threshold = leaf_pruning_threshold
        self.leaf_opt_ewma_alpha = leaf_opt_ewma_alpha

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        for leaf in self.tree_root.leaves:
            leaf.to(*args, **kwargs)
        return self

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

    def get_node_to_probs(
        self, similarities: torch.Tensor
    ) -> dict[Node, NodeProbabilities]:
        """
        Computes the log probabilities (left, right, arrival) for all nodes for the input x.

        :param similarities:
        :return: dictionary mapping each node to a dataclass containing tensors of shape (batch_size,)
        """
        node_to_log_p_right = self.get_node_to_log_p_right(similarities)
        return self._node_to_probs_from_p_right(node_to_log_p_right, self.tree_root)

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
    def update_leaf_distributions(
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
            self._update_leaf_distribution(leaf, node_to_prob, logits, y_true_logits)

    def _update_leaf_distribution(
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
        #count_alpha = 1 / leaf.dist_param_update_count
        #alpha = max(count_alpha, self.leaf_opt_ewma_alpha)
        leaf.dist_params.mul_(1.0 - self.leaf_opt_ewma_alpha)
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
