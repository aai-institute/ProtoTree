import numpy as np

from prototree.node import Leaf

# from prototree.test import eval_ensemble
from util.log import Log


def log_learning_rates(
    optimizer, net: str, log: Log, disable_derivative_free_leaf_optim=False
):
    log.log_message("Learning rate net: " + str(optimizer.param_groups[0]["lr"]))
    if "resnet50_inat" in net:
        log.log_message("Learning rate block: " + str(optimizer.param_groups[1]["lr"]))
        log.log_message(
            "Learning rate net 1x1 conv: " + str(optimizer.param_groups[2]["lr"])
        )
    else:
        log.log_message(
            "Learning rate net 1x1 conv: " + str(optimizer.param_groups[1]["lr"])
        )
    if disable_derivative_free_leaf_optim:
        log.log_message(
            "Learning rate prototypes: " + str(optimizer.param_groups[-2]["lr"])
        )
        log.log_message(
            "Learning rate leaves: " + str(optimizer.param_groups[-1]["lr"])
        )
    else:
        log.log_message(
            "Learning rate prototypes: " + str(optimizer.param_groups[-1]["lr"])
        )


# TODO: this has to be called as leaf_labels = func(leaf_labels...). Fix this pattern!
# TODO: the new name says it all, refactor this!
def log_pruned_leaf_analysis(
    leaves: list[Leaf],
    pruning_threshold: float,
    log: Log,
):

    unpruned_leaves = []
    classes_covered = set()
    for leaf in leaves:
        classes_covered.add(leaf.predicted_label())
        if leaf.conf_predicted_label() > pruning_threshold:
            unpruned_leaves.append(leaf.index)

    log.log_message(
        f"\nLeafs with max > {pruning_threshold:.2f}: {len(unpruned_leaves)}"
    )

    num_classes = leaves[0].num_classes
    class_labels_without_leaf = set(range(num_classes)) - classes_covered
    if class_labels_without_leaf:
        log.log_message(f"Classes without leaf: {class_labels_without_leaf}")


#
# def analyse_ensemble(
#     log,
#     args,
#     test_loader,
#     device,
#     trained_orig_trees,
#     trained_pruned_trees,
#     trained_pruned_projected_trees,
#     orig_test_accuracies,
#     pruned_test_accuracies,
#     pruned_projected_test_accuracies,
#     project_infos,
#     infos_sample_max,
#     infos_greedy,
#     infos_fidelity,
# ):
#     print(
#         "\nAnalysing and evaluating ensemble with %s trees of height %s..."
#         % (len(trained_orig_trees), args.depth),
#         flush=True,
#     )
#     log.log_message(
#         "\n-----------------------------------------------------------------------------------------------------------------"
#     )
#     log.log_message(
#         "\nAnalysing and evaluating ensemble with %s trees of height %s..."
#         % (len(trained_orig_trees), args.depth)
#     )
#
#     """
#     CALCULATE MEAN AND STANDARD DEVIATION BETWEEN RUNS
#     """
#     log.log_message(
#         "Test accuracies of original individual trees: %s" % str(orig_test_accuracies)
#     )
#     log.log_message(
#         "Mean and standard deviation of accuracies of original individual trees: \n"
#         + "mean="
#         + str(np.mean(orig_test_accuracies))
#         + ", std="
#         + str(np.std(orig_test_accuracies))
#     )
#
#     log.log_message(
#         "Test accuracies of pruned individual trees: %s" % str(pruned_test_accuracies)
#     )
#     log.log_message(
#         "Mean and standard deviation of accuracies of pruned individual trees: \n"
#         + "mean="
#         + str(np.mean(pruned_test_accuracies))
#         + ", std="
#         + str(np.std(pruned_test_accuracies))
#     )
#
#     log.log_message(
#         "Test accuracies of pruned and projected individual trees: %s"
#         % str(pruned_projected_test_accuracies)
#     )
#     log.log_message(
#         "Mean and standard deviation of accuracies of pruned and projected individual trees:\n "
#         + "mean="
#         + str(np.mean(pruned_projected_test_accuracies))
#         + ", std="
#         + str(np.std(pruned_projected_test_accuracies))
#     )
#
#     """
#     CALCULATE MEAN NUMBER OF PROTOTYPES
#     """
#     nums_prototypes = []
#     for t in trained_pruned_trees:
#         nums_prototypes.append(t.num_descendants)
#     log.log_message(
#         "Mean and standard deviation of number of prototypes in pruned trees:\n "
#         + "mean="
#         + str(np.mean(nums_prototypes))
#         + ", std="
#         + str(np.std(nums_prototypes))
#     )
#
#     """
#     CALCULATE MEAN DISTANCE TO NEAREST PROTOTYPE
#     """
#     distances = []
#     for i in range(len(trained_pruned_projected_trees)):
#         info = project_infos[i]
#         tree = trained_pruned_projected_trees[i]
#         distances += average_distance_nearest_image(info, tree, log, disable_log=True)
#     log.log_message(
#         "Mean and standard deviation of distance from prototype to nearest training patch:\n "
#         + "mean="
#         + str(np.mean(distances))
#         + ", std="
#         + str(np.std(distances))
#     )
#
#     """
#     CALCULATE MEAN AND STANDARD DEVIATION BETWEEN RUNS WITH DETERMINISTIC ROUTING
#     """
#     accuracies = []
#     for info in infos_sample_max:
#         accuracies.append(info["test_accuracy"])
#     log.log_message(
#         "Mean and standard deviation of accuracies of pruned and " \
#         "projected individual trees with sample_max routing:\n "
#         + "mean="
#         + str(np.mean(accuracies))
#         + ", std="
#         + str(np.std(accuracies))
#     )
#     accuracies = []
#     for info in infos_greedy:
#         accuracies.append(info["test_accuracy"])
#     log.log_message(
#         "Mean and standard deviation of accuracies of pruned and projected individual trees with greedy routing:\n "
#         + "mean="
#         + str(np.mean(accuracies))
#         + ", std="
#         + str(np.std(accuracies))
#     )
#
#     """
#     CALCULATE FIDELITY BETWEEN RUNS WITH DETERMINISTIC ROUTING
#     """
#     fidelities_sample_max = []
#     fidelities_greedy = []
#     for info in infos_fidelity:
#         fidelities_sample_max.append(info["distr_samplemax_fidelity"])
#         fidelities_greedy.append(info["distr_greedy_fidelity"])
#     log.log_message(
#         "Mean and standard deviation of fidelity of pruned and projected individual trees with sample_max routing:\n "
#         + "mean="
#         + str(np.mean(fidelities_sample_max))
#         + ", std="
#         + str(np.std(fidelities_sample_max))
#     )
#     log.log_message(
#         "Mean and standard deviation of fidelity of pruned and projected individual trees with greedy routing:\n "
#         + "mean="
#         + str(np.mean(fidelities_greedy))
#         + ", std="
#         + str(np.std(fidelities_greedy))
#     )
#
#     """
#     CALCULATE MEAN AND STANDARD DEVIATION OF PATH LENGTH WITH DETERMINISTIC ROUTING
#     """
#     depths_sample_max = []
#     depths_greedy = []
#     for i in range(len(trained_pruned_projected_trees)):
#         tree = trained_pruned_projected_trees[i]
#         eval_info_sample_max = infos_sample_max[i]
#         eval_info_greedy = infos_greedy[i]
#         depths_sample_max += get_avg_path_length(tree, eval_info_sample_max, log)
#         depths_greedy += get_avg_path_length(tree, eval_info_greedy, log)
#     log.log_message(
#         "Mean and standard deviation of path length of pruned " \
#         "and projected individual trees with sample_max routing:\n "
#         + "mean="
#         + str(np.mean(depths_sample_max))
#         + ", std="
#         + str(np.std(depths_sample_max))
#     )
#     log.log_message(
#         "Tree with sample_max deterministic routing. Longest path has length %s, shortest path has length %s"
#         % ((np.max(depths_sample_max)), np.min(depths_sample_max))
#     )
#     log.log_message(
#         "Mean and standard deviation of path length of pruned and projected individual trees with greedy routing:\n "
#         + "mean="
#         + str(np.mean(depths_greedy))
#         + ", std="
#         + str(np.std(depths_greedy))
#     )
#     log.log_message(
#         "Tree with greedy deterministic routing. Longest path has length %s, shortest path has length %s"
#         % ((np.max(depths_greedy)), np.min(depths_greedy))
#     )
#
#     """
#     EVALUATE ENSEMBLE OF PRUNED AND PROJECTED TREES
#     """
#     log.log_message(
#         "\nCalculating accuracy of tree ensemble with pruned and projected trees..."
#     )
#     eval_ensemble(
#         trained_pruned_projected_trees, test_loader, device, log, args, "distributed"
#     )
