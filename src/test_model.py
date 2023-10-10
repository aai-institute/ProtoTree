import logging
from argparse import Namespace
from pathlib import Path

from src.core.models import ProtoTree, ProtoPNet
from src.core.eval import eval_model, single_leaf_eval, eval_protopnet_model
from src.util.args import get_args
from src.util.data import get_dataloader
from src.config import test_dir
from src.visualize.create.explanation.decision_flows import (
    save_decision_flow_visualizations,
)
from src.visualize.create.explanation.multi_patch import save_multi_patch_visualizations
from src.visualize.create.patches import save_patch_visualizations
from src.visualize.create.tree import save_tree_visualization
from src.visualize.prepare.explanations import data_explanations

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_prototree")


# TODO: remove dependency on args everywhere
def test_model(args: Namespace):
    # data and paths
    output_dir = Path(
        args.output_dir
    ).resolve()  # Absolute path makes it easier for Graphviz to find images.

    # Architecture args
    model_type = args.model_type 

    # TODO: instead of args list the hyperparameters from the checkpoint
    log.info(f"Testing a {model_type} model with {args=}.")

    # PREPARE DATA
    test_loader = get_dataloader(dataset_dir=test_dir, augment=False, loader_batch_size=1, num_workers=4)

    class_names = test_loader.dataset.classes
    num_classes = len(class_names)
    log.info(f"Num classes: {num_classes}")

    # PREPARE PRUNING
    # Leaves that didn't learn anything will have a distribution close to torch.ones(num_classes) / num_classes, so we
    # prune leaves where no probability is fairly close to the "no-learning" value of 1 / num_classes. The
    # multiplication by leaf_pruning_multiplier determines what's considered close enough to "no-learning".
    # TODO: Perhaps we should instead tune this threshold with a search algorithm.
    
    # prototree specifics
    leaf_pruning_threshold = args.leaf_pruning_multiplier / num_classes
 
 
    # PREPARE MODEL and LOAD WEIGHTS
    match args.model_type:
        case "protopnet":
            model = ProtoPNet.load_from_checkpoint(args.model_checkpoint)
        case "prototree":
            model = ProtoTree.load_from_checkpoint(args.model_checkpoint)
        case _:
            raise ValueError(f"Unknown model type {model_type}.")

    model = model.eval()

    match (model_type):
        case "protopnet":
            acc = eval_protopnet_model(model, test_loader)
            log.info(f"\nTest acc.: {acc:.3f}")
            
        case "prototree":
            model.log_state()
            model.prune(leaf_pruning_threshold)
            pruned_acc = eval_model(model, test_loader)
            log.info(f"\nTest acc. after pruning: {pruned_acc:.3f}")
            model.log_state()
            single_leaf_eval(model, test_loader, "Pruned")

            def explanations_provider():
                return data_explanations(
                    model, test_loader, class_names
                )  # This is lazy to enable iterator reuse.

            model.print()
            
            
    # SAVE VISUALIZATIONS
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches"
    save_patch_visualizations(model.proto_patch_matches, patches_dir)
    if model_type == "prototree":
        save_tree_visualization(model, patches_dir, vis_dir / "tree", class_names)
        save_multi_patch_visualizations(
            explanations_provider(), vis_dir / "explanations"
        )
        save_decision_flow_visualizations(
            explanations_provider(), patches_dir, vis_dir / "explanations"
        )


if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        try:
            import lovely_tensors

            lovely_tensors.monkey_patch()
        except ImportError:
            log.warning(
                "lovely_tensors not installed, not monkey patching. "
                "For more efficient debugging, we recommend installing it with `pip install lovely-tensors`."
            )

    parsed_args = get_args()
    test_model(parsed_args)
