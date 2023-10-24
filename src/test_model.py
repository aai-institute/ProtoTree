import logging
from argparse import Namespace
from pathlib import Path
import torch
import json
import yaml
from src.core.models import ProtoTree, ProtoPNet
from src.core.eval import eval_model, single_leaf_eval
from src.util.args import get_args
from src.util.data import get_dataloader

from src.visualize.create.explanation.decision_flows import (
    save_decision_flow_visualizations,
)
from src.visualize.create.explanation.multi_patch import save_multi_patch_visualizations_with_local_score, save_multi_patch_visualizations
from src.visualize.create.patches import save_patch_visualizations
from src.visualize.create.tree import save_tree_visualization
from src.visualize.prepare.explanations import data_explanations

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_prototree")


def test_model(config: dict):
    # data and paths
    output_dir = Path(
        config["output_dir"]
    ).resolve()  # Absolute path makes it easier for Graphviz to find images.

    # Architecture args
    model_type = config[".model_type"]

    # TODO: instead of args list the hyperparameters from the checkpoint
    log.info(f"Testing a {model_type} model with {config=}.")

    # PREPARE DATA
    modifications = config["img_modifications"]
    explain_proto = config["explain_prototypes "]
    test_dir = Path(config["dataset_dir"]) / config["test_dir"]
    img_size = config["img_size"]
    test_loader = get_dataloader(dataset_dir=test_dir, augment=False, img_size=img_size, explain=explain_proto, modifications=modifications, loader_batch_size=1, num_workers=4)

    class_names = test_loader.dataset.classes
    num_classes = len(class_names)
    log.info(f"Num classes: {num_classes}")


    # prototree specifics
    leaf_pruning_threshold = config["leaf_pruning_multiplier"] / num_classes
 
    # PREPARE MODEL and LOAD WEIGHTS
    disable_cuda = config["disable_cuda"]
    device = "cpu" if disable_cuda else "cuda"
    match config["model_type"]:
        case "protopnet":
            model = ProtoPNet.load_from_checkpoint(config["model_checkpoint"], map_location=torch.device(device))
        case "prototree":
            model = ProtoTree.load_from_checkpoint(config["model_checkpoint"], map_location=torch.device(device))
        case _:
            raise ValueError(f"Unknown model type {model_type}.")

    model = model.eval()

    match (model_type):
        case "protopnet":
            if explain_proto:
                acc, local_score_expl = eval_model(model, test_loader, explain=True)
            else:
                acc = eval_model(model, test_loader)
            log.info(f"\nTest acc.: {acc:.3f}")
            
        case "prototree":
            model.log_state()
            model.prune(leaf_pruning_threshold)
            if explain_proto:
                pruned_acc, local_score_expl = eval_model(model, test_loader, explain=True)
            else:
                pruned_acc = eval_model(model, test_loader)
            log.info(f"\nTest acc. after pruning: {pruned_acc:.3f}")
            model.log_state()
            single_leaf_eval(model, test_loader, "Pruned")

            def explanations_provider():
                return data_explanations(
                        model, test_loader, class_names, local_score_expl, explain_proto
                    )  # This is lazy to enable iterator reuse.

            model.print()
            
    # SAVE LOCAL SCORES:
    scores_dir = output_dir / "scores" / model_type
    scores_dir.mkdir(parents=True, exist_ok=True)
    local_score_expl.to_csv(scores_dir / "local_score.csv")
        
    # SAVE VISUALIZATIONS
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches" / model_type
    with open(patches_dir / "proto_info.json") as f:
        prototypes_info = json.load(f)
    
    if model_type == "prototree":
        # save_multi_patch_visualizations(
        #     explanations_provider(), vis_dir / "explanations"
        # )
        # save_multi_patch_visualizations_with_local_score(
        #     explanations_provider(), vis_dir / "explanations", local_score_expl
        # )
        # TODOs: img size will be in the config!
        save_decision_flow_visualizations(
            explanations_provider(), prototypes_info, vis_dir / "explanations", img_size=(224, 244)
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
    
    with open(parsed_args.config_file, "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    config.update(vars(parsed_args))
    
    test_model(config)
