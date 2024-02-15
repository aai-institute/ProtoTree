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
from src.visualize.create.explanation.multi_patch import save_multi_patch_visualizations, save_multi_patch_visualizations
from src.visualize.prepare.explanations import data_explanations
from src.visualize.create.explanation.prototypes import save_top_k_prototypes

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_prototree")


def test_model(config: dict):
    # data and paths
    output_dir = Path(
        config["output_dir"]
    ).resolve()  # Absolute path makes it easier for Graphviz to find images.

    # Architecture args
    model_type = config["model_type"]

    log.info(f"Testing a {model_type} model with {config=}.")

    # PREPARE DATA
    modifications = config["img_modifications"]
    explain_proto = config["explain"]
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
        case _:
            raise ValueError(f"Unknown model type {model_type}.")
            
    # SAVE LOCAL SCORES:
    scores_dir = output_dir / "scores" / model_type
    scores_dir.mkdir(parents=True, exist_ok=True)
    local_score_expl.to_csv(scores_dir / "local_score.csv")
        
    # SAVE VISUALIZATIONS
    vis_dir = output_dir / "visualizations"
    patches_dir = vis_dir / "patches" / model_type
    with open(patches_dir / "proto_info.json") as f:
        prototypes_info = json.load(f)
    
    match (model_type):
        case "prototree":
            save_multi_patch_visualizations(
                explanations_provider(), vis_dir / "explanations", local_score_expl
            )
            save_decision_flow_visualizations(
                explanations_provider(), prototypes_info, vis_dir / "explanations", patches_dir=patches_dir, scores=local_score_expl,img_size=img_size, 
            )
        case "protopnet":
            assert local_score_expl is not None, "Saving top k prototypes works only with local scores previously computation"
            save_top_k_prototypes(prototypes_info=prototypes_info, scores=local_score_expl, top_k=config["save_top_k"], img_size=img_size)
            
        case _:
            raise ValueError(f"Unknown model type {model_type}.")

    
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

    parsed_args = vars(get_args())
    parsed_args =  {k: v for k, v in parsed_args.items() if v is not None}
    
    with open(parsed_args["config_file"], "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    
    config.update(parsed_args)
    
    test_model(config)
