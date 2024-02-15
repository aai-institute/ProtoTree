import argparse


# Utility functions for handling parsed arguments
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train a ProtoTree")
 
    parser.add_argument(
        "--config_file",
        type=str,
        default="./src/config/config.yml",
        help="Specify a configuration file",
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        help="Flag that disables GPU usage if set",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Specify type of the model [ProtoTree, ProtoPNet]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Specify type of the model [ProtoTree, ProtoPNet]",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        help="Specify model checkpoint to evaluate",
    )
    parser.add_argument(
        "--every_n_epochs", 
        type=int, 
        help="Checkpoint will be saved every n epochs during training."  
    )
    parser.add_argument(
        "--save_top_k", 
        type=int, 
        default=5,
        help="Number of best checkpoints saved during the training."  
    )
    parser.add_argument(
        "--explain_prototypes", 
        action="store_true", 
        help="Do prototypes explanation along model evaluation."
    )
    parser.add_argument(
        "--img_modifications", 
        '--names-list', 
        nargs='+', 
        help="Image modifications for prototypes explanation."
    )
    parser.add_argument(
        "--prototypes_info_path", 
        type=str, 
        help="Path of the json file containing prototype infos"
        )
    args = parser.parse_args()
 
    return args
