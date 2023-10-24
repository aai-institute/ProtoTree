import logging
from argparse import Namespace
from pathlib import Path
import json 
import cv2
import numpy as np
from torchvision import transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from src.util.data import get_dataloaders
from src.visualize.create.explanation.decision_flows_backup import (
    save_decision_flow_visualizations,
)
from src.visualize.create.explanation.multi_patch import save_multi_patch_visualizations
from src.visualize.create import patches  
from src.visualize.create.tree import save_tree_visualization
from src.visualize.prepare.explanations import data_explanations
from src.util.image import get_latent_to_pixel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("visualize_prototypes")

# TODO: put img size in config
IMG_SIZE = (224, 224)
LATENT_TO_PIXEL = get_latent_to_pixel(img_size=IMG_SIZE)

def show_global_explanation(proto_info: dict[str, dict]):

    for proto_id, proto_info in proto_info.items():
    
        img = cv2.resize(cv2.imread(proto_info["path"]), IMG_SIZE)
        patch = img[proto_info["bbox"][1]:proto_info["bbox"][3], proto_info["bbox"][0]:proto_info["bbox"][2]]
        bbox_inds = patches.BboxInds(w_low=proto_info["bbox"][0],
                                    h_low=proto_info["bbox"][1],
                                    w_high=proto_info["bbox"][2],
                                    h_high=proto_info["bbox"][3]
        )
        bbox = patches.Bbox(inds=bbox_inds, color=patches.ColorRgb(255, 255, 0), opacity=patches.Opacity(1))
    
        im_with_bbox = patches._superimpose_bboxs(img, [bbox])

        pixel_heatmap = LATENT_TO_PIXEL(np.array(proto_info["patch_similarities"]))
        colored_heatmap = patches._to_rgb_heatmap(pixel_heatmap)
        im_with_heatmap = 0.5 * (img / 255) + 0.2 * colored_heatmap

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(patch)
        axs[1].imshow(im_with_bbox)
        axs[2].imshow(im_with_heatmap) 
        plt.show()
            
def get_top_k_proto(local_scores: pd.DataFrame, top_k: int, use_similarity: bool =True):
           
    proto_similarities_df = local_scores.drop_duplicates("orig_similarity")
    proto_similarities_df = proto_similarities_df.sort_values(by="orig_similarity", ascending = not use_similarity)
    
    top_k_proto_sim = proto_similarities_df[:top_k]
    top_k_protos_id = top_k_proto_sim["prototype"]  
    
    return top_k_protos_id

def show_local_explanation(path: Path, prototypes_info: dict[str, dict], top_k_protos_id: list[int], scores: pd.DataFrame):
    img = plt.imread(path)
    
    fig = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig.subfigures(1, 2)
    
    axsLeft = subfigs[0].subplots(1, 1, sharey=True)
    subfigs[0].set_facecolor('0.85')
    
    axsLeft.set_xticks([])
    axsLeft.set_yticks([])
    axsLeft.imshow(img)
    
    
    axsRight= subfigs[1].subplots(2, len(top_k_protos_id), sharey=True)
    subfigs[1].set_facecolor('0.85') 
   
    for id, proto_id in enumerate(top_k_protos_id):
        
        proto_scores_df = scores.loc[scores["prototype"] == proto_id]
        scores_table = proto_scores_df[["modification", "delta"]]
        
        # def create_table(table: pd.DataFrame, ax):
        #     # ax = plt.subplot(111, frame_on=False) # no visible frame
        #     ax.xaxis.set_visible(False)  # hide the x axis
        #     ax.yaxis.set_visible(False)  # hide the y axis
    
        #     pd.plotting.table(ax, table, loc='center',
        #                     cellLoc='center')
        
        bbox = prototypes_info[str(proto_id)]["bbox"]
        axsRight[0][id].imshow(img[bbox[1]:bbox[3], bbox[0]:bbox[2]])

        axsRight[1][id].xaxis.set_visible(False)  # hide the x axis
        axsRight[1][id].yaxis.set_visible(False)  # hide the y axis
        scores_table.plot.base()
        #pd.plotting.table(axsRight[1][id], scores_table, loc='center', cellLoc='center')
         
    plt.show()
    
    
def explain_prototypes(args: Namespace):
   
    with open(args.prototypes_info_path) as f:
        prototypes_info = json.load(f) 
     
    if args.local_explanation:
    
        local_scores = pd.read_csv(Path(args.scores_dir) / "local_scores.csv")
        img_local_scores = local_scores.loc[local_scores["image"] == args.image]        
        top_k_protos_id = get_top_k_proto(img_local_scores, args.top_k, use_similarity=args.use_similarity)
        
        show_local_explanation(args.image, prototypes_info, top_k_protos_id, img_local_scores)
        
    else:
        # TODOs: add global scores to visualization
        global_scores = pd.read_csv(Path(args.scores_dir) / "global_scores.csv")
        show_global_explanation(prototypes_info)
     
        
        
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser("Visualize prototypes")
    parser.add_argument("--proto_info_dir", type=str, default="./runs/run_model/visualization/patches/prototree", 
                        help="Provide prototype info directory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of prototype to visualize")
    parser.add_argument("--scores_dir", type=str, default="./runs/run_model/scores/protoree", 
                        help="Provide scores csv file directory")
    parser.add_argument("--image", default="data/CUB_200_2011/dataset/test_crop/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg",
                        help="Provide image file")
    parser.add_argument("--local_explanation", action="store_true", help="Visualize local scores")
    parser.add_argument("--use_similarity", action="store_true", help="Protototype scores based on similarity or distance")
    args = parser.parse_args()
    
    explain_prototypes(args)

