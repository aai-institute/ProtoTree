from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from src.util.image import get_latent_to_pixel
from src.visualize.create import patches


def top_k_prototypes(local_scores: pd.DataFrame, top_k: int, use_distance: bool = True):
    """
    Get the k prototypes with highest similarity value

    :param local_scores: dataframe with prototype ids, their local scores (if computed) and similarity values
    :param top_k: number of best prototypes
    :param use_similarity: if True order prototypes in ascending order otherwise descending
    :return: list of best prototype ids
    """
    df = local_scores.drop_duplicates("prototype")
    df = df.sort_values(by="orig_similarity", ascending=use_distance)

    return df[:top_k]["prototype"]


def save_top_k_prototypes(
    prototypes_info: dict[str, dict],
    scores: pd.DataFrame = None,
    top_k: int = 5,
    img_size: tuple = (224, 244),
    save: bool = True,
):
    """
    Save or visualize the top k prototypes for a dataset of images

    :param dataloader: pytorch dataloader
    :param prototypes_info: json file with information regarding prototypes (bbox coordinates, original training image)
    :param scores: dataframe with prototype ids, their local scores (if computed) and similarity values
    :param top_k: number of best prototypes
    :param save: True if we want to save prototypes visualization
    """

    paths = scores["image"].drop_duplicates()
    plot_scores = scores is not None

    for path in paths:
        img_scores = scores.loc[scores["image"] == path]
        top_k_protos_id = top_k_prototypes(img_scores, top_k=top_k)
        img = plt.imread(path)

        n = 2 if plot_scores else 1
        fig, axs = plt.subplots(2, top_k, figsize=(15, 6), layout="constrained")
        for id, proto_id in enumerate(top_k_protos_id):
            bbox = prototypes_info[str(proto_id)]["bbox"]
            proto_scores_df = round(
                img_scores.loc[img_scores["prototype"] == proto_id], 3
            )

            if plot_scores:
                axs[0][id].imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2]])
                axs[0][id].yaxis.set_visible(False)
                axs[0][id].xaxis.set_visible(False)
                # axs[1][id] = proto_scores_df.plot.bar("modification", "delta")
                proto_scores_df.plot(
                    x="modification", y="delta", ax=axs[1][id], kind="bar", legend=False
                )
                axs[1][id].bar_label(axs[1][id].containers[0])
                axs[1][id].yaxis.set_visible(False)
                axs[1][id].xaxis.set_label_text("")
                axs[1][id].set_facecolor("0.9")
            else:
                axs[id].imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2]])
                axs[id].yaxis.set_visible(False)
                axs[id].xaxis.set_visible(False)

        # if save:
        #     # TODO
        #     plt.savefig("")
        #     return
        plt.show()


def save_prototypes(
    proto_info: dict[str, dict],
    global_expl: pd.DataFrame = None,
    img_size: tuple = (224, 224),
):
    """
    Save or visualize the learned prototypes

    :param prototypes_info: json file with information regarding prototypes (bbox coordinates, original training image)
    :param global_expl: dataframe with prototype ids and their global scores
    :param save: True if we want to save prototypes visualization
    """
    latent_to_pixel = get_latent_to_pixel(img_size)

    if global_expl is not None:
        global_expl = global_expl.set_index("prototype")

    for proto_id, proto in proto_info.items():
        img = cv2.cvtColor(
            cv2.resize(cv2.imread(proto["path"]), img_size), cv2.COLOR_BGR2RGB
        )
        img = img / 255
        patch = img[
            proto["bbox"][1] : proto["bbox"][3], proto["bbox"][0] : proto["bbox"][2]
        ]
        bbox_inds = patches.BboxInds(
            w_low=proto["bbox"][0],
            h_low=proto["bbox"][1],
            w_high=proto["bbox"][2],
            h_high=proto["bbox"][3],
        )
        bbox = patches.Bbox(
            inds=bbox_inds,
            color=patches.ColorRgb(255, 255, 0),
            opacity=patches.Opacity(1),
        )
        im_with_bbox = patches._superimpose_bboxs(img, [bbox])

        pixel_heatmap = latent_to_pixel(np.array(proto["patch_similarities"]))
        colored_heatmap = patches._to_rgb_heatmap(pixel_heatmap)
        im_with_heatmap = 0.5 * (img) + 0.2 * colored_heatmap

        n = 3 if global_expl is None else 4
        fig, axs = plt.subplots(1, n, figsize=(15, 5))
        axs[0].imshow(patch)
        axs[1].imshow(im_with_bbox)
        axs[2].imshow(im_with_heatmap)

        axs[0].set_title("Prototype patch")
        axs[1].set_title("Prototype bbox")
        axs[2].set_title("Prototype heatmap")

        axs[0].axis("off")
        axs[1].axis("off")
        axs[2].axis("off")

        if global_expl is not None:
            proto_global_expl = round(global_expl.loc[int(proto_id)], 3)

            axs[3] = proto_global_expl.plot.bar()
            axs[3].set_title("Prototype global explanation")
            axs[3].bar_label(axs[3].containers[0])
            asp = np.abs(np.diff(axs[3].get_xlim())[0] / np.diff(axs[3].get_ylim())[0])
            axs[3].set_aspect(asp)
            axs[3].get_yaxis().set_visible(False)
            axs[3].set_facecolor("0.85")

        plt.show()
