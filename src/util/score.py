import os

import pandas as pd


def globale_scores(scores: pd.DataFrame, out_dir: os.PathLike):
    """
    Compute the global scores for the learned prototypes

    :param scores: dataframe with local scores for each training image
    :param out_dir: directory where to save the global scores as a csv file
    :return the dataframe of the global scores
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    prototype_global_scores = list()
    n_proto = len(scores["prototype"].drop_duplicates())
    mods = scores["modification"].drop_duplicates()

    for proto_idx in range(n_proto):
        prototype_score = dict(prototype=proto_idx)
        for mod in mods:
            cond = (scores["prototype"] == proto_idx) & (scores["modification"] == mod)
            df_global = scores.loc[cond]

            deltas = df_global["delta"].values
            simils = df_global["orig_similarity"].values
            global_score = (deltas * simils).sum() / simils.sum()

            prototype_score[mod] = global_score
        prototype_global_scores.append(prototype_score)

    global_scores = pd.DataFrame(prototype_global_scores)

    global_scores.to_csv(out_dir / "global_scores.csv")
    return global_scores
