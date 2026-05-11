import os
import re
import time
from pathlib import Path
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn
import sklearn.metrics as skm

from DTE.data_generator import DataGenerator
from adbench.myutils import Utils
from data_prior.helen_feature_transform import FeatureTransform

from pfns.train_parallel import make_model_od
from pfns import encoders

import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize


# NOTE: This version can average weights only for checkpoints that have exactly the same model architecture and parameter shapes.
"""The Approach:
    1. scan all checkpoints
    2. group them by architecture signature
    3. average weights within each compatible group
    4. build one averaged model per group
    5. use those averaged models for prediction

So the sub-models in the ensemble need to have the same values of the following hyperparameters:
    1. seq_len
    2. max_feature_dim
    3. num_R
    4. emsize
    5. nhid
    6. nhead
    7. nlayer

For safety, the code also verifies that the actual state_dict keys and tensor shapes match before averaging.
"""



MODEL_PATH = Path("ckpt/weight_average_ensemble_without_data_numclusters")
RESULT_SAVE_PATH = Path("results/adbench/weight_average_ensemble_without_data_numclusters")


# =========================================================
# Model creation
# =========================================================
def make_model(seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None):
    criterion = nn.CrossEntropyLoss(
        weight=torch.ones(size=(num_class,)) / num_class,
        reduction="none",
        ignore_index=hps["ignore_index"],
    )

    model = make_model_od(
        criterion=criterion,
        encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        emsize=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayers,
        seq_len=seq_len,
        y_encoder_generator=encoders.Linear,
        extra_prior_kwargs_dict={"num_features": num_features},
        model_para_dict=model_para_dict,
    )
    return model


# =========================================================
# Inference
# =========================================================
def get_results(model, train_x, test_x, label, save_path, inst_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_y = None
    t0 = time.time()
    with torch.no_grad():
        logits = model(train_x, train_y, test_x.transpose(1, 0))
    t1 = time.time()

    return logits.cpu().numpy()[:, 0], t1 - t0


def low_density_anomalies(test_log_probs, num_anomalies):
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds


# =========================================================
# Parse checkpoint folders
# =========================================================
def collect_all_checkpoints(base_dir: Path = MODEL_PATH):
    """
    Collect every checkpoint and its model hyperparameters from the ensemble folder.
    No restriction to seeds. Every checkpoint found under each matching config directory
    is included.
    """
    pattern = re.compile(
        r"context(?P<seq_len>\d+)"
        r"\.feat(?P<max_feature_dim>\d+)"
        r"\.R(?P<num_R>\d+)"
        r"\.inf-full(?P<inflate_full>\w+)"
        r"\.LT(?P<apply_linear_transform>\w+)"
        r"\.gen1tr1(?P<gen_one_train_one>\w+)"
        r"\.reuse(?P<reuse_data_every_n>\d+)"
        r"\.E(?P<epochs>\d+)"
        r"\.step(?P<steps_per_epoch>\d+)"
        r"\.bs(?P<batch_size>\d+)"
        r"\.lr(?P<lr>[0-9.]+)"
        r"\.emb(?P<emsize>\d+)"
        r"\.hdim(?P<nhid>\d+)"
        r"\.nhead(?P<nhead>\d+)"
        r"\.nlayer(?P<nlayer>\d+)"
        r"\.ndevice(?P<num_device>\d+)"
    )

    checkpoints = []

    for context_dir in base_dir.iterdir():
        if not context_dir.is_dir():
            continue

        match = pattern.search(context_dir.name)
        if not match:
            print(f"Skipping (no match): {context_dir.name}")
            continue

        params = match.groupdict()
        for k, v in params.items():
            if v in ["True", "False"]:
                params[k] = (v == "True")
            elif re.fullmatch(r"\d+", v):
                params[k] = int(v)
            elif re.fullmatch(r"\d*\.\d+", v):
                params[k] = float(v)

        # find all best.ckpt recursively under this config folder
        found_any = False
        for ckpt_path in context_dir.rglob("best.ckpt"):
            found_any = True
            ckpt_info = dict(params)
            ckpt_info["ckpt_path"] = ckpt_path

            # optional metadata from parent folder name if it is like seedX
            parent_name = ckpt_path.parent.name
            if parent_name.startswith("seed"):
                try:
                    ckpt_info["seed"] = int(parent_name.replace("seed", ""))
                except ValueError:
                    ckpt_info["seed"] = parent_name
            else:
                ckpt_info["seed"] = parent_name

            checkpoints.append(ckpt_info)
            print(f"Found checkpoint: {ckpt_path}")

        if not found_any:
            print(f"No checkpoint found under: {context_dir}")

    return checkpoints


# =========================================================
# Grouping by architecture
# =========================================================
def architecture_signature(model_info):
    """
    Only checkpoints with identical signature are eligible for weight averaging.
    """
    return (
        model_info["seq_len"],
        model_info["max_feature_dim"],
        model_info["num_R"],
        model_info["emsize"],
        model_info["nhid"],
        model_info["nhead"],
        model_info["nlayer"],
    )


def group_checkpoints_by_architecture(checkpoints):
    groups = defaultdict(list)
    for ckpt in checkpoints:
        sig = architecture_signature(ckpt)
        groups[sig].append(ckpt)
    return groups


# =========================================================
# Weight averaging utilities
# =========================================================
def load_clean_state_dict(ckpt_path, device="cpu"):
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "state_dict" not in checkpoint:
        raise KeyError(f"'state_dict' not found in checkpoint: {ckpt_path}")

    state_dict = checkpoint["state_dict"]

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("model.", "", 1) if k.startswith("model.") else k
        cleaned_state_dict[new_k] = v

    return cleaned_state_dict


def verify_state_dict_compatibility(state_dicts, ckpt_paths):
    """
    Verify identical keys and tensor shapes before averaging.
    """
    ref_sd = state_dicts[0]
    ref_keys = list(ref_sd.keys())

    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != ref_keys:
            missing_in_current = set(ref_keys) - set(sd.keys())
            extra_in_current = set(sd.keys()) - set(ref_keys)
            raise ValueError(
                f"Checkpoint incompatible:\n"
                f"Reference: {ckpt_paths[0]}\n"
                f"Current:   {ckpt_paths[i]}\n"
                f"Missing keys: {missing_in_current}\n"
                f"Extra keys: {extra_in_current}"
            )

        for k in ref_keys:
            if ref_sd[k].shape != sd[k].shape:
                raise ValueError(
                    f"Shape mismatch for key '{k}':\n"
                    f"Reference {ckpt_paths[0]} shape: {ref_sd[k].shape}\n"
                    f"Current   {ckpt_paths[i]} shape: {sd[k].shape}"
                )


def average_state_dicts(ckpt_paths, device="cpu"):
    if len(ckpt_paths) == 0:
        raise ValueError("No checkpoint paths provided for averaging.")

    state_dicts = [load_clean_state_dict(p, device=device) for p in ckpt_paths]
    verify_state_dict_compatibility(state_dicts, ckpt_paths)

    avg_state_dict = OrderedDict()
    ref_sd = state_dicts[0]

    for k in ref_sd.keys():
        ref_tensor = ref_sd[k]

        if torch.is_floating_point(ref_tensor):
            stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
            avg_state_dict[k] = stacked.mean(dim=0).to(dtype=ref_tensor.dtype)
        else:
            # keep non-floating tensors unchanged from the first checkpoint
            avg_state_dict[k] = ref_tensor.clone()

    return avg_state_dict


def build_weight_averaged_model_from_group(group_ckpts, hyperparameters, device):
    """
    group_ckpts: list of compatible checkpoints
    """
    ref = group_ckpts[0]

    seq_len = ref["seq_len"]
    max_feature_dim = ref["max_feature_dim"]
    num_R = ref["num_R"]
    emsize = ref["emsize"]
    nhid = ref["nhid"]
    nhead = ref["nhead"]
    nlayer = ref["nlayer"]

    model = make_model(
        seq_len=seq_len,
        num_features=max_feature_dim,
        hps=hyperparameters,
        emsize=emsize,
        nhead=nhead,
        nhid=nhid,
        nlayers=nlayer,
        num_class=2,
        model_para_dict={"num_R": num_R},
    ).to(device)

    ckpt_paths = [x["ckpt_path"] for x in group_ckpts]
    avg_state_dict = average_state_dicts(ckpt_paths, device=device)

    missing_keys, unexpected_keys = model.load_state_dict(avg_state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        raise RuntimeError(
            f"Problem loading averaged weights.\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}"
        )

    model.eval()
    return model


# =========================================================
# Main
# =========================================================
@hydra.main(version_base="1.3", config_path="configuration", config_name="config")
def main(cfg: DictConfig):
    train_cfg = cfg.train
    test_cfg = cfg.test

    hyperparameters = train_cfg.hyperparameters
    device = train_cfg.device
    train_seed = train_cfg.seed

    feature_truncation = test_cfg.feature_truncation
    seed = test_cfg.seed
    preprocess_transform = test_cfg.preprocess_transform

    all_checkpoints = collect_all_checkpoints()
    if len(all_checkpoints) == 0:
        raise RuntimeError(f"No valid checkpoints found under {MODEL_PATH}")

    grouped_ckpts = group_checkpoints_by_architecture(all_checkpoints)

    print("\nGrouped compatible checkpoints:")
    for sig, ckpts in grouped_ckpts.items():
        print(f"Architecture {sig}: {len(ckpts)} checkpoints")

    utils = Utils()
    utils.set_seed(seed)

    setting = "semi"
    datagenerator = DataGenerator(seed=seed, test_size=0.5, normal=True)

    def add_feature_transform(x, eval_position, FT, max_feature_dim):
        if preprocess_transform is not None:
            x = FT.pfn_inference_transform(
                eval_xs=x,
                preprocess_transform=preprocess_transform,
                eval_position=eval_position,
                normalize_with_test=True,
                rescale_with_sqrt=False,
            )

        feature_dim = x.shape[-1]
        if feature_dim > max_feature_dim:
            if feature_truncation == "projection":
                x = FT.feature_sparse_projection(x=x, num_feature=max_feature_dim)
            else:
                x = FT.feature_subsampling(x=x, num_feature=max_feature_dim)
        elif feature_dim < max_feature_dim:
            x = FT.feature_padding(x=x, num_feature=max_feature_dim)

        assert x.shape[-1] == max_feature_dim, (
            f"Feature mismatch: got {x.shape[-1]}, expected {max_feature_dim}"
        )
        return x

    def make_train_test(data, FT, max_feature_dim, seq_len):
        x_train, y_train = data["X_train"], data["y_train"]
        x_test, y_test = data["X_test"], data["y_test"]

        if x_train.shape[0] <= seq_len - 1:
            train_x = x_train
        else:
            idx = np.random.choice(x_train.shape[0], seq_len - 1, replace=False)
            train_x = x_train[idx]

        train_and_test = add_feature_transform(
            x=np.concatenate([train_x, x_test], axis=0),
            eval_position=len(train_x),
            FT=FT,
            max_feature_dim=max_feature_dim,
        )

        train_x = train_and_test[:len(train_x), :]
        x_test = train_and_test[len(train_x):, :]

        test_in = x_test[y_test == 0]
        test_out = x_test[y_test == 1]

        train_x = torch.from_numpy(train_x).to(device).unsqueeze(1).float()
        test_in = torch.from_numpy(test_in).to(device).unsqueeze(0).float()
        test_out = torch.from_numpy(test_out).to(device).unsqueeze(0).float()

        return train_x, test_in, test_out

    # =====================================================
    # Evaluate on all datasets
    # =====================================================
    for dataset_list in [
        datagenerator.dataset_list_classical,
        datagenerator.dataset_list_cv,
        datagenerator.dataset_list_nlp,
    ]:
        for dataset in dataset_list:
            save_path = RESULT_SAVE_PATH / f"train_seed={train_seed}/{dataset}/{setting}"
            os.makedirs(save_path, exist_ok=True)

            datagenerator.dataset = dataset
            data = datagenerator.generator(la=0, max_size=50000)

            all_scores = []
            timing_list = []
            final_labels = None
            final_context_len = None

            print(f"\n{'=' * 80}")
            print(f"Dataset: {dataset}")
            print(f"{'=' * 80}")

            for sig, group_ckpts in grouped_ckpts.items():
                ref = group_ckpts[0]
                seq_len = ref["seq_len"]
                max_feature_dim = ref["max_feature_dim"]

                print(
                    f"\nEvaluating architecture group: "
                    f"seq_len={seq_len}, feat_dim={max_feature_dim}, "
                    f"num_ckpts={len(group_ckpts)}"
                )

                FT = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim)

                trained_model = build_weight_averaged_model_from_group(
                    group_ckpts=group_ckpts,
                    hyperparameters=hyperparameters,
                    device=device,
                )

                train_x, test_in, test_out = make_train_test(data, FT, max_feature_dim, seq_len)

                logits_in, time_in = get_results(trained_model, train_x, test_in, 0, save_path, "in")
                logits_out, time_out = get_results(trained_model, train_x, test_out, 1, save_path, "out")

                logits = np.concatenate([logits_in, logits_out], axis=0)
                labels = np.array([0] * len(logits_in) + [1] * len(logits_out))

                logits = logits - np.max(logits, axis=1, keepdims=True)
                exp_logits = np.exp(logits)
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                all_scores.append(probabilities[:, 1])
                timing_list.append((time_in + time_out) / len(labels))

                final_labels = labels
                final_context_len = train_x.shape[0]

            # Combine predictions from all averaged architecture groups
            all_scores = np.vstack(all_scores)
            np.savez(
                save_path / f"all_scores_seed{seed}.npz",
                all_scores=all_scores,
                labels=final_labels,
            )

            avg_score = np.mean(all_scores, axis=0)
            avg_score[np.isnan(avg_score)] = 0

            preds = low_density_anomalies(-avg_score, np.sum(final_labels == 1))
            f1 = skm.f1_score(final_labels, preds)

            metric_result = utils.metric(y_true=final_labels, y_score=avg_score)

            result = {
                seed: {
                    "aucroc": metric_result["aucroc"],
                    "aucpr": metric_result["aucpr"],
                    "f1": f1,
                    "time-per-inst": float(np.mean(timing_list)),
                    "context-len": final_context_len,
                }
            }

            df = pd.DataFrame(result).T.reset_index()
            df.columns = ["seed", "aucroc", "aucpr", "f1", "time-per-inst", "context-len"]

            result_csv = save_path / "result.csv"
            if result_csv.exists():
                df.to_csv(result_csv, mode="a", index=False, header=False)
            else:
                df.to_csv(result_csv, mode="w", index=False, header=True)

            print(
                f"[{dataset}] "
                f"AUCROC={metric_result['aucroc']:.4f}, "
                f"AUCPR={metric_result['aucpr']:.4f}, "
                f"F1={f1:.4f}"
            )


def run_with_overrides(base_overrides, overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="configuration")
    cfg = compose(config_name="config", overrides=base_overrides + overrides)
    main(cfg)


if __name__ == "__main__":
    main()