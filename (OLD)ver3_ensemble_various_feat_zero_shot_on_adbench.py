import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from DTE.data_generator import DataGenerator
from adbench.myutils import Utils
from data_prior.helen_feature_transform  import FeatureTransform
import sklearn.metrics as skm

from pfns.train_parallel import make_model_od
from pfns import encoders

import hydra
from omegaconf import DictConfig
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

MODEL_PATH = Path("ckpt/test_validation_ensemble_11")
RESULT_SAVE_PATH = Path("results/adbench/test_validation_ensemble_11")

# ADDING numclusters parameters to the folder name of the model

# Generate a TransformerModel
def make_model(seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None):
    criterion = nn.CrossEntropyLoss(weight=torch.ones(size=(num_class,)) / num_class, reduction='none',
                                    ignore_index=hps['ignore_index'])

    # now train
    model = make_model_od(
        criterion=criterion, encoder_generator=encoders.get_normalized_uniform_encoder(encoders.Linear),
        # define the transformer size
        emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers,
        # seq_len defines the size of your datasets (including the test set)
        seq_len=seq_len,
        y_encoder_generator=encoders.Linear,
        # these are given to the prior, which needs to know how many features we have etc
        extra_prior_kwargs_dict={'num_features': num_features},
        model_para_dict=model_para_dict,
    )
    return model


def get_results(model, train_x, test_x, label, save_path, inst_type):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # train_x: (seq_len-1, 1, d)
    train_y = None
    # test_x: (1[seq-len], num_text_x[batch_size], d[#feat])

    total_time = 0
    with torch.no_grad():
        model.eval()
        # we add our batch dimension, as our transformer always expects that
        logits = []
        from tqdm import tqdm
        for i in tqdm(range(test_x.shape[1])):
            start_time = time.time()
            inst_logits = model(train_x, train_y, test_x[:, i, :].unsqueeze(1))
            # (bs=1, num_test_x=1, num_classes=2)

            duration = time.time() - start_time
            total_time += duration

            inst_logits = inst_logits.squeeze(0)  # squeeze the batch_size=1, get: (num_test_x=1, num_classes=2)
            logits.append(inst_logits)

        logits = torch.concatenate(logits, dim=0).cpu().numpy()

        return logits, total_time
    

def low_density_anomalies(test_log_probs, num_anomalies):
    """ Helper function for the F1-score, selects the num_anomalies lowest values of test_log_prob
    """
    anomaly_indices = np.argpartition(test_log_probs, num_anomalies - 1)[:num_anomalies]
    preds = np.zeros(len(test_log_probs))
    preds[anomaly_indices] = 1
    return preds


def extra_parameters_of_models_in_ensemble(base_dir : Path = MODEL_PATH):
    """extract model configurations from ensemble directories"""
    # Base directory where contains all models of an ensemble 
    #base_dir = Path("../ckpt")

    # Regex pattern for the top-level context directories
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
        r"\.numclusters(?P<max_num_cluster>\d+)"   # ADDING numclusters
    )

    results = []

    for context_dir in base_dir.iterdir():
        if not context_dir.is_dir():
            continue

        match = pattern.search(context_dir.name)
        if not match:
            print(f"Skipping (no match): {context_dir.name}")
            continue

        # Extract parameters and convert types
        params = match.groupdict()
        for k, v in params.items():
            if v in ["True", "False"]:
                params[k] = v == "True"
            elif re.fullmatch(r"\d+", v):
                params[k] = int(v)
            elif re.fullmatch(r"\d*\.\d+", v):
                params[k] = float(v)

        # Get seed directories and files within
        seeds = []
        for seed_dir in context_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith("seed"):
                ckpt_path = seed_dir / "best.ckpt"
                print(ckpt_path)
                seed_info = {
                    "seed": int(seed_dir.name.replace("seed", "")),
                    "ckpt_path": ckpt_path
                }
                seeds.append(seed_info)

        params["seeds"] = seeds
        results.append(params)
    
    return results


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

    models_in_ensemble = extra_parameters_of_models_in_ensemble()
    utils = Utils()
    utils.set_seed(seed)

    setting = "semi"
    datagenerator = DataGenerator(seed=seed, test_size=0.5, normal=True)

    # make train/test tensors with correct feature count
    def add_feature_transform(x, eval_position, FT, max_feature_dim):
        feature_dim = x.shape[-1]

        # optional preprocessing 
        if preprocess_transform is not None:
            x = FT.pfn_inference_transform(
                eval_xs=x,
                preprocess_transform=preprocess_transform,
                eval_position=eval_position,
                normalize_with_test=True,
                rescale_with_sqrt=False,
            )

        # ensure final feature count equals model expectation
        feature_dim = x.shape[-1]
        if feature_dim > max_feature_dim:
            if feature_truncation == "projection":
                x = FT.feature_sparse_projection(x=x, num_feature=feature_dim)
            else:
                x = FT.feature_subsampling(x=x, num_feature=feature_dim)
        elif feature_dim < max_feature_dim:
            x = FT.feature_padding(x=x, num_feature=feature_dim)

        assert x.shape[-1] == max_feature_dim, f"Feature mismatch: got {x.shape[-1]}, expected {max_feature_dim}"
        return x

    def make_train_test(data, FT, max_feature_dim, seq_len):
        x_train, y_train, x_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

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

        train_x, x_test = train_and_test[: len(train_x), :], train_and_test[len(train_x) :, :]
        test_in, test_out = x_test[y_test == 0], x_test[y_test == 1]

        train_x = torch.from_numpy(train_x).to(device).unsqueeze(1).float()
        test_in = torch.from_numpy(test_in).to(device).unsqueeze(0).float()
        test_out = torch.from_numpy(test_out).to(device).unsqueeze(0).float()
        return train_x, test_in, test_out

    # Ensemble evaluation across ADBench datasets
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

            for model in models_in_ensemble:
                seq_len = model["seq_len"]
                max_feature_dim = model["max_feature_dim"]
                num_R = model["num_R"]
                emsize = model["emsize"]
                nhid = model["nhid"]
                nhead = model["nhead"]
                nlayer = model["nlayer"]
                ckpt_path = model["seeds"][0]["ckpt_path"]

                print(f"\n=== Evaluating {dataset} with model expecting {max_feature_dim} features ===")

                # Model-specific FeatureTransform 
                FT = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim)

                # Build & load model
                trained_model = make_model(
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

                checkpoint = torch.load(ckpt_path, map_location=device)
                state_dict = checkpoint["state_dict"]
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                trained_model.load_state_dict(new_state_dict)

                # Prepare tensors & evaluate
                train_x, test_in, test_out = make_train_test(data, FT, max_feature_dim, seq_len)
                logits_in, time_in = get_results(trained_model, train_x, test_in, 0, save_path, "in")
                logits_la, time_la = get_results(trained_model, train_x, test_out, 1, save_path, "la")

                logits = np.concatenate([logits_in, logits_la], axis=0)
                labels = np.array([0] * len(logits_in) + [1] * len(logits_la))
                exp_logits = np.exp(logits)
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                all_scores.append(probabilities[:, 1])

            # Combine ensemble predictions
            all_scores = np.vstack(all_scores)
            np.savez(f"{save_path}/all_scores_seed{seed}.npz", all_scores=all_scores, labels=labels)
            avg_score = np.mean(all_scores, axis=0)

            preds = low_density_anomalies(-avg_score, np.sum(labels == 1))
            f1 = skm.f1_score(labels, preds)
            avg_score[np.isnan(avg_score)] = 0

            result = utils.metric(y_true=labels, y_score=avg_score)
            result = {
                seed: {
                    "aucroc": result["aucroc"],
                    "aucpr": result["aucpr"],
                    "f1": f1,
                    "time-per-inst": (time_in + time_la) / len(labels),
                    "context-len": train_x.shape[0],
                }
            }
            df = pd.DataFrame(result).T.reset_index()
            df.columns = ["seed", "aucroc", "aucpr", "f1", "time-per-inst", "context-len"]
            df.to_csv(save_path / "result.csv", mode="a", index=False)


def run_with_overrides(base_overrides, overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="configuration")
    cfg = compose(config_name="config", overrides=base_overrides + overrides)
    main(cfg)


if __name__ == "__main__":
    main()