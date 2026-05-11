import os
import re
import time
import random
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

MODEL_PATH = Path("ckpt/ensemble_5")
RESULT_SAVE_PATH = Path("results/adbench/ensemble_5")



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
        layer_src_right_att = {}
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


@hydra.main(version_base='1.3', config_path='configuration', config_name='config')
def main(cfg: DictConfig):
    # Use my extra feature transform
    #FT = FeatureTransform(cfg=cfg)
    train_cfg = cfg.train
    prior_gmm_cfg = cfg.prior.gmm
    test_cfg = cfg.test
    # train hyperparameters
    hyperparameters = train_cfg.hyperparameters
    device = train_cfg.device
    train_seed = train_cfg.seed
    # test hyperparameters
    feature_truncation = test_cfg.feature_truncation
    seed = test_cfg.seed
    preprocess_transform = test_cfg.preprocess_transform

    models_in_ensemble = extra_parameters_of_models_in_ensemble()

    setting = 'semi'
    utils = Utils()  # utils function
    utils.set_seed(seed)
    if "semi" in setting:
        datagenerator = DataGenerator(seed=seed, test_size=0.5, normal=True)  # data generator
    else:
        datagenerator = DataGenerator(seed=seed, test_size=0, normal=False)  # data generator
    utils = Utils()  # utils function
    utils.set_seed(seed)

    def add_feature_transform(x, eval_position, max_feature_dim):
        """
        Ensures the input x has exactly `max_feature_dim` features,
        regardless of what preprocess_transform does or what the dataset provides.
        """
        # Step 1: (optional) apply Hydra-configured preprocessing
        if preprocess_transform is not None:
            x = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim).pfn_inference_transform(
                eval_xs=x,
                preprocess_transform=preprocess_transform,
                eval_position=eval_position,
                normalize_with_test=True,
                rescale_with_sqrt=False,
            )
        else:
            # (Optional) some datasets may still need normalization / nothing
            pass

        # Step 2: check feature dimension AFTER preprocessing
        feature_dim = x.shape[-1]

        # Step 3: adjust feature count to match the model
        if feature_dim > max_feature_dim:
            # Too many features → reduce (truncate or subsample)
            if feature_truncation == "projection":
                x = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim).feature_sparse_projection(x=x, num_feature=max_feature_dim)
            else:
                # random subsample of features
                x = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim).feature_subsampling(x=x, num_feature=max_feature_dim)
        elif feature_dim < max_feature_dim:
            # Too few features → pad with zeros or repeat features
            x = FeatureTransform(cfg=cfg, max_feature_dim=max_feature_dim).feature_padding(x=x, num_feature=max_feature_dim)

        # Step 4: sanity check
        assert (
            x.shape[-1] == max_feature_dim
        ), f"Feature mismatch after transform: got {x.shape[-1]}, expected {max_feature_dim}"

        # Optional diagnostic print (helpful during dev)
        print(f"[Transform] Final feature dim: {x.shape[-1]} (expected {max_feature_dim})")

        return x
    
    def make_train_test(data, max_feature_dim, seq_len):
        x_train, y_train, x_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]

        if x_train.shape[0] <= seq_len - 1:
            train_x = x_train
        else:
            train_sub_indices = np.random.choice(x_train.shape[0], seq_len - 1, replace=False)
            train_x = x_train[train_sub_indices]

        train_and_test = add_feature_transform(
            x=np.concatenate([train_x, x_test], axis=0),
            eval_position=len(train_x),
            max_feature_dim=max_feature_dim,
        )

        train_x, x_test = train_and_test[: len(train_x), :], train_and_test[len(train_x) :, :]
        test_in, test_out = x_test[y_test == 0], x_test[y_test == 1]

        train_x = torch.from_numpy(train_x).to(device).unsqueeze(1).float()
        test_in = torch.from_numpy(test_in).to(device).unsqueeze(0).float()
        test_out = torch.from_numpy(test_out).to(device).unsqueeze(0).float()
        return train_x, test_in, test_out

    # Ensemble evaluation across ADBench datasets
    for dataset_list in [datagenerator.dataset_list_classical, datagenerator.dataset_list_cv,
                         datagenerator.dataset_list_nlp]:
        for dataset in dataset_list:
            '''
            la: ratio of labeled anomalies, from 0.0 to 1.0
            realistic_synthetic_mode: types of synthetic anomalies, can be local, global, dependency or cluster
            noise_type: inject data noises for testing prior robustness, can be duplicated_anomalies, irrelevant_features or label_contamination
            '''
            save_path = RESULT_SAVE_PATH / f'train_seed={train_seed}/{dataset}/{setting}'    # f'results/adbench/ensemble/train_seed={train_seed}/{dataset}/{setting}'

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # import the dataset
            datagenerator.dataset = dataset  # specify the dataset name
            data = datagenerator.generator(la=0, max_size=50000)  # maximum of 50,000 data points are available

            if "unsup" in setting:
                X = data['X_test']
                y = data['y_test']

                indices = np.arange(len(X))
                subset = np.random.choice(indices, size=len(indices), replace=True)

                data = {}
                data['X_train'] = X[subset]
                data['y_train'] = y[subset]

                data['X_test'] = X
                data['y_test'] = y

            all_scores = []   # store scores from each model
            # Loop over each model in ensemble 
            for model in models_in_ensemble:
                #Loading models and their parameters
                seq_len = model['seq_len']
                max_feature_dim= model['max_feature_dim']
                num_R = model['num_R']
                inflate_full = model['inflate_full']
                apply_linear_transform = model['apply_linear_transform']
                gen_one_train_one = model['gen_one_train_one']
                reuse_data_every_n = model['reuse_data_every_n']
                epochs = model['epochs']
                steps_per_epoch = model['steps_per_epoch']
                batch_size = model['batch_size']
                lr = model['lr']
                emsize = model['emsize']
                nhid = model['nhid']
                nhead = model['nhead']
                nlayer = model['nlayer']
                num_device = model['num_device']
                ckpt_path = model['seeds'][0]['ckpt_path']
                #seed = model['seeds'][0]['seed']
                
                print(f"\n=== Evaluating model expecting {max_feature_dim} features on dataset {dataset} ===")

                # Build Model
                # seq_len, num_features, hps, emsize, nhead, nhid, nlayers, num_class=2, model_para_dict=None
                trained_model = make_model(seq_len=seq_len, num_features=max_feature_dim, hps=hyperparameters,
                                           emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayer,
                                           num_class=2, model_para_dict={'num_R': num_R}).to(device)

                checkpoint = torch.load(ckpt_path, map_location=device)

                # Extract the state_dict
                state_dict = checkpoint['state_dict']
                # Remove the 'model.' prefix from the keys (if any) saved by pytorch_lightning
                new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
                trained_model.load_state_dict(new_state_dict)

                # adapt dataset for this model
                train_x, test_in, test_out = make_train_test(data, max_feature_dim, seq_len)

                logits_in, time_in = get_results(model=trained_model, train_x=train_x, test_x=test_in, label=0,
                                             save_path=save_path / f'seed{seed}', inst_type='in')
                logits_la, time_la = get_results(model=trained_model, train_x=train_x, test_x=test_out, label=1,
                                             save_path=save_path / f'seed{seed}', inst_type='la')

                logits = np.concatenate([logits_in, logits_la], axis=0)  # #samples, 2
                labels = np.array([0] * len(logits_in) + [1] * len(logits_la))

                exp_logits = np.exp(logits)
                probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

                # Use the probability of the positive class
                score = probabilities[:, 1]    #TODO: Phuong  ->> take average scores of models

                all_scores.append(score)

            # Ensemble combine
            # Stack all scores (num_models x num_samples)
            all_scores = np.vstack(all_scores)

            # save all_scores and labels for later reuse
            np.savez(f"{save_path}/all_scores_seed{seed}.npz", all_scores=all_scores, labels=labels)

            # Compute the ensemble average score per sample
            avg_score = np.mean(all_scores, axis=0)

            # following DTE:
            indices = np.arange(len(labels))
            p = low_density_anomalies(-avg_score, len(indices[labels == 1]))
            f1_score = skm.f1_score(labels, p)

            inds = np.where(np.isnan(avg_score))
            avg_score[inds] = 0
            result = utils.metric(y_true=labels, y_score=avg_score)

            result = {seed: {'aucroc': result['aucroc'], 'aucpr': result['aucpr'], 'f1': f1_score,
                             'time-per-inst': (time_in + time_la) / len(labels),
                             'context-len': train_x.shape[0]}}

            df = pd.DataFrame(result)

            # Transpose the DataFrame to match the desired structure
            df = df.transpose().reset_index()
            df.columns = ['seed', 'aucroc', 'aucpr', 'f1', 'time-per-inst', 'context-len']

            # Save the DataFrame to a CSV file
            df.to_csv(save_path / 'result.csv', mode='a', index=False)


def run_with_overrides(base_overrides, overrides):
    GlobalHydra.instance().clear()
    initialize(config_path="configuration")
    cfg = compose(config_name="config", overrides=base_overrides + overrides)
    main(cfg)


if __name__ == "__main__":
    main()

