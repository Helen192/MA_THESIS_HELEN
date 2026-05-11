# FoMo-0D
Official implementation of FoMo-0D, a foundation model for zero-shot outlier detection.

## Setup Instructions
Please follow the instructions in [DTE](https://github.com/vicliv/DTE/tree/main) (https://arxiv.org/abs/2305.18593) and [PFN](https://github.com/automl/PFNs) (https://arxiv.org/abs/2112.10510)

## Pretrain FoMo-0D

To pretrain our model, the parameters are:

`prior.gmm.max_feature_dim` and `prior.gmm.max_model_dim`: input feature of the model

`train.reuse_data_every_n`: reuse at epoch-level (how many epochs to reuse)

`train.apply_linear_transform`: whether to apply linear transformation (LT) on the synthetic data

`train.gen_one_train_one`: whether to generate data on-the-fly, if `True`, ignores `train.reuse_data_every_n`

(below we pretrain FoMo-0D with `input_feature=100`, `generate-on-the-fly`, and `no-LT`)
- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500`
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 python pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500`

To use our **pretrained FoMo-0D model** (supporting input feature=100), please unzip `ckpt.zip`.

## Test on [ADBench](https://arxiv.org/abs/2206.09426) (Zero-Shot)
We follow [DTE](https://arxiv.org/abs/2305.18593) and run the seeds={0,1,2,3,4} for the zero-shot experiments on ADBench. 
- **zero-shot outlier detection**: (test seed=0 with quantile transformation) `CUDA_VISIBLE_DEVICES=0 python zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.seed=0 test.preprocess_transform='quantile'`
- **sort results over 5 seeds**: `python sort_zero_shot_result.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.preprocess_transform='quantile'`
- **p-value test**: `python wilcoxon_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 train.seed=0 train.num_R=500 test.preprocess_transform='quantile'`
