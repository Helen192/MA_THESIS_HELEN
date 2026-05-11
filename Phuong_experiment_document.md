# 1. On 16 Oct: New experiment : **epoch=20, batch_size=2, max_feature_dim=80, max_model_dim=80, max_num_cluster=10**

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 python pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`

- **zero-shot outlier detection**: (test seed=0 with quantile transformation) `CUDA_VISIBLE_DEVICES=0 python zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`


# 2. On 07.11 Nov: **epoch=20, batch_size=4, max_feature_dim=80, max_model_dim=80, max_num_cluster=10**

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=2 python pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`

- **zero-shot outlier detection**: (test seed=0 with quantile transformation) `CUDA_VISIBLE_DEVICES=0 python zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`



**NOTE:
In prior -> gmm.yaml, modifing 
    - nhid: 512  # number of nodes in hidden layers
    - nlayer: 4    # number of hidden layers
to have a smaller model architecture