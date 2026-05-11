
simon_code: has already tested

Test default code (Default GMM):

# TEST EPOCHS=10

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.batch_size=4 train.nlayer=4 train.num_R=500 train.data_method='DefaultGMM'`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.data_method='DefaultGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 default_zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.nlayer=4 train.data_method='DefaultGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=20
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.data_method='DefaultGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 default_zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.nlayer=4 train.data_method='DefaultGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=30
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.data_method='DefaultGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 default_zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.nlayer=4 train.data_method='DefaultGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


Test Modified GMM:
- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.batch_size=4 train.nlayer=4 train.num_R=500 train.data_method='GMM'`

# TEST EPOCHS=10
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.data_method='GMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.nlayer=4 train.data_method='GMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=20
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.data_method='GMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.nlayer=4 train.data_method='GMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=30
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.data_method='GMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.nlayer=4 train.data_method='GMM' test.seed=0 test.preprocess_transform='quantile'` 

# TEST EPOCHS=40
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.data_method='GMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.nlayer=4 train.data_method='GMM' test.seed=0 test.preprocess_transform='quantile'` 




Plot distribution of Modified GMM:
- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=10 prior.gmm.max_model_dim=10 prior.gmm.max_num_cluster=3 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.batch_size=4 train.nlayer=4 train.num_R=500 train.data_method='GMM'`


Plot distribution of SGMM:
- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=10 prior.gmm.max_model_dim=10 prior.gmm.max_num_cluster=3 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.batch_size=4 train.nlayer=4 train.num_R=500 train.data_method='SGMM'`