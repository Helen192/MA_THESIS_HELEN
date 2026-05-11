### TEST THE INFLUENCE OF LEARNING RATE lr =   0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01
#TEST SquareGMM of Simon (TEST for inflate_full=True  - Changing the architecture of Transformer: attention heads=2, layers=4, lr=0.0001, batch_size=4)
# Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=30, max_model_dim=30, prior.gmm.max_num_cluster=30, EPOCHS=50, BATCH_SIZE=4, nlayer=4, nhead=2, lr=...

# lr = 0.0001

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.num_R=500 train.lr=0.0001`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.0001`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.0001 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100

# lr = 0.0003

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.0003`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.0003 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100

# lr = 0.0006

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.0006`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.0006 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


# lr = 0.001
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.001`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.001 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


# lr = 0.003
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.003`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.003 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100

# lr = 0.006
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.006`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.006 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100

# lr = 0.01
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4 train.nhead=2 train.lr=0.01`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.nhead=2 train.data_method='SGMM' train.lr=0.01 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100