# TEST EPOCHS=10
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=20
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=30
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=40
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=50
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=60
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=60 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=60 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=70
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=70 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=70 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80
  

# TEST EPOCHS=80
- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=80 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=80 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=90
- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=90 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=90 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=100
- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST Epochs again, but remove 'np.random.seed(epoch_id + step + os.getpid())' in the simon_generator.py - function process_one_dataset

#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 17.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=10, BATCH_SIZE=4, nlayer=4

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.epochs=10 train.batch_size=4 train.nlayer=4 train.num_R=500`


# TEST EPOCHS=10
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=10 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=20
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=30
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=30 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=40
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=40 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# TEST EPOCHS=50
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


# TEST EPOCHS=60
- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=60 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=60 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


