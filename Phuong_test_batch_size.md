#TEST SquareGMM of Simon
# 3. On 25.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=4

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4`    --> This code is to run on screen: max_dim_80 (Run at 20:00 on 26.12 - finished at around 9:00 on 27.12 - using 16cpus)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

# 3. On 02.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (Run at 5:20pm on 02.01)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 27.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=8
- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=8`    --> This code is to run on screen: max_dim_80  (->STOPPED after iteration 25)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=8 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80



# 3. On 02.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=2

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2`    

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'` 


# Next TEST
# TEST EPOCHS=50, batch_size=4
- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`


# BATCH_SIZE=2
- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`

# BATCH_SIZE=6
- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=4 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`


# Max_feat_dim=70: model_24: SquareGMM of Simon, feat=70, inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=2
# BATCH_SIZE=2
- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2 train.data_method='BATCHSGMM' train.nlayer=4`    --> This code is to run on screen: max_dim_80 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=2 train.nlayer=4 train.data_method='BATCHSGMM' test.seed=0 test.preprocess_transform='quantile'`



# TEST nhead= 2, nlayer=6
# 3. On 02.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2`    --> This code is to run on screen: max_dim_80 (Run at 5:20pm on 02.01)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile' train.nlayer=6 train.nhead=2`  --> This code was ran on screen: max_dim_80