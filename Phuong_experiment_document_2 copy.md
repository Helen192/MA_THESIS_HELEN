# 1. On 16 Oct: New experiment : **epoch=20, batch_size=2, max_feature_dim=80, max_model_dim=80, max_num_cluster=10**

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 python pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`

- **zero-shot outlier detection**: (test seed=0 with quantile transformation) `CUDA_VISIBLE_DEVICES=0 python zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`


# 2. On 22.11 Nov: **epoch=20, batch_size=4, max_feature_dim=80, max_model_dim=80, max_num_cluster=10**

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 python data_prior/parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=2 python3 pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`

- **zero-shot outlier detection**: (test seed=0 with quantile transformation) `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=10 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`


# NOTE: FROM HERE: weights of clusters are not uniform

# 3. On 23.11: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=100, max_model_dim=100,prior.gmm.max_num_cluster=10

- **generate validation data**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`   ---> NOTE: It was broken after the 9-Iteration


- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`  -->NOTE: Evaluating the best model within 9 itetations


# 3. On 26.11: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70 (NOTE: By mistake, I have trained a model with max_num_cluster=70)

- **generate validation data**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`    --> This code was ran on screen: max_dim_70_batch_2

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_70_batch_2


- **pretrain**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`   --> ran This code on screen: max_dim_70_batch_4

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> ran on screen max_dim_70_batch_4


# 4. On 29.11: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=10

- **generate validation data**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`   --> This code was ran on screen: max_dim_20

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_20


TEST with batch_size=2
- **pretrain**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`   --> This code was ran on screen: max_dim_20

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_20


# 5. On 29.11: Rerunning experiments on 23.11
Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=100, max_model_dim=100,prior.gmm.max_num_cluster=10

- **generate validation data**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2`   ---> NOTE on 02.12: running on screen max_dim_100


- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=3  nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=10 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=2 test.seed=0 test.preprocess_transform='quantile'`  -->NOTE on 02.12: need to run on screen max_dim_100



# 4. On 04.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=30

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`  -> is running on screen: max_dim_20_cluster_30

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`   --> This code was ran on screen: max_dim_20_cluster_30 <Started on 04.12 at 10:50pm - Done at around 22:00 on 05.12>

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to run on screen: max_dim_20_cluster_30


# 4. On 04.12: Experiment with new code of generating validation data: **inflate_full=False, delta=0.01, max_feature_dim=30, max_model_dim=30, prior.gmm.max_num_cluster=70

- **generate validation data**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=70 prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=70 prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4`   --> This code is to run on screen: max_dim_30 (start at 10:50 on 06.10 - complete on 09.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=2 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=30 prior.gmm.max_model_dim=30 prior.gmm.max_num_cluster=70 prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to run on screen: max_dim_30



# 4. On 09.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=50, batch_size=8

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`  -> is running on screen: max_dim_20_cluster_30

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=8`   --> This code was ran on screen: max_dim_20_cluster_30 (start at 9:40am on 10.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=8 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to run on screen: max_dim_20_cluster_30


# 5. On 13.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=100, batch_size=8


- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=8`    --> This code is running on screen: max_dim_70_batch_2 (started at 13:40on 14.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=8 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: max_dim_70_batch_2



**NOTE:
In prior -> gmm.yaml, modifing 
    - nhid: 512  # number of nodes in hidden layers
    - nlayer: 4    # number of hidden layers
to have a smaller model architecture

**NOTE on 14.12.2026:
- Running zero_shot_on_adbench.py by using get_results() function of Simon is much faster, but it requires much more GPU



### Given many trained sub-models, how do I automatically discover the best ensemble composition?
    1. If you have N sub-models (say N = 10), then:
        - All possible non-empty ensembles = 2ᴺ − 1
        - For N = 10 → 1023 ensembles
    
    2. So the correct approach is:
        - **Run inference ONCE per sub-model → cache scores → search ensemble combinations using cached scores**



## Note: test model with lower 'seq_len' in train configuration: 2500 or 500 , 200

# 6. On 15.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=20, batch_size=4, seq_len=2500

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.num_workers=16 train.seq_len=2500`    --> This code is running on screen: test_seq_len (started at 13:40on 14.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.seq_len=2500 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: test_seq_len


# 6. On 18.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=20, batch_size=4, seq_len=500

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.num_workers=16 train.seq_len=500`    --> This code is running on screen: test_seq_len (started at 9:35am on 18.12)   ->>> ERROR

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.seq_len=500 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: test_seq_len


# 6. On 18.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=20, batch_size=4, seq_len=1000

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.num_workers=16 train.seq_len=1000`    --> This code is running on screen: test_seq_len_2 (started at 9:35am on 18.12 - end at around 3:15pm on 19.12)   

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=4 train.seq_len=1000 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: test_seq_len_2



# 5. On 18.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=100, batch_size=4

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=4`    --> This code is running on screen: max_dim_70_batch_2 (started at 21:50pm on 18.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: max_dim_70_batch_2



# 6. On 19.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, max_num_cluster=70, epochs=20, batch_size=8, seq_len=2500

- **pretrain**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=8 train.num_workers=25 train.seq_len=2500`    --> This code is running on screen: test_seq_len_2 (started at 13:40on 14.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=3 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=8 train.seq_len=2500 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to on screen: test_seq_len_2



#TEST SquareGMM of Simon
# 3. On 25.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=4

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4`    --> This code is to run on screen: max_dim_80 (Run at 20:00 on 26.12 - finished at around 9:00 on 27.12 - using 16cpus)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80

#TEST SquareGMM of Simon
# 3. On 27.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=8
- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=8`    --> This code is to run on screen: max_dim_80  (->STOPPED after iteration 25)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=8 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 28.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=100, max_model_dim=100, prior.gmm.max_num_cluster=100, EPOCHS=50, BATCH_SIZE=4

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=100 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=100 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4`    --> This code is to run on screen: max_dim_100 (run at 0:00 on 28.12 - done at around 10:00am)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=100 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon
# 3. On 28.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=4

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4`    --> This code is to run on screen: max_dim_100 (start at 12:40pm on 28.12 - done at 17:40pm on 28.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=4 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon
# 3. On 28.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=60, max_model_dim=60, prior.gmm.max_num_cluster=60, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=60 prior.gmm.max_model_dim=60 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=60 prior.gmm.max_model_dim=60 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (start at 17:40pm on 28.12)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=60 prior.gmm.max_model_dim=60 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 29.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (start at 9:20am on 29.12 )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 29.12: Experiment with new code of generating validation data: **inflate_full=False, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=20, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=False prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=False prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_100 (I have tried to execute this experiment few times, but it could only be run till iteration 11 and then stopped. I assume that by setting inflate_full=False, there is something wrong. So now I'll test with max_feature_dim=40 and inflate_full=True, to see if it works)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=False prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon
# 3. On 31.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=40, max_model_dim=40, prior.gmm.max_num_cluster=40, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=40 prior.gmm.max_model_dim=40 prior.gmm.max_num_cluster=40 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=40 prior.gmm.max_model_dim=40 prior.gmm.max_num_cluster=40 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (started at 8:00am - completed at around 12:00pm)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=40 prior.gmm.max_model_dim=40 prior.gmm.max_num_cluster=40 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 31.12: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=20, EPOCHS=100, BATCH_SIZE=8

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=8`    --> This code is to run on screen: max_dim_100 (started at 12:30pm on 31.12 - done before 13:00 on 1.1.2026 )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=8 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon
# 3. On 01.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=100, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (start at 13:50pm on 01.01.2026 - done before 9:00am on 02.01 )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 02.1: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=50, max_model_dim=50, prior.gmm.max_num_cluster=50, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_100

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon
# 3. On 02.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.01, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6`    --> This code is to run on screen: max_dim_80 (Run at 5:20pm on 02.01)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 03.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=100, BATCH_SIZE=6, nlayer=5

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=6 train.nlayer=5`    --> This code is to run on screen: max_dim_80 (start at 9:30 on 03.01 )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=100 train.batch_size=6 train.nlayer=5 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 03.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=5

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=5`    --> This code is to run on screen: max_dim_80 (start at 9:30 on 03.01 )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=5 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 03.01.2026: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=7

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=7`    --> This code is to run on screen: max_dim_80 (start at 11:25am on 04.01 - done at 19:30 on 04.01)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=7 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_80


#TEST SquareGMM of Simon
# 3. On 04.1: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=50, max_model_dim=50, prior.gmm.max_num_cluster=50, EPOCHS=50, BATCH_SIZE=6, nlayer=5

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=5`    --> This code is to run on screen: max_dim_100

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=50 prior.gmm.max_model_dim=50 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=5 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 05.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=20, EPOCHS=50, BATCH_SIZE=6, nlayer=5

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=5 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.01 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=5`    --> This code is to run on screen: max_dim_100 (start at 9:05am on 05.1)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=20 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=5 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100




# TEST again with GMM
# 4. On 05.1: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=20, max_model_dim=20, prior.gmm.max_num_cluster=30, EPOCHS=50, BATCH_SIZE=6, nlayer=5

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/helen_parallel_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500`  -> is running on screen: max_dim_100

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=6 train.nlayer=5`   --> This code was ran on screen: max_dim_100(started at 20:00pm on 05.1 - Done around 7:00am on 07.1)

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=20 prior.gmm.max_model_dim=20 prior.gmm.max_num_cluster=30 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=20 train.batch_size=6 train.nlayer=5 test.seed=0 test.preprocess_transform='quantile'`  --> This code is to run on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 07.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=80, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 (start at 8:10am on 07.1 - done before 4:30pm )

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=80 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 07.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=60, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 07.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=100, max_model_dim=100, prior.gmm.max_num_cluster=60, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=100 prior.gmm.max_model_dim=100 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 08.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=90, max_model_dim=90, prior.gmm.max_num_cluster=60, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=90 prior.gmm.max_model_dim=90 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=90 prior.gmm.max_model_dim=90 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=90 prior.gmm.max_model_dim=90 prior.gmm.max_num_cluster=60 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 07.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=50, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=50 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 07.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=80, max_model_dim=80, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=80 prior.gmm.max_model_dim=80 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True)  - test for the influence of delta
# 3. On 11.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.0025, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.0025 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.0025 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.0025 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 11.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100




TEST NEW CODE of simon_generator
#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 11.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=75, max_model_dim=75, prior.gmm.max_num_cluster=75, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


TEST WITH OLD CODE of simon_generator
#TEST SquareGMM of Simon (TEST for inflate_full=True)
# 3. On 11.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=75, max_model_dim=75, prior.gmm.max_num_cluster=75, EPOCHS=50, BATCH_SIZE=6, nlayer=6

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=75 prior.gmm.max_model_dim=75 prior.gmm.max_num_cluster=75 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100




#TEST SquareGMM of Simon (TEST for inflate_full=True  - Changing the architecture of Transformer: attention heads=2, layers=6)
# 3. On 30.01: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=2

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.num_R=500`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=2`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True  - Changing the architecture of Transformer: attention heads=4, layers=6, lr=0.0003, batch_size=6)
# 3. On 04.02: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=4, lr=0.0003

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=4 train.num_R=500 train.lr=0.0003`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=4 train.lr=0.0003`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=4 train.data_method='SGMM' train.lr=0.0003 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100


#TEST SquareGMM of Simon (TEST for inflate_full=True  - Changing the architecture of Transformer: attention heads=2, layers=6, lr=0.0003, batch_size=6)
# 3. On 04.02: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=2, lr=0.0003

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=4 train.num_R=500 train.lr=0.0003`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=2 train.lr=0.0003`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' train.lr=0.0003 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



#TEST SquareGMM of Simon (TEST for inflate_full=True  - Changing the architecture of Transformer: attention heads=2, layers=6, lr=0.003, batch_size=6)
# 3. On 10.02: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=2, lr=0.003

- **generate validation data**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=4 train.num_R=500 train.lr=0.003`

- **pretrain**: `CUDA_VISIBLE_DEVICES=0 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=2 train.lr=0.003`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' train.lr=0.003 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



# Test again model_24_10 with percentile in SquareGMM = 0.90:
24.10. model_24_10: SquareGMM of Simon, feat=70, inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=2,

# 3. On 10.02: Experiment with new code of generating validation data: **inflate_full=True, delta=0.005, max_feature_dim=70, max_model_dim=70, prior.gmm.max_num_cluster=70, EPOCHS=50, BATCH_SIZE=6, nlayer=6, nhead=2, lr=0.001

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=4 train.num_R=500 train.lr=0.001`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 simon_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=2 train.lr=0.001`    --> This code is to run on screen: max_dim_100 

- **zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' train.lr=0.001 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100



(BESTMODEL)dataSGMM.context5000.feat70.R500.inf-fullTrue.LTFalse.gen1tr1True.reuse100.E50.step1000.bs6.lr0.001.emb256.hdim512.nhead2.nlayer6.ndevice1.numclusters70

- **generate validation data**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 data_prior/simon_generator.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.num_R=500 train.lr=0.001`

- **pretrain**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 helen_weights_pretrain_parallel.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.data_method='SGMM' train.nlayer=6 train.nhead=2 train.lr=0.001`    --> This code is to run on screen: max_dim_100 

- **test_with_best_model: zero-shot outlier detection**: `CUDA_VISIBLE_DEVICES=1 nice -n 10 python3 zero_shot_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' train.lr=0.001 test.seed=0 test.preprocess_transform='quantile'`  --> This code was ran on screen: max_dim_100

- **p-value test**: `python3 helen_wilcoxon_on_adbench.py train.apply_linear_transform=False train.gen_one_train_one=True prior.gmm.max_feature_dim=70 prior.gmm.max_model_dim=70 prior.gmm.max_num_cluster=70 prior.gmm.inflate_full=True prior.gmm.delta=0.005 train.seed=0 train.num_R=500 train.epochs=50 train.batch_size=6 train.nlayer=6 train.nhead=2 train.data_method='SGMM' train.lr=0.001 test.seed=0 test.preprocess_transform='quantile'`
