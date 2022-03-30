import sys
import os
import shutil
import papermill as pm
import scrapbook as sb
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # only show error messages
from data_exploration import *
from preprocessing import *
from recommenders.utils.timer import Timer
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k,
                                                     recall_at_k, get_top_k_items)
from recommenders.utils.constants import SEED as DEFAULT_SEED


print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))
print("TEST:", tf.test.is_built_with_cuda())
print("TEST:", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print("TEST Physical GPU:", tf.config.list_physical_devices('GPU'))

#%%

# top k items to recommend
TOP_K = 10

# Select MovieLens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '100k'

# Model parameters
EPOCHS = 50
BATCH_SIZE = 10000

SEED = DEFAULT_SEED  # Set None for non-deterministic results


df = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"]
)

df.head()
print(df)
df = load_data()
train, test = python_chrono_split(df, 0.75)

#%% md

#%%

test = test[test["userID"].isin(train["userID"].unique())]
test = test[test["itemID"].isin(train["itemID"].unique())]

print(test)
#%%

leave_one_out_test = test.groupby("userID").last().reset_index()
print(leave_one_out_test)

#%% md

#%%

train_file = "./train.csv"
test_file = "./test.csv"
leave_one_out_test_file = "./leave_one_out_test.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)
leave_one_out_test.to_csv(leave_one_out_test_file, index=False)



data = NCFDataset(train_file=train_file, test_file=leave_one_out_test_file, seed=SEED, overwrite_test_file_full=True)
print(data)

model = NCF(
    n_users=data.n_users,
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=16,
    layer_sizes=[32,16,8],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=10,
    seed=SEED
)



with Timer() as train_time:
    model.fit(data)

print("Took {} seconds for training.".format(train_time.interval))



# predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
#                for (_, row) in test.iterrows()]
#
#
# predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
# predictions.head()
#
#
# with Timer() as test_time:
#
#     users, items, preds = [], [], []
#     item = list(train.itemID.unique())
#     for user in train.userID.unique():
#         user = [user] * len(item)
#         users.extend(user)
#         items.extend(item)
#         preds.extend(list(model.predict(user, item, is_list=True)))
#
#     all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})
#
#     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
#     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
#
# print("Took {} seconds for prediction.".format(test_time.interval))
#
#
#
# eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print("MAP:\t%f" % eval_map,
#       "NDCG:\t%f" % eval_ndcg,
#       "Precision@K:\t%f" % eval_precision,
#       "Recall@K:\t%f" % eval_recall, sep='\n')


k = TOP_K

ndcgs = []
hit_ratio = []

for b in data.test_loader():
    user_input, item_input, labels = b
    output = model.predict(user_input, item_input, is_list=True)

    output = np.squeeze(output)
    rank = sum(output >= output[0])
    if rank <= k:
        ndcgs.append(1 / np.log(rank + 1))
        hit_ratio.append(1)
    else:
        ndcgs.append(0)
        hit_ratio.append(0)

eval_ndcg = np.mean(ndcgs)
eval_hr = np.mean(hit_ratio)

print("HR:\t%f" % eval_hr)
print("NDCG:\t%f" % eval_ndcg)

# model = NCF (
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="GMF",
#     n_factors=16,
#     layer_sizes=[32,16,8],
#     n_epochs=200,
#     batch_size=BATCH_SIZE,
#     learning_rate=1e-3,
#     verbose=10,
#     seed=SEED
# )
#
# with Timer() as train_time:
#     model.fit(data)
#
# print("Took {} seconds for training.".format(train_time.interval))
#
# model.save(dir_name=".pretrain/GMF")
#
# model = NCF (
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="MLP",
#     n_factors=16,
#     layer_sizes=[32,16,8],
#     n_epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     learning_rate=1e-3,
#     verbose=10,
#     seed=SEED
# )
#
# with Timer() as train_time:
#     model.fit(data)
#
# print("Took {} seconds for training.".format(train_time.interval))
#
# model.save(dir_name=".pretrain/MLP")
#
#
# model = NCF (
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="NeuMF",
#     n_factors=16,
#     layer_sizes=[32,16,8],
#     n_epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     learning_rate=1e-3,
#     verbose=10,
#     seed=SEED
# )
#
# model.load(gmf_dir=".pretrain/GMF", mlp_dir=".pretrain/MLP", alpha=0.5)
#
# with Timer() as train_time:
#     model.fit(data)
#
# print("Took {} seconds for training.".format(train_time.interval))
#
# with Timer() as test_time:
#
#     users, items, preds = [], [], []
#     item = list(train.itemID.unique())
#     for user in train.userID.unique():
#         user = [user] * len(item)
#         users.extend(user)
#         items.extend(item)
#         preds.extend(list(model.predict(user, item, is_list=True)))
#
#     all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})
#
#     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
#     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
#
# print("Took {} seconds for prediction.".format(test_time.interval))
#
# eval_map2 = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg2 = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision2 = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall2 = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print("MAP:\t%f" % eval_map2,
#       "NDCG:\t%f" % eval_ndcg2,
#       "Precision@K:\t%f" % eval_precision2,
#       "Recall@K:\t%f" % eval_recall2, sep='\n')
#
# # # Record results with papermill for tests
# # sb.glue("map", eval_map)
# # sb.glue("ndcg", eval_ndcg)
# # sb.glue("precision", eval_precision)
# # sb.glue("recall", eval_recall)
# # sb.glue("map2", eval_map2)
# # sb.glue("ndcg2", eval_ndcg2)
# # sb.glue("precision2", eval_precision2)
# # sb.glue("recall2", eval_recall2)
#
# for b in data.test_loader():
#     user_input, item_input, labels = b
#     output = model.predict(user_input, item_input, is_list=True)
#
#     output = np.squeeze(output)
#     rank = sum(output >= output[0])
#     if rank <= k:
#         ndcgs.append(1 / np.log(rank + 1))
#         hit_ratio.append(1)
#     else:
#         ndcgs.append(0)
#         hit_ratio.append(0)
#
# eval_ndcg = np.mean(ndcgs)
# eval_hr = np.mean(hit_ratio)
#
# print("HR:\t%f" % eval_hr)
# print("NDCG:\t%f" % eval_ndcg)