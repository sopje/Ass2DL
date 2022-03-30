import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # only show error messages

from data_exploration import *
from preprocessing import *
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.datasets.python_splitters import \
    python_chrono_split  # TODO: why do we need the chrono split and what does it do? Snap het nog niet, te moe
        #TODO: what is does it splits data chronocillay, remaining 75 % train and 25% test with 25% test are the latest itemm for one user
from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k,
                                                       precision_at_k,
                                                       recall_at_k, get_top_k_items)
from recommenders.utils.constants import SEED as default_seed
from recommenders.utils.timer import Timer
import itertools


# top k items to recommend
top_k = 10

# Grid search parameters
grid_search = False

# Model parameters
if grid_search:
    batch_sizes = [128, 256, 512, 1024]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005]
    layer_sizes_list = [[32, 16, 8]] #TODO: how do n_factors and n_factors_list combine?
    n_factors_list = [16]
else:
    batch_sizes = [10000, 256]
    learning_rates = [1e-3]
    layer_sizes_list = [[32, 16, 8]]
    n_factors_list = [16]

seed = default_seed
n_epochs = 2

print('--------- load data ----------')
ratings = load_data()

# %%
# print('---------- data exploration -------------')
#data_exploration(ratings)
test, train = create_test(ratings, grid_search)
print(test)
print(train)

# %% Make train-test split
print('---------- pre-processing -----------')
train_file = "./train.csv"
test_file = "./test.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

data = NCFDataset(train_file=train_file, test_file=test_file, seed=seed, overwrite_test_file_full=True) #TODO: what does this overwrite do?'


#TODO: all models grid search or not?
#TODO: or do first grid search of one model? so first NeuMF without training..
grid_search_results_NeuMF_npt = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])

models_NeuMF_npt = [item for item in itertools.product(batch_sizes, learning_rates, layer_sizes_list, n_factors_list)]

print('---------- NeuMF model without pre-training -----------')
for batch_size, learning_rate, layer_sizes, n_factors in models_NeuMF_npt:
    key = f"NeuMF_npt: batch_size={batch_size} learning_rate={learning_rate} layer_sizes={layer_sizes} n_factors={n_factors}"
    print("Specific model: ", key)
    model = NCF(
        n_users=data.n_users,
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=n_factors,
        layer_sizes=layer_sizes,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=10,
        seed=seed
    )

    with Timer() as train_time:
        model.fit(data)

    print("Took {} seconds for training model with key: ".format(train_time.interval), key) #TODO: how to save this loss?

    # predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
    #                for (_, row) in test.iterrows()]  #TODO: you use test here!!
    #
    #
    # predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
    # predictions.head()
    #
    # with Timer() as test_time:
    #     users, items, preds = [], [], []
    #     item = list(train.itemID.unique())
    #     for user in train.userID.unique():
    #         user = [user] * len(item)
    #         users.extend(user)
    #         items.extend(item)
    #         preds.extend(list(model.predict(user, item, is_list=True)))
    #
    #     all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})
    #
    #     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    #     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
    #
    # print("Took {} seconds for prediction.".format(test_time.interval))
    #
    # eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    # eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    # eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
    # eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

    # TODO: these are all metrics that are not in the paper......
    #
    # print("MAP:\t%f" % eval_map,
    #       "NDCG:\t%f" % eval_ndcg,
    #       "Precision@K:\t%f" % eval_precision,
    #       "Recall@K:\t%f" % eval_recall, sep='\n')

    k = top_k

    ndcgs = []
    hit_ratio = []

    # #TODO: this evaluation for test does not work
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

    grid_search_results_NeuMF_npt = grid_search_results_NeuMF_npt.append({
        'Parameters': key,
        'Final Loss': 10,
        'Train time': train_time.interval,
        'NDCG': 10,
        'HR': 10,
    }, ignore_index=True)  # TODO: how to get loss? #TODO: change NDCG and HR to actual values if they work


grid_search_results_NeuMF_npt.to_csv('gridsearch_NeuMF_npt.csv')

# # TODO: which of these parameters does GMF actually use??
# model = NCF(
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="GMF",
#     n_factors=16,
#     layer_sizes=[32, 16, 8],
#     n_epochs=n_epochs,
#     batch_size=batch_size,
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
# # TODO: which of these parameters does the MLP actually use??
# model = NCF(
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="MLP",
#     n_factors=16,
#     layer_sizes=[32, 16, 8],
#     n_epochs=n_epochs,
#     batch_size=batch_sizes,
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
# model = NCF(
#     n_users=data.n_users,
#     n_items=data.n_items,
#     model_type="NeuMF",
#     n_factors=16,
#     layer_sizes=[32, 16, 8],
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
# # with Timer() as test_time:
# #     users, items, preds = [], [], []
# #     item = list(train.itemID.unique())
# #     for user in train.userID.unique():
# #         user = [user] * len(item)
# #         users.extend(user)
# #         items.extend(item)
# #         preds.extend(list(model.predict(user, item, is_list=True)))
# #
# #     all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})
# #
# #     merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
# #     all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
# #
# # print("Took {} seconds for prediction.".format(test_time.interval))
# #
# # eval_map2 = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# # eval_ndcg2 = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# # eval_precision2 = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# # eval_recall2 = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# #
# # print("MAP:\t%f" % eval_map2,
# #       "NDCG:\t%f" % eval_ndcg2,
# #       "Precision@K:\t%f" % eval_precision2,
# #       "Recall@K:\t%f" % eval_recall2, sep='\n')
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
