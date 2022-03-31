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
grid_search = True

# Model parameters
if grid_search:
    # batch_sizes = [256, 512, 1024]
    batch_sizes = [256]
    # learning_rates = [0.0001, 0.001]
    learning_rates = [0.001]
    # layer_sizes_list = [[32, 16, 8], [64, 32, 16], [128, 64, 32]] #TODO: how do n_factors and n_factors_list combine?
    layer_sizes_list = [[32, 16, 8]] #TODO: how do n_factors and n_factors_list combine?
    # n_factors_list = [4, 8, 16]
    n_factors_list = [4, 8]
else:
    batch_sizes = [256]
    learning_rates = [1e-3]
    layer_sizes_list = [[32, 16, 8]]
    n_factors_list = [16]

seed = default_seed
n_epochs = 1

print('--------- load data ----------')
ratings = load_data()
ratings.to_csv('ratings.csv', index=False)
# %%
# print('---------- data exploration -------------')
#data_exploration(ratings)

# %% Make train-test split
print('---------- pre-processing -----------')
# train, test = python_chrono_split(ratings, 0.75)
# testRandom, trainRandom = create_test(train, grid_search)
test, train = create_test(ratings, grid_search)
train_file = "./train.csv"
test_file = "./test.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)
train_fileRANDOM = "./trainRANDOM.csv"
test_fileRANDOM = "./testRANDOM.csv"
# trainRandom.to_csv(train_fileRANDOM, index=False)
# testRandom.to_csv(test_fileRANDOM, index=False)

data = NCFDataset(train_file=train_file, test_file=test_file, seed=seed, overwrite_test_file_full=False) #TODO: what does this overwrite do?'


#TODO: all models grid search or not? --> maye only NeuMF

#TODO: or do first grid search of one model? so first NeuMF without training..
print('---------- NeuMF model without pre-training -----------')
grid_search_results_NeuMF_npt = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])
models_NeuMF_npt = [item for item in itertools.product(batch_sizes, learning_rates, layer_sizes_list, n_factors_list)]

for batch_size, learning_rate, layer_sizes, n_factors in models_NeuMF_npt:
    key = f"NeuMF_npt: batch_size={batch_size} learning_rate={learning_rate} layer_sizes={layer_sizes} n_factors={n_factors}"
    print("Specific model that is now trained: ", key)
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
        train_loss = model.fit(data)

    print("Took {} seconds for training model with key: ".format(train_time.interval), key) #TODO: how to save this loss?

    k = top_k
    ndcgs = []
    hit_ratio = []

    # HR @K and NDCG@K evaluation
    STOP = 0
    for b in data.test_loader():
        user_input, item_input, labels = b #TODO: check difference this row with leave one out and random
        # print("user input: ", user_input)
        # print("item input: ", item_input)
        # print("labels ", labels)
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
        'Final Loss': train_loss[len(train_loss)-1],
        'Train time': train_time.interval,
        'NDCG': eval_ndcg,
        'HR': eval_hr,
    }, ignore_index=True)  # TODO: how to get loss? #TODO: change NDCG and HR to actual values if they work

grid_search_results_NeuMF_npt.to_csv('gridsearch_NeuMF_npt.csv')

# print('---------- GMF model -----------')
# grid_search_results_GMF = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])
# models_GMF = [item for item in itertools.product(batch_sizes, learning_rates, n_factors_list)]
#
# for batch_size, learning_rate, n_factors in models_NeuMF_npt:
#     key = f"GMF: batch_size={batch_size} learning_rate={learning_rate} n_factors={n_factors}"
#     print("Specific model that is now trained: ", key)
#     model = NCF(
#         n_users=data.n_users,
#         n_items=data.n_items,
#         model_type="GMF",
#         n_factors=n_factors,
#         layer_sizes=[32,16,8], # not used in GMF
#         n_epochs=n_epochs,
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#         verbose=10,
#         seed=seed
#     )
#
#     with Timer() as train_time:
#         train_loss = model.fit(data)
#
#     print("Took {} seconds for training model with key: ".format(train_time.interval), key)
#     # TODO: save HR, NDCG to determine best model
#
#     if not grid_search:
#         model.save(dir_name=".pretrain/GMF")
#
# grid_search_results_GMF.to_csv('gridsearch_GMF.csv')
#
# print('---------- MLP model -----------')
# grid_search_results_NeuMF_npt = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])
# models_MLP = [item for item in itertools.product(batch_sizes, learning_rates, layer_sizes_list)]
#
# for batch_size, learning_rate, layer_sizes, n_factors in models_MLP:
#     key = f"MLP: batch_size={batch_size} learning_rate={learning_rate} layer_sizes={layer_sizes}"
#     print("Specific model that is now trained: ", key)
#     model = NCF(
#         n_users=data.n_users,
#         n_items=data.n_items,
#         model_type="MLP",
#         n_factors=4, #not used for MLP
#         layer_sizes=layer_sizes,
#         n_epochs=n_epochs,
#         batch_size=batch_size,
#         learning_rate=learning_rate,
#         verbose=10,
#         seed=seed
#     )
#
#     with Timer() as train_time:
#         train_loss = model.fit(data)
#
#     if not grid_search:
#         model.save(dir_name=".pretrain/MLP")
#
#     print("Took {} seconds for training model with key: .".format(train_time.interval))
#
#     #TODO: save HR, NDCG to determine best model
#
# grid_search_results_MLP.to_csv('gridsearch_MLP.csv')


#TODO: NeuMF pt gridsearch
#TODO: save GMF MLP csv shizzle

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
#    train_loss = model.fit(data)
#
# print("Took {} seconds for training.".format(train_time.interval))

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
# eval_map2 = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_ndcg2 = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_precision2 = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
# eval_recall2 = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
#
# print("MAP:\t%f" % eval_map2,
#       "NDCG:\t%f" % eval_ndcg2,
#       "Precision@K:\t%f" % eval_precision2,
#       "Recall@K:\t%f" % eval_recall2, sep='\n')

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
