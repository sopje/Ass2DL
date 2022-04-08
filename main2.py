# Imports
import pandas as pd
from data_exploration import *
from preprocessing import *
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Display only error messages
from recommenders.models.ncf.ncf_singlenode import NCF
from recommenders.models.ncf.dataset import Dataset as NCFDataset
from recommenders.utils.constants import SEED as default_seed
from recommenders.utils.timer import Timer
import itertools
import numpy as np

# Run parameters
top_k = 7  # Top k items to recommend
seed = default_seed
n_epochs = 100

# Grid search and data exploration parameter
grid_search = False
explore_data = True

# Model parameters
if grid_search:
    batch_sizes = [256, 512, 1024]
    learning_rates = [0.0001, 0.001]
    layer_sizes_list = [[32, 16, 8], [64, 32, 16], [128, 64, 32]]
    n_factors_list = [4, 8, 16]
else:
    batch_sizes = [256]
    learning_rates = [0.0001]
    layer_sizes_list = [[32, 16, 8]]
    n_factors_list = [16]

# Load data
print('--------- load data ----------')
ratings = load_data()
ratings.to_csv('ratings.csv', index=False)

# Data exploration
if explore_data:
    print('----------data exploration -------------')
    data_exploration(ratings)


# Make train-test split
print('---------- pre-processing -----------')
test, train = create_test(ratings, grid_search)
train_file = "./train.csv"
test_file = "./test.csv"
test_file_full = "./test_full_our.csv"
train.to_csv(train_file, index=False)
test.to_csv(test_file, index=False)

data = NCFDataset(train_file=train_file, test_file=test_file, seed=seed, overwrite_test_file_full=True)

print("Parameter for grid_search: ", grid_search)
print('---------- NeuMF model without pre-training -----------')
results_NeuMF_npt = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])
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

    print("Took {} seconds for training model with key: ".format(train_time.interval), key)
    train_time = train_time.interval

    k = top_k
    ndcgs = []
    hit_ratio = []

    # HR @K and NDCG@K evaluation
    for data_one_user in data.test_loader():
        user_input, item_input, labels = data_one_user
        output = model.predict(user_input, item_input, is_list=True)
        output = np.squeeze(output)
        rank = sum(output >= output[0])
        # Test example is in top k items
        if rank <= k:
            ndcgs.append(1 / np.log(rank + 1))
            hit_ratio.append(1)
        else:
            ndcgs.append(0)
            hit_ratio.append(0)

    # Calculate NDCG@K and HR@K
    ndcg_at_k = np.mean(ndcgs)
    hr_at_k = np.mean(hit_ratio)

    print("HR:\t%f" % hr_at_k)
    print("NDCG:\t%f" % ndcg_at_k)

    results_NeuMF_npt = results_NeuMF_npt.append({
        'Parameters': key,
        'Final loss': train_loss[len(train_loss)-1],
        'Train time': train_time,
        'NDCG': ndcg_at_k.item(),
        'HR': hr_at_k.item(),
    }, ignore_index=True)

    if not grid_search:
        train_loss_df = pd.DataFrame(train_loss)
        train_loss_df.to_csv("train_loss_NeuMF_npt.csv")

results_NeuMF_npt.to_csv('results_NeuMF_npt.csv')

# Pre-train GMF and MLP model and use them for NeuMF model
if not grid_search:
    print('---------- Pre-train GMF model -----------')
    models_GMF = [item for item in itertools.product(batch_sizes, learning_rates, n_factors_list)]

    for batch_size, learning_rate, n_factors in models_GMF:
        key = f"GMF: batch_size={batch_size} learning_rate={learning_rate} n_factors={n_factors}"
        print("Specific model that is now trained: ", key)
        model = NCF(
            n_users=data.n_users,
            n_items=data.n_items,
            model_type="GMF",
            n_factors=n_factors,
            layer_sizes=[32,16,8], # not used in GMF
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=10,
            seed=seed
        )

        with Timer() as train_time:
            train_loss = model.fit(data)

        print("Took {} seconds for training model with key: ".format(train_time.interval), key)

        train_loss_df = pd.DataFrame(train_loss)
        train_loss_df.to_csv("train_loss_GMF.csv")
        model.save(dir_name="PretrainModels/GMF")

    print('---------- Pre-train MLP model -----------')
    models_MLP = [item for item in itertools.product(batch_sizes, learning_rates, layer_sizes_list)]

    for batch_size, learning_rate, layer_sizes in models_MLP:
        key = f"MLP: batch_size={batch_size} learning_rate={learning_rate} layer_sizes={layer_sizes}"
        print("Specific model that is now trained: ", key)
        model = NCF(
            n_users=data.n_users,
            n_items=data.n_items,
            model_type="MLP",
            n_factors=4, #not used for MLP
            layer_sizes=layer_sizes,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=10,
            seed=seed
        )

        with Timer() as train_time:
            train_loss = model.fit(data)

        print("Took {} seconds for training model with key: ".format(train_time.interval), key)

        train_loss_df = pd.DataFrame(train_loss)
        train_loss_df.to_csv("train_loss_MLP.csv")
        model.save(dir_name="PretrainModels/MLP")

    print('---------- NeuMF model with pre-training -----------')
    models_NeuMF_pt = [item for item in itertools.product(batch_sizes, learning_rates, layer_sizes_list, n_factors_list)]
    results_NeuMF_pt = pd.DataFrame(columns=['Parameters', 'Final loss', 'Train time', 'NDCG', 'HR'])

    for batch_size, learning_rate, layer_sizes, n_factors in models_NeuMF_pt:
        key = f"NeuMF_pt: batch_size={batch_size} learning_rate={learning_rate} layer_sizes={layer_sizes} n_factors={n_factors}"
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

        model.load(gmf_dir="PretrainModels/GMF", mlp_dir="PretrainModels/MLP", alpha=0.5)

        with Timer() as train_time:
           train_loss = model.fit(data)

        print("Took {} seconds for training.".format(train_time.interval))

        k = top_k
        ndcgs = []
        hit_ratio = []

        # HR @K and NDCG@K evaluation
        for data_one_user in data.test_loader():
            user_input, item_input, labels = data_one_user
            output = model.predict(user_input, item_input, is_list=True)
            output = np.squeeze(output)
            rank = sum(output >= output[0])
            # Test example is in top k items
            if rank <= k:
                ndcgs.append(1 / np.log(rank + 1))
                hit_ratio.append(1)
            else:
                ndcgs.append(0)
                hit_ratio.append(0)

        # Calculate NDCG@K and HR@K
        ndcg_at_k = np.mean(ndcgs)
        hr_at_k = np.mean(hit_ratio)

        print("HR:\t%f" % hr_at_k)
        print("NDCG:\t%f" % ndcg_at_k)

        results_NeuMF_pt = results_NeuMF_pt.append({
            'Parameters': key,
            'Final Loss': train_loss[len(train_loss) - 1],
            'Train time': train_time.interval,
            'NDCG': ndcg_at_k.item(),
            'HR': hr_at_k.item(),
        }, ignore_index=True)

        train_loss_df = pd.DataFrame(train_loss)
        train_loss_df.to_csv('train_loss_NeuMF_pt.csv')

    results_NeuMF_pt.to_csv('results_NeuMF_pt.csv')
