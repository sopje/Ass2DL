import pandas as pd
import numpy as np


def load_data():
    with open('data/ml-100k/u.data') as input_file:
        lines = input_file.readlines()
        newLines = []
        for line in lines:
            newLine = line.strip().split()
            newLines.append(newLine)

    ratings = pd.DataFrame(newLines)
    ratings.columns = ["userID", "itemID", "rating", "timestamp"]
    ratings['userID'] = ratings.userID.astype(int)
    ratings['itemID'] = ratings.itemID.astype(int)
    ratings['rating'] = ratings.rating.astype(int)
    ratings['timestamp'] = ratings.timestamp.astype(int)
    return ratings


def create_small_practice_dataset(ratings):
    selected_ids = np.random.choice(ratings['userId'].unique(), size=int(len(ratings['userId'].unique()) * 0.002))
    ratings_small = ratings.loc[ratings['userId'].isin(selected_ids)]
    # print(ratings_small)
    return ratings_small


def data_exploration(ratings_small):
    print('data selected from ', len(ratings_small['userId'].unique()), ' unique users')
    print('there are ', len(ratings_small), ' ratings given')
    print('there are ', len(ratings_small['movieId'].unique()), ' different movies rated')

    id_number_ratings = ratings_small['userId'].value_counts(ascending=True)
    id_number_ratings = id_number_ratings.to_frame()
    print(id_number_ratings)
    print('Max movies rated ', id_number_ratings['userId'].max())
    print('Min movies rated ', id_number_ratings['userId'].min())
    print(id_number_ratings.columns)

    return
