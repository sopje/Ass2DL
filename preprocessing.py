# imports
import pandas as pd
import numpy as np
from tqdm import tqdm


def train_test_split(ratings):
    # order per user the ratings s.t. timestamp is descending, then newest_rated = 1 for the newest ranked
    ratings['newest_rated'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
    # @ sophie: deze regel geeft een warning maar dat is een vals positief in pandas, dus maakt niet uit

    # make train test split
    train = ratings[ratings['newest_rated'] != 1]
    test = ratings[ratings['newest_rated'] == 1]

    # drop time step column because we no longer need it
    train = train[['userId', 'movieId', 'rating']]
    test = test[['userId', 'movieId', 'rating']]

    return train, test


def transform_to_implicit(train):
    train.loc[:, 'rating'] = 1  # because these are the movies a user interacted with
    return train


def add_negatives(train, ratings):
    num_negatives = 5  # TODO: now ratio 5:1, but maybe adjust
    all_movies = ratings['movieId'].unique()  # get all different movie Ids from the whole data set
    user_movie_set = set(zip(train['userId'], train['movieId']))  # make set with user-movie pairs

    # create user, item and labels array
    users = list()
    movies = list()
    labels = list()

    #TODO: kan dit efficienter? Want het duurt lang
    for user, movie in tqdm(user_movie_set):    # met tqdm krijg je zo'n cool progress balkje
        users.append(user)
        movies.append(movie)
        labels.append(1)  # because user interacted with this model


        # add negative examples
        for _ in range(num_negatives):
            negative_movie = np.random.choice(all_movies)
            while (user, negative_movie) in user_movie_set:        # if user already interacted with picked movie
                negative_movie = np.random.choice(all_movies)   # pick other movie
            users.append(user)
            movies.append(negative_movie)
            labels.append(0)        # because negative example

    return users, movies, labels


def reindex_ID(data):
    unique_users = data['userID'].unique()
    user_to_id = {user: id for id, user in enumerate(unique_users)}

    unique_movies = data['movieID'].unique()
    movie_to_id = {movie: id for id, movie in enumerate(unique_movies)}

    data['userID'] = data['userID'].apply(lambda x: user_to_id[x])
    data['movieID'] = data['movieID'].apply(lambda x: movie_to_id[x])

    return data


# - one-hot-encode user id's?
# - transform to implicit data --> TODO: dit is gedaan toch?
