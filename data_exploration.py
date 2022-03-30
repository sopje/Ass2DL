import pandas as pd
import numpy as np
from recommenders.datasets.python_splitters import python_chrono_split

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
