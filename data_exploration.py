import matplotlib.pyplot as plt

# Data exploration function for providing information about the users and the movies
import pandas as pd
def data_exploration(ratings):
    print('Data selected from ', len(ratings['userID'].unique()), ' unique users')
    print('there are ', len(ratings), ' ratings given')
    print('there are ', len(ratings['itemID'].unique()), ' different movies rated')

    id_number_ratings = ratings['userID'].value_counts(ascending=True)
    id_number_ratings = id_number_ratings.to_frame()
    print(id_number_ratings)
    print('Max movies rated ', id_number_ratings['userID'].max())
    print('Min movies rated ', id_number_ratings['userID'].min())
    print(id_number_ratings.columns)
    return


def make_distribution_plot(ratings):
    count_users = ratings['userID'].value_counts(sort=False, ascending=True)
    plt.figure(figsize=(13,8))
    count_users.plot.hist(bins=50,color='lightseagreen')
    plt.xlabel('Number of rated movies', fontsize=24)
    plt.ylabel('Frequency', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig('plot_distribution.png')
    plt.show()
    return
