# Imports
import pandas as pd

# Load data from 100k dataset and put into pandas dataframe
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

# Create test and train data from ratings
def create_test(ratings, grid_search):
    # Make train test split
    if grid_search:
        test = ratings.groupby("userID").sample(random_state=42).reset_index()  # TODO: set randomstate
        train = pd.merge(ratings, test, indicator=True, how='outer').query('_merge=="left_only"').drop(
            '_merge', axis=1)                                                   # delete test samples out training set
        train = train.drop(['index'], axis=1)
        test = test.drop(['index'], axis=1)
        test = test[test["userID"].isin(train["userID"].unique())]
        test = test[test["itemID"].isin(train["itemID"].unique())]
    else:
        test = ratings.groupby("userID").last().reset_index()
        train = pd.merge(ratings, test, indicator=True, how='outer').query('_merge=="left_only"').drop(
            '_merge', axis=1)
        test = test[test["userID"].isin(train["userID"].unique())]
        test = test[test["itemID"].isin(train["itemID"].unique())]

    # Sort train and test data by user id
    train = train.sort_values(by='userID')
    test = test.sort_values(by='userID')

    return test, train
