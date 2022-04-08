def add_negatives_test(test_data, train):
    num_negatives = 100
    all_movies = train['itemID'].unique()  # get all different movie Ids from the whole data set
    user_movie_set = set(zip(test_data['userID'], test_data['itemID']))  # make set with user-movie pairs

    # Create user, item and labels array
    users = list()
    movies = list()
    labels = list()
    test_batch = list()

    for user, movie in tqdm(user_movie_set):    # met tqdm krijg je zo'n cool progress balkje
        users.append(user)
        movies.append(movie)
        labels.append(1)  # because user interacted with this model
        test_batch.append(user-1)

        # Add negative examples
        for _ in range(num_negatives):
            negative_movie = np.random.choice(all_movies)
            while (user, negative_movie) in user_movie_set:        # if user already interacted with picked movie
                negative_movie = np.random.choice(all_movies)   # pick other movie
            users.append(user)
            test_batch.append(user-1)
            movies.append(negative_movie)
            labels.append(0)        # because negative example

    # test_full = pd.DataFrame(list(zip(users, movies, labels)), columns=['userID', 'itemID', 'rating'])
    test_full = pd.DataFrame(list(zip(users, movies, labels, test_batch)), columns=['userID', 'itemID', 'rating',
                                                                                    'test_batch'])
    test_full = test_full.sort_values(by='userID')

    return test_full