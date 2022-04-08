# DL Assignment 2 Group 6 Task 2

## Overview
Hi! You have found the code of Deep Learning Group 6 for Assignment 2, task 2.
In this assignment, we implemented NeuMF using the Microsoft recommenders package.

## Installation
To install all packages that our code is dependent on:
```bash
pip install -r /path/to/requirements.txt
```

### Python Version
We used Python 3.7, as this is the only version guaranteed to work with the Microsoft recommenders package.


### Microsoft Recommenders Note
When you have installed the correct version of Microsoft recommenders, you are not there yet.
There is still legacy code regarding the NCF part of this package in the package you download from pip.
What you now do is you go to: 
https://github.com/microsoft/recommenders/tree/main/recommenders/models/ncf and replace the files 
dataset.py and ncf_singlenode.py with the version you find on this page.
Furthermore, in order for the recommenders package to work, make sure you have installed Microsoft C++ Build Tools and 
have followed all the steps mentioned in their README file in order for the package to work as intented.

## Running

To run both models:
```bash
python main2.py
```

### Exploratory analysis
Set the variable explore_data in main2.py:
```bash
explore_data = True
```

### Grid Search
Set the variable grid_search in main2.py:
```bash
grid_search = True
```

### Settings
Furthermore,  in the top of the main2.py file you can set the following settings:
top_k (top k items to recommend), seed (random seed), n_epochs,
batch_size, learning_rate, layer_sizes_list (architecture of MLP), n_factors_list 
(which number of factors for GMF embedding).
   