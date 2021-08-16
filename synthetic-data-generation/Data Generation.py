import numpy as np
from random import randint
import pandas as pd

# generate expected y value based on useful attributes
# Y = XW + Bias
size = 200000 # number of data points
transaction_list = []
source_accs = []
target_accs = []
age_numbers = []
amounts = []
for i in range(size):
    source = float(randint(1000000, 9999999))
    source_accs.append(source)
    target = float(randint(1000000, 9999999))
    target_accs.append(target)
    age = float(randint(18,60))
    age_numbers.append(age)
    number = randint(1, 1000000)
    transaction_list.append(number)
    
mu, sigma = 0, 0.2 # mean and standard deviation for bias
bias = np.array(np.random.normal(mu, sigma, size)).reshape(size, 1) # Gaussian Distribution
print('the shape of bias is ', bias.shape)

source_accs = np.asarray(source_accs).reshape(size, 1)
target_accs = np.asarray(target_accs).reshape(size, 1)
age_numbers = np.asarray(age_numbers).reshape(size, 1)
# each row of X represents a transcation
# X = np.array([source_accs, target_accs, age_numbers])
X = np.concatenate((source_accs, target_accs, age_numbers), axis=1)
print('the shape of X is ', X.shape)
# weights, each element is a weight for corresponding feature in X
W = np.array([0.2, 0.25, 4]).reshape(3, 1)
print('the shape of W is ', W.shape)
# expected amounts for transcations
amounts = np.dot(X, W) + bias # dot product
print('the shape of amounts is ', amounts.shape)


# Add high_corr_feature (irerevlent feature)
bonus_points = np.asarray(age+5*np.random.poisson(2,size)).reshape(size, 1)
X = np.concatenate((source_accs, target_accs, age_numbers, bonus_points), axis=1)
print('the shape of X after adding irerevlent feature is ', X.shape)

# Make outliers for amounts
for i in range(10):
    rand_idx = randint(0, size)
    amounts[rand_idx] += randint(1000000, 9999999)

# Make missing values for X
for i in range(200):
    idx_x = np.random.randint(0, X.shape[0])
    idx_y = np.random.randint(0, X.shape[1])
    X[idx_x][idx_y] = np.nan

# Make Data Frame 
df = pd.DataFrame(transaction_list, columns=['Transaction Number'])
df['Source Account Number'] = pd.DataFrame(source_accs)
df['Target Account Number'] = pd.DataFrame(target_accs)
df['Age'] = pd.DataFrame(age_numbers)
df['Bonus'] = pd.DataFrame(bonus_points)
df['Amounts'] = pd.DataFrame(amounts)
# Add a path to the argument so the csv file can be saved
df.to_csv()