
# As per our dicussion, please do the following parts in Scala


# identify outliers by using Interquartile range (IQR)
sorted_amounts = sorted(amounts)
half = int(size/2)
# First quartile (Q1)
Q1 = np.median(sorted_amounts[:half])
Q3 = np.median(sorted_amounts[half:])
# Interquaritle range (IQR)
IQR = Q3 - Q1

outliers_row = []
# remove outliers from X and amounts, adjust the size of bias accordingly
threhold = Q3 + 0.8*IQR # lower the coefficient of IQR if do not remove outliers completely
for i in range(size):
    if amounts[i] > threhold:
        outliers_row.append(i)
        
X = np.delete(X, outliers_row, 0)
amounts = np.delete(amounts, outliers_row, 0)
bias = np.delete(bias, outliers_row, 0)


# Replace NaN with reasonable values
for j in range(X.shape[1]):
    min_val = np.nanmin(X[:,j])
    max_val = np.nanmax(X[:,j])
    for i in range(X.shape[0]):
        if np.isnan(X[i][j]):
            counter += 1
            X[i][j] = float(randint(min_val, max_val))

amounts = np.dot(X, W) + bias # dot product


# split training set and test set (0.8, 0.2)
x_train, x_test, y_train, y_test = train_test_split(X, amounts, train_size = 0.8)

# fit the model from training set
reg = LinearRegression().fit(x_train, y_train)
# predict on test set using the model
y_pred = reg.predict(x_test)
# calculate error
error = mean_squared_error(y_test, y_pred)
print('the accuracy from prediction of linear regression is ', (1-error))

X = StandardScaler().fit_transform(X)
# now evalute after dimension reduction
pca = PCA(n_components=4)
# now fit the model and evalaute again
X_reduced = pca.fit_transform(X) # reduce-dim of input X
x_train, x_test, y_train, y_test = train_test_split(X_reduced, amounts, train_size = 0.8)
x_train_reduced = pca.fit_transform(x_train)
x_test_reduced = pca.fit_transform(x_test)
reg = LinearRegression().fit(x_train, y_train)
y_pred = reg.predict(x_test)
error = mean_squared_error(y_test, y_pred)
print('the accuracy from prediction after dimension reduction is ', (1-error))

# After the above steps, if possible, please try other algorithms to compute the prediction accuracy
# Because we pretend that we do not know the pattern of the datset, so we need to try different
# algorithms to see which one gives us higher accuracy, only through this way,
# we can finally conclude that our dataset is in which pattern
