# example python with jupyther

I assume you have python and jupyter Labs installed.
If not check [install the tools](./../installTools.md) for details on how to do it.

We will try to use some testdata in order implement a house pricing prediction.
We will use python and the SciKit lib for it. As algorithm we will use Gradient Boosting for regression. So we will use a regression algorithm which is optimized using the ensemble method of Gradient boosting.

1. Open a new notebook in jupyter

2. For our example we are using a free test data file from kaggle.
Download it from https://www.kaggle.com/anthonypino/melbourne-housing-market
and store it in the folder with your notebook under "Melbourne_housing_FULL.csv".

3. Import the libraries we need in out jupyter notebook

```
# We will Import the libraries we need
# pandas is a useful helper lib
import pandas as pd
# from sklearn we need the method to split test data from training data
from sklearn.model_selection import train_test_split
# we want to use the GradientBoostingRegressor ensemble method
from sklearn.ensemble import GradientBoostingRegressor
# metrics will help us to calculate the prediction error
from sklearn.metrics import mean_absolute_error
```

4. Lets import the data

```
# Lets import the data
full_dataset = pd.read_csv("./Melbourne_housing_FULL.csv")
```

5. Data Scrubbing

```
# Data Scrubbing
# let's remove not needed colums
# and delete all uncomplete rows
# then we will use One-hot encoding
# and define the price column as the value to predict

del full_dataset['Address']
del full_dataset['Method']
del full_dataset['SellerG']
del full_dataset['Date']
del full_dataset['Postcode']
del full_dataset['Lattitude']
del full_dataset['Longtitude']
del full_dataset['Regionname']
del full_dataset['Propertycount']

# Remove missing values, see https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html for details
full_dataset.dropna(axis = 0, how = 'any', thresh = None, subset = None, inplace = True)

# Convert categorical variable into indicator variables https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
full_dataset = pd.get_dummies(full_dataset, columns = ['Suburb', 'CouncilArea', 'Type'])

# as input we use everything apart from price
input_vars = full_dataset.drop('Price', axis = 1)
# aus output variable we will use only price
output_vars = full_dataset['Price']
full_dataset
```

6. Define a test and training set

```
# Define a test and training set, as split we will use 70/30% and we will shuffle the data before splitting it
input_vars_train, input_vars_test, output_vars_train, output_vars_test = train_test_split(input_vars,output_vars, test_size = 0.3, shuffle = True)
```

7. Define an algorithm/model

We will use https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html. It is an regression algorithm which is optimized via the ensemble method of Gradient Boosting.

```
# Define an algorithm/model
# We will use https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
model = GradientBoostingRegressor(
  n_estimators = 250,
  max_depth = 8,
  min_samples_split = 4,
  min_samples_leaf = 6,
  max_features = 0.6
)
```

8. Train the model

```
# Train the model
model.fit(input_vars_train, output_vars_train)
```

9. Calculate the quality of results

```
# Calculate the quality of results
model_prediction_training_data = model.predict(input_vars_train)
model_prediction_test_data = model.predict(input_vars_test)
error_training_data = mean_absolute_error(output_vars_train, model_prediction_training_data)
error_test_data = mean_absolute_error(output_vars_test, model_prediction_test_data)

print("mean error training data vs test data %.2f %.2f" % (error_training_data,error_test_data))
print("%.2f" % error_training_data)
print("%.2f" % error_test_data)
```

10. use it to forecast a value

```
# let's say we are getting new values, then we could use forecast the price.
# let's for example assume the line 3 of our dataset would be new and let's see
# what our model predicts
sample = full_dataset.iloc[[2]]
del sample['Price']
samplePrediction = model.predict(sample)
print("%.2f" % samplePrediction)
```

11. Tweak and tune

I will skip this here, but maybe it's worth investing some time in tuning parameters and compare results.
