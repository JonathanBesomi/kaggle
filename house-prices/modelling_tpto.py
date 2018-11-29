# Import the optimizer class
from tpot import TPOTRegressor
import pandas as pd


MAX_TIME_MINS = 5

# Create a tpot optimizer with parameters
tpot = TPOTRegressor(scoring='neg_mean_absolute_error',
                     max_time_mins=MAX_TIME_MINS,
                     n_jobs=-1,
                     verbosity=2,
                     cv=5)

print("START MODELLING (TPTO) ...")

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train_ID = train['Id']
test_ID = test['Id']

train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)


y_train = train.SalePrice.values
train.drop(['SalePrice'], axis=1, inplace=True)


tpot.fit(train, y_train)

tpot.export('tpot_exported_pipeline.py')
