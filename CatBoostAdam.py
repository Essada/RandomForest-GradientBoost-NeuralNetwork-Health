
import pandas as pd
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import catboost


from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as rmse

from sklearn.preprocessing import TargetEncoder

from sqlalchemy import create_engine



#Adding CCR:


#Create a sql engine and get data from sql table. 

df = pd.read_sql(f"SELECT * FROM randomforest_with_ccr", con=engine)


df.dropna(subset=['LOS'], inplace=True)
#drop year, does nothing, TOTCHG IS CORRELATED:
df.drop(columns=['YEAR', 'S_DISC_U', 'KEY_NIS', 'S_HOSP_U', 'N_DISC_U', 'DISCWT'], inplace=True)

df['cost'] = df['TOTCHG'] * df["'CCR_NIS'"]

print(df['cost'])

#Set option to see all rows when printing.
pd.set_option("display.max_rows", 150)

#Drop any column with over 55 percent missing values:
df.dropna(thresh=df.shape[0]*0.55, axis=1, inplace=True)

print("The columns of the dataset are:")
print(df.columns)


feature_col = []
#Convert all object types to strings except age and LOS for catboost model:
for name in df.columns:
    if name != 'AGE' and name != 'LOS' and name != 'TOTCHG' and name != 'cost':
        feature_col.append(name)
        print(name)
        df[name] = df[name].astype(str)
        print("is name in this loop, shouldnt be numeric!")



#Now replace all null values with a string "-1":
df = df.replace(to_replace=np.nan, value="-1")


#One last check of the data set:
print("Number of missing values:")
print(df.isnull().sum())
print("The columns of the dataset are:")
print(df.columns)
print("The shape of the dataset is:")
print(len(df))


#Split data into Train and Test split, 80% train, 20% test:
df_x = df.drop(columns=['LOS'])
X_train_before, X_test, y_train_before, y_test = train_test_split(df_x, df['LOS'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_before, y_train_before, test_size=0.1, random_state=42)




print("The length of the X_train dataset is: " + str(len(X_train)) + " It should be 80 percent of a million")
print("The length of the X_test dataset is: " + str(len(X_test)) + " It should be 20 percent of a million")
print("The length of the y_train dataset is: " + str(len(y_train)) + " It should be the same as X_train")
print("The length of the y_test dataset is: " + str(len(y_test)) + " It should be the same as X_test")
print("The length of the X_val dataset is: " + str(len(X_val)) + " It should be 10 percent of 80 percent of a million")
print("The length of the y_val dataset is: " + str(len(y_val)) + " It should be the same as X_val")


#model = CatBoostRegressor(loss_function='RMSE', cat_features=feature_col)

model = CatBoostRegressor(
    cat_features=feature_col,
    iterations=1000,
    learning_rate=0.1,
    eval_metric='RMSE',
    use_best_model=True,
    od_type='Iter',  # Use iteration-based overfitting detection
    od_wait=50  # Number of iterations to wait before stopping
)

eval_pool = (X_val, y_val)

model.fit(X_train, y_train,  eval_set=eval_pool, verbose=100)

#model.fit(X_train, y_train, verbose=100)
model.save_model('my_catboost_w_cost.cbm')

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as rmse

# Generate predictions on the training and validation sets using the trained 'model'
y_out = model.predict(X_train)
y_test_out = model.predict(X_test)

print("Training RMSE: ", (rmse(y_train, y_out)))
print("Training MAE: ", (mae(y_train, y_out)))
print("Training R2: ", r2_score(y_train, y_out))

print("Testing RMSE: ", (rmse(y_test, y_test_out)))
print("TEST MAE: ", (mae(y_test, y_test_out)))
print("Test R2: ", r2_score(y_test, y_test_out))

feature_importance = model.get_feature_importance()
feature_names = df_x.columns

# Display feature importance
for name, importance in zip(feature_names, feature_importance):
    print(f"Feature: {name}, Importance: {importance:.2f}")

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()




