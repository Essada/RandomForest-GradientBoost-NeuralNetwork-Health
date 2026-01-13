#imports
import pandas as pd
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

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

from sklearn.preprocessing import TargetEncoder


#Set option for debugging: 
pd.set_option("display.max_rows", 150)

#First, load in the entire dataset that you will be working with:
df = pd.read_csv("/Users/adamessawi/Downloads/Dataset_and_Result/data/raw/merged_training_dataset.csv") 

print("The columns of the dataset are:")
print(df.columns)

#Now, replace all null values with the np.nan data type:
replacement_values=[-9, -8, -6, -5, -999, -888, -666, -9, -8, -6, -5, 
                -999999999, -888888888, -666666666, -99, -88, -66, -9999, -8888, -6666, -999999999, -888888888, 
                -666666666, -99.9999999, -88.8888888, -66.6666666]

df = df.replace(to_replace=replacement_values, value=np.nan)

#Debug Check, should now have null values showing up. 
print(df.isnull().sum())

#Now, lets make the LOS a categorical variable since we are doing classification into seperate buckets: 
bins = [0, 2, 4, 6, 8, float("inf")] 
labels = ["0-2", "2-4", "4-6", "6-8", "8+"]
df["LOS"] = pd.cut(df["LOS"], bins=bins, labels=labels, right=False)

print("The LOS column should now be categorical: ")
print(df['LOS'])



#Split data into Train and Test split, 80% train, 20% test:
df_x = df.drop(columns=['LOS', 'LOS.1'])
X_train, X_test, y_train, y_test = train_test_split(df_x, df['LOS'], test_size=0.2, random_state=42)

print("The length of the X_train dataset is: " + str(len(X_train)) + " It should be 80 percent of a million")
print("The length of the X_test dataset is: " + str(len(X_test)) + " It should be 20 percent of a million")
print("The length of the y_train dataset is: " + str(len(y_train)) + " It should be the same as X_train")
print("The length of the y_test dataset is: " + str(len(y_test)) + " It should be the same as X_test")


#Now, replace missing values in the age category with the mean age of train dataset. Do this for both test and train. 

avg_age = X_train['AGE'].mean()

#Debug statement:
print("The average age of the training dataset is: " + str(avg_age))

X_train['AGE'] = X_train['AGE'].fillna(avg_age)
X_test['AGE'] = X_test['AGE'].fillna(avg_age)

#Now, replace missing values in all other categories that are not non-numeric codes (these do not have any missing values):

#Categories that are non numeric codes:
non_numeric_code_columns = ['CORE_C0053', 'CORE_C0054', 'CORE_C0055', 'CORE_C0056',
    'CORE_C0057', 'CORE_C0058', 'CORE_C0078', 'CORE_C0079',
    'CORE_C0080', 'CORE_C0081', 'CORE_C0082', 'CORE_C0083',
    'CORE_C0084', 'CORE_C0085', 'CORE_C0086', 'CORE_C0087',
       ]

for column in X_train.columns:
    if column not in non_numeric_code_columns:
        avg_age = X_train[column].mode()[0]
        X_train[column] = X_train[column].fillna(avg_age)
        X_test[column] = X_test[column].fillna(avg_age)


#Now, test if all missing values have been imputed: 
print("The number of missing values in the training dataset:")
print(X_train.isnull().sum())
print("The number of missing values in the test dataset:")
print(X_test.isnull().sum())


#Finally, we want to do label encoding for the non-numeric code columns:
enc = TargetEncoder(smooth="auto", target_type="auto")

#Want to remove the AGE column from this list, not a categorical variable. Save it first:

#Save the original dataframe for when we have to transform the numpy array into a dataframe. 

Age_column_train = X_train['AGE']
X_train.drop(columns=['AGE'], axis=1, inplace=True)

X_train_copy = X_train.copy()

X_train = enc.fit_transform(X_train, y_train)

X_train = pd.DataFrame(X_train, columns=X_train_copy.columns, index=X_train_copy.index)

X_train['AGE'] = Age_column_train

#Now, do the same for the test set:
Age_column_test = X_test['AGE']
X_test.drop(columns=['AGE'], axis=1, inplace=True)

X_test_copy = X_test.copy()
X_test = enc.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X_test_copy.columns, index=X_test_copy.index)
X_test['AGE'] = Age_column_test

#Now, should be done with target encoding:

print("The training dataset after target encoding is: ")
print(X_train)
print("The test dataset after target encoding is: ")
print(X_test)


#Finally, do cross validation and training with random forest classifier:






import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

#Get test subset of data.
X_small, y_small = X_train.head(100000), y_train.head(100000)

# Base classifier
rf = RandomForestClassifier(
    n_jobs=-1,
    random_state=42, # helps if classes are imbalanced
)

# Reasonable search space for classification
param_dist = {
    "n_estimators":      randint(10, 100),
    "max_depth":         [10, 20, 40],
    #"min_samples_split": randint(2, 20),
   # "min_samples_leaf":  randint(1, 10),
    #"max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
    #"bootstrap":         [True],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=15,                     # lower for speed, raise if fast
    cv=cv,
    scoring="accuracy",   
    n_jobs=-1,
    verbose=1,
    random_state=42,
    return_train_score=True
)

search.fit(X_small, y_small)
print("Best CV accuracy:", search.best_score_)
print("Best params:", search.best_params_)

# Train best model on the full training set, then evaluate on the hold-out test set
best_rf = search.best_estimator_
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, y_pred))
print("Test balanced accuracy:", balanced_accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



















