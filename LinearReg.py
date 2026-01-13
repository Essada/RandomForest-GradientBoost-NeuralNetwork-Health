import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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
from sklearn.metrics import mean_squared_error as rmse

from sklearn.preprocessing import TargetEncoder


df = pd.read_sql(f"SELECT * FROM randomforest_with_ccr ORDER BY RAND()", con=engine)

df.dropna(subset=['LOS'], inplace=True)

#drop year, does nothing, TOTCHG IS CORRELATED:
df.drop(columns=['YEAR', 'S_DISC_U', 'KEY_NIS', 'S_HOSP_U', 'N_DISC_U', 'DISCWT'], inplace=True)

df['cost'] = df['TOTCHG'] * df["'CCR_NIS'"]

print("cost column")
#print(df['cost'])

#Set option to see all rows when printing.
pd.set_option("display.max_rows", 200)

#Drop any column with over 55 percent missing values:
df.dropna(thresh=df.shape[0]*0.55, axis=1, inplace=True)

print("The columns of the dataset are:")
print(df.columns)

onehot = ["AMONTH", "PAY1", "RACE", "TRAN_IN", "TRAN_OUT", "HOSP_LOCTEACH", "HOSP_REGION", "H_CONTRL"]

print(df.isnull().sum())

onehotcolumn = []
for name in onehot:
    for i in range(0, 20):
        onehotcolumn.append(name + "_" + str(i))

df = pd.get_dummies(df, columns=onehot, drop_first=True)

#print data that should now have columns with onehot encoding
print("Should now have one hot encoding")
print(df)

#Split the data:
df_x = df.drop(columns=['LOS'])
X_train, X_test, y_train, y_test = train_test_split(df_x, df['LOS'], test_size=0.10, random_state=42)

enc_auto = TargetEncoder(smooth="auto", target_type="continuous")


for name in X_train.columns:
    if name != 'AGE' and name != 'LOS' and name != 'TOTCHG' and name != 'cost' and name != 'HOSP_BEDSIZE' and name != "APRDRG_Severity" and name != "APRDRG_Risk_Mortality" and name != "ZIPINC_QRTL" and name not in onehotcolumn:
        X = X_train[[name]]
        Y = y_train
        X_train[name] = enc_auto.fit_transform(X, Y)

        xtest = X_test[[name]]
        X_test[name] = enc_auto.transform(xtest)

Age_mean = X_train['AGE'].mean()
TOTCHG_mean = X_train['TOTCHG'].mean()
ZIPINC_QRTL_mode = X_train['ZIPINC_QRTL'].mode()[0]
cost_mean = X_train['cost'].mean()

#Replace w/ mean.


X_train['AGE'].replace(np.nan, Age_mean, inplace=True)
X_test['AGE'].replace(np.nan, Age_mean, inplace=True)

X_train['TOTCHG'].replace(np.nan, TOTCHG_mean, inplace=True)
X_test['TOTCHG'].replace(np.nan, TOTCHG_mean, inplace=True)

X_train['ZIPINC_QRTL'].replace(np.nan, ZIPINC_QRTL_mode, inplace=True)
X_test['ZIPINC_QRTL'].replace(np.nan, ZIPINC_QRTL_mode, inplace=True)

X_train['cost'].replace(np.nan, cost_mean, inplace=True)
X_test['cost'].replace(np.nan, cost_mean, inplace=True)


model = LinearRegression()

model.fit(X_train, y_train)

coef_df = pd.DataFrame({"variable": X_train.columns, "coef": np.ravel(model.coef_)})
coef_df = pd.concat([pd.DataFrame({"variable":["Intercept"], "coef":[model.intercept_]}), coef_df], ignore_index=True)
coef_df.to_csv("los_linear_coefficients.csv", index=False)
print(coef_df)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R^2:  {r2:.4f}")






