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
import torch
import torch.optim as optim  # you already have this line for optimizers
from tqdm.notebook import tqdm
from sqlalchemy import create_engine
import torch.optim as optim # Import the optim module
from tqdm.notebook import tqdm
import torch
from torch import nn
import torch.nn.functional as F



#Create a sql engine and get data from sql table. 


df = pd.read_sql(f"SELECT * FROM randomforest_with_ccr LIMIT 20000", con=engine)

#Data pre-processing:
df.dropna(subset=['LOS'], inplace=True)
#drop year, does nothing, TOTCHG IS CORRELATED:
df.drop(columns=['YEAR', 'S_DISC_U', 'KEY_NIS', 'S_HOSP_U', 'N_DISC_U', 'DISCWT', 'DRG_NoPOA', 'MDC_NoPOA', 'I10_PR2'], inplace=True)

df['cost'] = df['TOTCHG'] * df["'CCR_NIS'"]

print("cost column")
print(df['cost'])

df.dropna(thresh=df.shape[0]*0.55, axis=1, inplace=True)

print("The columns of the dataset are:")
print(df.columns)

onehot = ["AMONTH", "AWEEKEND", "ELECTIVE", "FEMALE", "PAY1", "RACE", "TRAN_IN", "TRAN_OUT", "HOSP_LOCTEACH", "HOSP_REGION", "H_CONTRL"]

onehotcolumn = []
for name in onehot:
    for i in range(0, 20):
        onehotcolumn.append(name + "_" + str(i))

df = pd.get_dummies(df, columns=onehot, drop_first=True)

#print data that should now have columns with onehot encoding
print("Should now have one hot encoding")
print(df)


#Get max values for each embedding column so we can set embedding sizes: 


df['DRG'] = df['DRG'].factorize()[0]
df['I10_DX1'] = df['I10_DX1'].factorize()[0]
df['I10_DX2'] = df['I10_DX2'].factorize()[0]
df['I10_DX3'] = df['I10_DX3'].factorize()[0]
df['I10_DX4'] = df['I10_DX4'].factorize()[0]
df['I10_DX5'] = df['I10_DX5'].factorize()[0]
df['I10_PR1'] = df['I10_PR1'].factorize()[0]
df['APRDRG'] = df['APRDRG'].factorize()[0]
df['MDC'] = df['MDC'].factorize()[0]

df.replace(-1, 0, inplace=True)
print(min(df['I10_DX2']))

max_drg = df['DRG'].max()
max_aprdrg = df['APRDRG'].max()
max_dx1 = df['I10_DX1'].max()
max_dx2 = df['I10_DX2'].max()
max_dx3 = df['I10_DX3'].max()
max_dx4 = df['I10_DX4'].max()
max_dx5 = df['I10_DX5'].max()
max_pr1 = df['I10_PR1'].max()
max_mdc = df['MDC'].max()





#Full Neural Network Model With Embeddings For Regression
class FullRegressionModel(nn.Module):
    #Input layer -> Hidden layer 1 (number of neurons) -> Hidden layer 2 (number of neurons) -> Output
    def __init__(self, in_features, out_features):
        super(FullRegressionModel, self).__init__()
        
        #Embedding Layers:
        
        self.emb_drg     = nn.Embedding(max_drg + 1, 5)
        self.emb_aprdrg  = nn.Embedding(max_aprdrg + 1, 5)
        
        self.emb_primary   = nn.Embedding(max_dx1 + 1, 8)
        self.emb_secondary = nn.Embedding(max_dx2 + 1, 8)
        self.emb_third     = nn.Embedding(max_dx3 + 1, 8)
        self.emb_fourth    = nn.Embedding(max_dx4 + 1, 8)
        self.emb_fifth     = nn.Embedding(max_dx5 + 1, 8)
        
        self.pr1 = nn.Embedding(max_pr1 + 1, 8)
        self.mdc = nn.Embedding(max_mdc + 1, 8)

        #layers
        self.fc1 = nn.Linear(in_features, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 40)
        self.out = nn.Linear(40, out_features)
        
    

    
    def forward(self, numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc):
        #Embed our categorical variables
        embedded_drg = self.emb_drg(drg_id)
        embedded_aprdrg = self.emb_aprdrg(aprdrg_id)
        embedded_primary = self.emb_primary(primary_id)
        embedded_secondary = self.emb_secondary(secondary_id)
        embedded_third = self.emb_third(third_id)
        embedded_fourth = self.emb_fourth(fourth_id)
        embedded_fifth = self.emb_fifth(fifth_id)
        embedded_pr1 = self.pr1(pr1_id)
        embedded_mdc = self.mdc(mdc)
        

        emb_tens = torch.cat(
                [
        embedded_drg,
        embedded_aprdrg,
        embedded_primary,
        embedded_secondary,
        embedded_third,
        embedded_fourth,
        embedded_fifth,
        embedded_pr1,
        embedded_mdc,
        ],
                dim=1
            )
        final_tens = torch.cat([numerical_data, emb_tens], dim=1)
        lay_1 = F.relu(self.fc1(final_tens))
        lay_2 = F.relu(self.fc2(lay_1))
        lay_3 = F.relu(self.fc3(lay_2))
        output = self.out(lay_3)

        return output




#Dataset
import torch
from torch.utils.data import Dataset
import numpy as np


class FullRegressionFast(Dataset):
    def __init__(self, df):
        
        drop_cols = [
            'I10_DX1','I10_DX2','I10_DX3','I10_DX4','I10_DX5',
            'I10_PR1','APRDRG','DRG','MDC','LOS'
        ]
        cont_cols = [c for c in df.columns if c not in drop_cols]

        X_cont_np = df[cont_cols].to_numpy(dtype=np.float32, copy=True)
        drg_np    = df['DRG'].to_numpy(dtype=np.int64,  copy=False)
        apr_np    = df['APRDRG'].to_numpy(dtype=np.int64, copy=False)
        dx1_np    = df['I10_DX1'].to_numpy(dtype=np.int64, copy=False)
        dx2_np    = df['I10_DX2'].to_numpy(dtype=np.int64, copy=False)
        dx3_np    = df['I10_DX3'].to_numpy(dtype=np.int64, copy=False)
        dx4_np    = df['I10_DX4'].to_numpy(dtype=np.int64, copy=False)
        dx5_np    = df['I10_DX5'].to_numpy(dtype=np.int64, copy=False)
        pr1_np    = df['I10_PR1'].to_numpy(dtype=np.int64, copy=False)
        mdc_np    = df['MDC'].to_numpy(dtype=np.int64,  copy=False)
        y_np      = df['LOS'].to_numpy(dtype=np.int32, copy=True)

        # Convert entire columns to tensors ONCE.
        # (These stay on CPU; DataLoader pin_memory + non_blocking .to(device) will be fast.)
        self.X_cont = torch.from_numpy(X_cont_np)             # shape [N, D], float32
        self.drg    = torch.from_numpy(drg_np)                # int64 (Long)
        self.apr    = torch.from_numpy(apr_np)
        self.dx1    = torch.from_numpy(dx1_np)
        self.dx2    = torch.from_numpy(dx2_np)
        self.dx3    = torch.from_numpy(dx3_np)
        self.dx4    = torch.from_numpy(dx4_np)
        self.dx5    = torch.from_numpy(dx5_np)
        self.pr1    = torch.from_numpy(pr1_np)
        self.mdc    = torch.from_numpy(mdc_np)
        self.y      = torch.from_numpy(y_np)                  # float32

    def __len__(self):
        return self.X_cont.shape[0]

    def __getitem__(self, i):
        # Return tensor views (no new allocations / no pandas calls)
        return (
            self.X_cont[i],
            self.drg[i],
            self.apr[i],
            self.dx1[i], 
            self.dx2[i], 
            self.dx3[i], 
            self.dx4[i], 
            self.dx5[i],
            self.pr1[i],
            self.mdc[i],
        ), self.y[i]



#Get the data into train, test, val splits.
df_x = df.drop(columns=['LOS'])

X_train, X_test, y_train, y_test = train_test_split(df, df['LOS'], test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)



#Actual training loop:

from tqdm import tqdm

X_train_dataset = FullRegressionFast(X_train)
X_val_dataset = FullRegressionFast(X_val)
X_test_dataset = FullRegressionFast(X_test)


train_dataloader = DataLoader(
    X_train_dataset, shuffle=False, batch_size=100
)
val_dataloader = DataLoader(
    X_val_dataset, shuffle=False, batch_size=100
)
test_dataloader = DataLoader(
    X_test_dataset, shuffle=False, batch_size=100
)


num_epochs = 10
train_losses, val_losses, test_losses = [], [], []


model = FullRegressionModel(138, 1)

print(model.parameters())

criterion = torch.nn.SmoothL1Loss(beta=1.0)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    print("EXECUTE")
    model.train()
    running_loss = 0.0
    for (numerical_data,
     drg_id, aprdrg_id,
     primary_id, secondary_id, third_id, fourth_id, fifth_id,
     pr1_id, mdc), labels in tqdm(train_dataloader, desc="Training loop"):
        optimizer.zero_grad()
        outputs = model(numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc)
        loss = criterion(outputs.squeeze(1), labels)
        loss.backward()
        optimizer.step()
        running_loss += (loss.item() * numerical_data.size(0))
    train_loss = running_loss / len(train_dataloader.dataset)
    train_losses.append(train_loss)

#Validation mode
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for (numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc), labels in tqdm(val_dataloader, desc="Validation loop"):
            outputs = model(numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc)
            loss = criterion(outputs.squeeze(1), labels)
            running_loss += (loss.item() * numerical_data.size(0))
        val_loss = running_loss / len(val_dataloader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
        # plt.plot(train_losses, label='Training loss')
        # plt.plot(val_losses, label='Validation loss')
        # plt.legend()
        # plt.title("Loss over epochs")
        # plt.show()


#Test:
model.eval()
running_loss = 0.0

y_pred = []
y_true = []

from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = (mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)

with torch.no_grad():
    for (numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc), labels in tqdm(test_dataloader, desc="Test Loop"):
        outputs = model(numerical_data, drg_id, aprdrg_id, primary_id, secondary_id, third_id, fourth_id, fifth_id, pr1_id, mdc)
        y_pred.append(outputs.squeeze(1).numpy())
        y_true.append(labels.numpy())
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
    plt.plot(test_losses, label='test loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.show()















