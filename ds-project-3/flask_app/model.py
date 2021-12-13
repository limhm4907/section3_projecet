import pandas as pd
import numpy as np
import os, csv, sqlite3
import pickle

import sklearn, xgboost
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score



########## STEP 1 ##########
# DB File Path
#DB_FILEPATH = ('C:/Users/Hyunmin/ai-bootcamp/ds-project-3/Churn.db')

# connect database
#conn = sqlite3.connect(DB_FILEPATH)
#cur = conn.cursor()

# export data into CSV file
#cur.execute("select * from Customer")

#with open("Churn.csv", "w", newline='') as f:
#    writer = csv.writer(f) # delimiter="\t"
#    writer.writerow([i[0] for i in cur.description]) # 1st row: field name (=column name)
#    writer.writerows(cur)

#conn.close()



########## STEP 2 ##########
# CSV File Path
CSV_EXPORTED = os.path.join(os.path.dirname(__file__), 'Churn.csv')

# import the exported CSV file from database
df = pd.read_csv(CSV_EXPORTED)

# data wrangling 1: remove id column
df.drop('id', axis=1, inplace=True)

# data wrangling 2: remove duplicated data 
df_rev = df.drop_duplicates(keep='first')
df_rev.reset_index(drop=True, inplace=True)

# data wrangling 3: change type of target variable
df_rev['churn'] = df_rev['churn'].apply(lambda x: 0 if x=="No" else 1).astype(int)

# data wrangling 4: change type of 'totalcharges' column & remove nan value
df_rev['totalcharges']= df_rev['totalcharges'].apply(lambda v: v if v!= ' ' else np.nan).astype(float)
df_rev.dropna(axis=0, inplace=True)
df_rev.reset_index(drop=True, inplace=True)



########## STEP 3 ##########
# feature engineering
def engineer(data):

    # create new column: 'numadd' (counting additional services)
    add1 = [1 if a=='Yes' else 0 for a in list(data['multiplelines'])]
    add2 = [1 if a=='Yes' else 0 for a in list(data['onlinesecurity'])]
    add3 = [1 if a=='Yes' else 0 for a in list(data['onlinebackup'])]
    add4 = [1 if a=='Yes' else 0 for a in list(data['deviceprotection'])]
    add5 = [1 if a=='Yes' else 0 for a in list(data['techsupport'])]
    add6 = [1 if a=='Yes' else 0 for a in list(data['streamingtv'])]
    add7 = [1 if a=='Yes' else 0 for a in list(data['streamingmovies'])]
    ziplist = zip(add1,add2,add3,add4,add5,add6,add7)
    addsum = [a+b+c+d+e+f+g for a,b,c,d,e,f,g in ziplist]
    data['numadd'] = addsum
    
    return data

df_eng = engineer(df_rev)

# encoding
enc = OrdinalEncoder(mapping=[
    {'col': 'phone', 'mapping': {None:-2, 'No':1, 'Yes':2}},
    {'col': 'internet', 'mapping': {None:-2, 'No':1, 'DSL':2, 'Fiber optic':3}},
    {'col': 'contract', 'mapping': {None:-2, 'Month-to-month':1, 'One year':2, 'Two year':2}},
    {'col': 'paymentmethod', 'mapping': {None:-2, 'Mailed check':1, 'Electronic check':2, 'Credit card (automatic)':3, 'Bank transfer (automatic)':4}},
    ])

# split data
target = 'churn'
features = ['tenure', 'phone', 'internet', 'contract', 'paymentmethod', 'monthlycharges', 'numadd']

X_train, X_test, y_train, y_test = train_test_split(df_eng[features], df_eng[target], test_size=0.2, random_state=2, stratify=df_eng[target])

X_train_enc = enc.fit_transform(X_train)
X_test_enc = enc.transform(X_test)



########## STEP 4 ##########
# model fit / Random Forest

model = RandomForestClassifier(min_samples_leaf=4, class_weight="balanced", oob_score=True, n_jobs=-1, random_state=2)

model.fit(X_train_enc, y_train)

#y_pred = model.predict(X_train_enc)

#print("train accuracy: ", pipe.score(X_train_eng, y_train))
#print("train recall_score: ", recall_score(y_train,y_pred))
#print("train f1_score: ", f1_score(y_train,y_pred), '\n')



########## STEP 5 ##########
# test

# X_test_sample = [[4,'Yes','DSL','Month-to-moth','Credit card (automatic)',51.75,1]]
X_test_sample = [[4,2,2,1,3,51.75,1]]
y_pred_sample = model.predict(X_test_sample)

print(f'해당 조건의 고객의 탈퇴여부는 {y_pred_sample}로 예측됩니다.') #No:0, Yes:1



########## STEP 6 ##########
# pickle

# with open('model.pkl', 'wb') as pickle_file:
#    pickle.dump(model, pickle_file)