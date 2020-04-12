
# coding: utf-8

# In[1]:


# import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


# the custom scaler class 
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]),
                                columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# create the special class that we are going to use from here on to predict new data
class titanic_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            
            df['Sex'].replace({'female':1, 'male':0},inplace=True)
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
            df['Embarked'] = df['Embarked'].fillna('S')
            df = df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1)
            
            # 把已有的数值型特征取出来丢进Random Forest Regressor中
            age_df = df[['Age', 'Fare', 'Sex', 'Parch', 'SibSp', 'Pclass']]
            # 乘客分成已知年龄和未知年龄两部分
            known_age = age_df[age_df.Age.notnull()].as_matrix()
            unknown_age = age_df[age_df.Age.isnull()].as_matrix()

            # y即目标年龄
            y = known_age[:, 0]
            # X即特征属性值
            X = known_age[:, 1:]
            
            # fit到RandomForestRegressor之中
            rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
            rfr.fit(X, y)

            # 用得到的模型进行未知年龄结果预测
            predictedAges = rfr.predict(unknown_age[:, 1:])

            # 用得到的预测结果填补原缺失数据
            df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
            
            df = pd.get_dummies(df)
            
            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)\
                [:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data


