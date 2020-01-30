import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

train = pd.read_csv('./train.csv') 
test = pd.read_csv('./test.csv')

"""**************Exploratory data analysis EDA**************"""

#our target distribution
prices_distribution = sns.distplot(train['SalePrice']) #distribution plot

print ( train.dtypes )

"""create numeric plots"""
num = [f for f in train.columns if train.dtypes[f] != 'object'] #just names of numerical attributes 
num.remove('Id') #id with no info 
nd = pd.melt(train, value_vars = num) #every attribute and it's corrsp value in a row
n1 = sns.FacetGrid (nd, col='variable', col_wrap=3, sharex=False, sharey = False)
n1 = n1.map(sns.distplot, 'value') #map : iterate 
n1

"""Correlation matrix"""
#see the corrolation of variables with our target(sales price)
corrmat = train.corr()
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(corrmat, vmax=.9, square=True);
#corrolation twarri el 9araba , dépendance bin les attribues , 
#on s'interrese lil ligna li5reniya eli fiha el corr( sale price , touts les autre attribues )  
#kol maykoun el couleur feta7 kol ma corééralition tzid => influance ta3 varible % target tzid 

"""**************dealing with missing values****************"""

missing = train.isnull().sum().sort_values(ascending=False)
print(missing)
#pourcentage of missing values 
#miss_percentage = train.isnull().sum()/len(train)
#print(miss_percentage)

def fill_missing_values(df):
    ''' This function imputes missing values with median for numeric columns 
        and most frequent value for categorical columns'''
    missing = df.isnull().sum()
    missing = missing[missing > 0] # we only want names of attributes eli fihom de s valeur missing , kn ne5dhou l kol nécrasiw el réel 
    for column in list(missing.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)   #index[0] to get the top frequent
        elif df[column].dtype == 'int64' or 'float64' or 'int16' or 'float16':
            df[column].fillna(df[column].median(), inplace=True) # we can use median() , mean(), zero 
            #Median is robust when we have outliers 

fill_missing_values(train)
fill_missing_values(test)

#print(train.dtypes) 
"""numerical data  & categorical data """
#num_data = [ f for f in train.columns if train.dtypes[f] != 'object']

#categ_data = [ f for f in train.columns if train.dtypes[f] == 'object']

"""***********************combine the data set*************************"""
alldata = train.append(test)
#alldata.shape
#(2915, 81)

"""************************* Encoding ************************************"""

#grouping neighborhood variable based on this plot
train['SalePrice'].groupby(train['Neighborhood']).median().sort_values().plot(kind='bar')

neighborhood_map = {"MeadowV" : 0, "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1, "Edwards" : 1,
                    "BrkSide" : 1, "Sawyer" : 1, "Blueste" : 1, "SWISU" : 2, "NAmes" : 2,
                    "NPkVill" : 2, "Mitchel" : 2, "SawyerW" : 2, "Gilbert" : 2, "NWAmes" : 2, 
                    "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 3, "Crawfor" : 3, "Veenker" : 3, 
                    "Somerst" : 3, "Timber" : 3, "StoneBr" : 4, "NoRidge" : 4, "NridgHt" : 4}

alldata['Neighborhood'] = alldata['Neighborhood'].map(neighborhood_map) #ordinal encoding manuel en se basant 3al figure 

#print ( alldata['Neighborhood'] )
"""ordinal data """
#Variable names which have 'quality' or 'qual', 'cond' in their names can be treated as ordinal variables
qual_dict = {np.nan: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
name = np.array(['ExterQual','PoolQC' ,'ExterCond','BsmtQual','BsmtCond',
                 'HeatingQC','KitchenQual','FireplaceQu', 'GarageQual','GarageCond'])

for i in name:
     alldata[i] = alldata[i].map(qual_dict).astype(int) #astype to do conversion of type to numerical instead of categorical
     

alldata["BsmtExposure"] = alldata["BsmtExposure"].map({np.nan: 0, "No": 1,
       "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

bsmt_fin_dict = {np.nan: 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
alldata["BsmtFinType1"] = alldata["BsmtFinType1"].map(bsmt_fin_dict).astype(int)
alldata["BsmtFinType2"] = alldata["BsmtFinType2"].map(bsmt_fin_dict).astype(int)

alldata["Functional"] = alldata["Functional"].map({np.nan: 0, "Sal": 1, "Sev": 2, "Maj2": 3,
       "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7, "Typ": 8}).astype(int)

alldata["GarageFinish"] = alldata["GarageFinish"].map({np.nan: 0, "Unf": 1, "RFn": 2, "Fin": 3}).astype(int)

alldata["Fence"] = alldata["Fence"].map({np.nan: 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)


# Simplifications of existing features into bad/average/good based on counts
alldata["SimplOverallQual"] = alldata.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, 
       4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})
alldata["SimplOverallCond"] = alldata.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, 
       4 : 2, 5 : 2, 6 : 2, 7 : 3, 8 : 3, 9 : 3, 10 : 3})


"""****************************feature engineering************************ """

#month sold and seasons influence ; 0 : winter , 1 : spring , 2 : summer , 3 : autumn    
alldata["SeasonSold"] = alldata["MoSold"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 
       6:2, 7:2, 8:2, 9:3, 10:3, 11:3}).astype(int)

#calculating total area using all area columns
area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
             'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF',
             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']

alldata["TotalArea"] = alldata[area_cols].sum(axis=1) #sum sur les lignes des attribus pour chaque maison 


alldata["HasGarage"] = (alldata["GarageArea"] == 0) * 1 
alldata["HasSecoFloor"] = ( alldata["2ndFlrSF"] == 0 ) * 1 
alldata["VeryNewHouse"] = (alldata["YearBuilt"] == alldata["YrSold"]) * 1
alldata["HasPool"] = ( alldata["PoolArea"] == 0 ) * 1 
alldata["hasBasement"] = (alldata["TotalBsmtSF"] == 0 ) * 1 

alldata["TotalArea1st2nd"] = alldata["1stFlrSF"] + alldata["2ndFlrSF"]
alldata["Age"] = 2010 - alldata["YearBuilt"]



#create new data
train_new = alldata[alldata['SalePrice'].notnull()]
test_new = alldata[alldata['SalePrice'].isnull()]

print ("Train" , train_new.shape )
print ('----------------')
print ("Test", test_new.shape)


"""***********************dealing with skewness*********************"""
#Skewness is the degree of distortion from the symmetrical bell curve or the normal curve.
#So, a symmetrical distribution will have a skewness of "0".

# having skewness effects the static models not robust with outliers
#if we remove skewness the distribution will become guassian again (with the log function)
def skewness(df):
    #get numeric features
    numeric_features = [f for f in df.columns if df[f].dtype != object]
    #transform the numeric features using log(x + 1) since a lot of values are eqal to 0 : log(0) indefined
    from scipy.stats import skew
    skewed = df[numeric_features].apply(lambda x: skew(x.dropna().astype(float))) #searching for skweed attributes 
    skewed = skewed[skewed > 0.75]  # n5aliw kn les attribus eli skeww barcha 
    skewed = skewed.index
    df[skewed] = np.log1p(df[skewed]) #log(x + 1)

skewness(train_new)
skewness(test_new)     
            
"""****************** Standarize Data *****************************"""
# to make all attributes centrée réduits 
def stand(df):
    numeric_features = [f for f in df.columns if df[f].dtype != object]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df[numeric_features])
    scaled = scaler.transform(df[numeric_features])
    
    for i, col in enumerate(numeric_features):
           df[col] = scaled[:,i]

stand(train_new)    
stand(test_new)
""" **************** Data Encoding : Get_dummies******************** """
del test_new["SalePrice"] 

train_new= pd.get_dummies(train_new)
test_new = pd.get_dummies(test_new ) 

# dealing with different attributes after encoding 
tt = []
for t in set(test_new.columns): 
    tt.append(t)

tr =[]
for g in set(train_new.columns): 
    tr.append(g)

diff_colum = set(tr) -set (tt)

print (diff_colum)

drop_colum = ['Condition2_RRAe','Condition2_RRAn', 'Condition2_RRNn', 'Electrical_Mix', 'Exterior1st_ImStucc',
 'Exterior1st_Stone','Exterior2nd_Other', 'Heating_Floor', 'Heating_OthW', 'HouseStyle_2.5Fin',
 'MiscFeature_TenC','RoofMatl_ClyTile', 'RoofMatl_Membran','RoofMatl_Metal','RoofMatl_Roll','Utilities_NoSeWa']

train_new.drop(drop_colum, axis=1, inplace=True)
train_new.shape

""" ************************ train / test split *********************** """
target = train_new["SalePrice"]
X = train_new.drop(["SalePrice"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
#80% of our data to training set (X_train and y_train) and the remaining 20% to the test set. 
#random_state parameter is important for reproducibility(when we train again the model same data chosen).

"""********************************* models stacking/ensembling ******************************"""

"""XGBOOST"""
#to install : pip install xgboost in the anaconda promt
import xgboost as xgb
regr = xgb.XGBRegressor(colsample_bytree=0.2,
                       gamma=0.0,
                       learning_rate=0.05,
                       max_depth=6,
                       min_child_weight=1.5,
                       n_estimators=7200,
                       reg_alpha=0.9,
                       reg_lambda=0.6,
                       subsample=0.2,
                       seed=42,
                       silent=1)

regr.fit(X_train, y_train) #training 

from sklearn.metrics import mean_squared_error
#we wil create the root mean squared error function( we don't have it predifined)
def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))
#the closer rmse to zero the better ( distance bitween regression line and real points)
#we always want to minimize errors      

# run prediction on training set to get an idea of how well it does
y_pred = regr.predict(X_train)
y_test = y_train
print("XGBoost score on training set: ", rmse(y_test, y_pred))

# make prediction on test set
y_pred_xgb = regr.predict(X_test)

#submit this prediction and get the score  #teba3 kaggle 
pred1 = pd.DataFrame({'Id': test['Id'], 'SalePrice': np.exp(y_pred_xgb)})
pred1.to_csv('xgbnono.csv', header=True, index=False)

""" LASSO regression """
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score, mean_absolute_error

# L1 regularization (sparse) + otomatis grid search
model = LassoCV(
            cv=10,
            # alphas=[0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1,
            #               0.3, 0.6, 1], # 166 dpt dr pilihan otomatis, tanpa param alphas
            max_iter=50000,
            n_jobs=-1, 
            random_state=1
      )

model = model.fit(X_train, y_train)

y_pred_cv = model.predict(X_train)
y_pred_test = model.predict(X_test)

print('Best Alpha : ' + str(model.alpha_))

print('Train MAE : ' + str(mean_absolute_error(y_train, y_pred_cv)))
print('Train R^2 : ' + str(r2_score(y_train, y_pred_cv)))

print('Test R^2 : ' + str(r2_score(y_test, y_pred_test)))
print('Test MSE : ' + str(mean_squared_error(y_test, y_pred_test)))
print('Test MAE : ' + str(mean_absolute_error(y_test, y_pred_test)))


#staking models will give better performance 
from vecstack import stacking


