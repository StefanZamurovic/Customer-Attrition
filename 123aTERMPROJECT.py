# -*- coding: utf-8 -*-
"""
@author: Stefan Zamurovic

"""
import pandas as pd, numpy as np
import xlrd
import xlwt
import lightgbm as lgb
import skopt
import matplotlib.pyplot as plt
from statsmodels import robust
from scipy import interp
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score




# number of Attrited Customer = 1627

'''
    Plotting data to help Visualize relationships bettween 
different Features and Classification

'''

import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


data = pd.read_csv('C:/Users/funny/Desktop/BankChurners.csv')

fig = px.scatter(x=data['Months_on_book'], y=data['Total_Trans_Amt'], 
                  color = data['Attrition_Flag'], template = 'presentation', 
                  opacity = 0.5, facet_col = data['Income_Category'], 
                  title = 'Customer Attrition by Months, Charges, and Income',
                  labels = {'x' : 'Customer Tenure', 'y' : 'Total Charges $'})

fig2 = px.scatter(x=data['Customer_Age'], y=data['Months_Inactive_12_mon'], 
                  color = data['Attrition_Flag'], template = 'presentation', 
                  opacity = 0.5, facet_col = data['Card_Category'], 
                  title = 'Customer Attrition by Age, Inactivity, and Card Type',
                  labels = {'x' : 'Age', 'y' : 'Months Inactive (12m)'})

fig3 = px.scatter(x=data['Avg_Utilization_Ratio'], y=data['Total_Trans_Ct'], 
                  color = data['Attrition_Flag'], template = 'presentation', 
                  opacity = 0.5, facet_col = data['Marital_Status'], 
                  title = 'Customer Attrition by Utilization, # of Transaction, and Martial Status',
                  labels = {'x' : 'Utilization', 'y' : '# of Transaction'})

fig4 = px.scatter(x=data['Avg_Utilization_Ratio'], y=data['Total_Trans_Ct'], 
                  color = data['Attrition_Flag'], template = 'presentation', 
                  opacity = 0.5, facet_col = data['Education_Level'], 
                  title = 'Customer Attrition by Utilization, # of Transaction, and Education',
                  labels = {'x' : 'Utilization', 'y' : '# of Transaction'})

fig5 = px.scatter(x=data['Months_on_book'], y=data['Total_Trans_Amt'], 
                  color = data['Attrition_Flag'], template = 'presentation', 
                  opacity = 0.5, facet_col = data['Gender'], 
                  title = 'Customer Attrition by Months, Charges, and Gender',
                  labels = {'x' : 'Customer Tenure', 'y' : 'Total Charges $'})


# fig.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()

'''
    End plotting and data visualization
'''

'''
    Start Data Preperation
'''


# number of Attrited Customer = 1627
df = pd.read_csv('C:/Users/funny/Desktop/BankChurners.csv')
df = df.drop("CLIENTNUM", 1) 

# counting the number of attried data points in the data
count =0
for i in range(len(df)):
    if df.iloc[i][0] == "Attrited Customer":
        count+=1

# for Light Gradient Boost Modifer or LightGBM for short, data that arent integers or floats
# need to be of type catagory


df2 = pd.read_csv('C:/Users/funny/Desktop/BankChurners.csv')
df2 = df2.drop("CLIENTNUM", 1) 
df2["Attrition_Flag"] = df2["Attrition_Flag"].astype('category')
df2["Gender"] = df2["Gender"].astype('category')
df2["Education_Level"] = df2["Education_Level"].astype('category')
df2["Marital_Status"] = df2["Marital_Status"].astype('category')
df2["Income_Category"] = df2["Income_Category"].astype('category')
df2["Card_Category"] = df2["Card_Category"].astype('category')

# second data set to ensure i do not run into any data copying issues

df3 = pd.read_csv('C:/Users/funny/Desktop/BankChurners.csv')
df3 = df3.drop("CLIENTNUM", 1) 
# df3["Attrition_Flag"] = df3["Attrition_Flag"].astype('category')
df3["Gender"] = df3["Gender"].astype('category')
df3["Education_Level"] = df3["Education_Level"].astype('category')
df3["Marital_Status"] = df3["Marital_Status"].astype('category')
df3["Income_Category"] = df3["Income_Category"].astype('category')
df3["Card_Category"] = df3["Card_Category"].astype('category')

df3 = df3.sort_values(by=['Attrition_Flag'])

# re-writing the attrited and exisiting values as ones and zeros

count0 = 0
numberVal = []
# for i in range(len(df)):
#     if df.iloc[i][0] == "Attrited Customer":
#         numberVal.append(0)
#         count0 += 1
#     else:
#         numberVal.append(1)

for i in range(len(df)):
    if df.iloc[i][0] == "Attrited Customer":
        numberVal.append(1)
        count0 += 1
    else:
        numberVal.append(0)

# spliting ones and zeros into two different sets

numberVal = np.array(numberVal)
numberVal = np.sort(numberVal)
# AC = numberVal[:1627]
# EC = numberVal[1627:]
EC = numberVal[:8500]
AC = numberVal[8500:]

#spliting the data into two sets, the attrited and the exisiting
grouped = df3.groupby(df3.Attrition_Flag)
df_A = grouped.get_group("Attrited Customer")
df_E = grouped.get_group("Existing Customer")

#replacing Atrited and existing with ones and zeros in dataframe
df_A["Attrition_Flag"] = AC
df_E["Attrition_Flag"] = EC

# defining the different values the model will score

scoring = ['precision', 'recall', 'auc', 'accuracy']

# doing 80-20 split with the data. it is not perfectly 80-20 as there is not an even number of data points

A_train, A_test = np.split(df_A, [1302])
E_train, E_test = np.split(df_E, [6800])

AfoldTrain = []
EfoldTrain = []
for i in range(5):
    fold = []
    #spliting th training data into 5 even folds. not perfect because not an even amount of data points
    a, b, c, d, e = np.split(A_train, [260,520,780,1040])
    fold.append(a)
    fold.append(b)
    fold.append(c)
    fold.append(d)
    fold.append(e)
    AfoldTrain.append(fold)
    fold = []
    a, b, c, d, e = np.split(E_train, 5)
    fold.append(a)
    fold.append(b)
    fold.append(c)
    fold.append(d)
    fold.append(e)
    EfoldTrain.append(fold)
    A_train = A_train.sample(frac = 1)
    E_train = E_train.sample(frac = 1)


# putting together the attrited and exisiting data and shuffling them
allTrainX = []
allTrainY = []
for i in range(len(AfoldTrain)):
    tempX = []
    tempY = []
    for j in range(5):
        partsT = [AfoldTrain[i][j], EfoldTrain[i][j]]
        tv = pd.concat(partsT)
        tv = tv.sample(frac = 1)
        trainX = tv.drop('Attrition_Flag', axis=1)
        trainY = tv['Attrition_Flag']
        tempX.append(trainX)
        tempY.append(trainY)
    allTrainX.append(tempX)
    allTrainY.append(tempY)



df2["Attrition_Flag"] = numberVal
X = df2.drop('Attrition_Flag', axis=1)
y = df2['Attrition_Flag']

partsT = [A_train,E_train]
train = pd.concat(partsT)
train = train.sample(frac = 1)
X_train = train.drop('Attrition_Flag', axis=1)
y_train = train['Attrition_Flag']

partsV = [A_test,E_test]
valid = pd.concat(partsV)
valid = valid.sample(frac = 1)
X_valid = valid.drop('Attrition_Flag', axis=1)
y_valid = valid['Attrition_Flag']


lgb_base = lgb.LGBMClassifier(random_state=0)
lgb_base.fit(X_train, y_train, eval_set=(X_valid, y_valid), eval_metric=scoring)

# # scoring = ['precision', 'recall', 'auc']
# # scores = cross_validate(lgb_base, X, y, cv=5, scoring=scoring, return_train_score=False)
# # print(scores)



TbestprAuc = 0 
#number of training itereation for the model
iterations = 100 #100
# of different peramaters to test one the data
for i in range(100): #100
    try:
        d_train = lgb.Dataset(X_train, label=y_train) #Load in data
        params = {} #initialize parameters
        params['learning_rate'] = np.random.uniform(0, 1) #0
        params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss']) #  _type
        params['objective'] = 'binary' #  binary
        params['metric'] = 'scoring' # accuracy scoring
        params['sub_feature'] = np.random.uniform(0, 1) #4
        params['colsample_bytree'] = np.random.uniform(0, 1) #5
        params['num_leaves'] = np.random.randint(20, 300) #6
        params['max_depth'] = np.random.randint(5, 200) #7
        lgb_baseTuned = lgb.train(params, d_train, iterations)
        y_scoreT=lgb_baseTuned.predict(X_valid)
        precisionT, recallT, thresholds = precision_recall_curve(y_valid, y_scoreT)
        TprAuc = auc(recallT, precisionT)
        if TprAuc>TbestprAuc:
            bestTprAuc = TprAuc
            bestParams = params
    except: #in case something goes wrong
            print('failed with')
            print(params)


boostingtype = []
boostingtype.append(bestParams['learning_rate'])
boostingtype.append(bestParams['boosting_type'])
boostingtype.append(bestParams['objective'])
boostingtype.append(bestParams['metric'])
boostingtype.append(bestParams['sub_feature'])
boostingtype.append(bestParams['colsample_bytree'])
boostingtype.append(bestParams['num_leaves'])
boostingtype.append(bestParams['max_depth'])

#creating a new classifier with the new parameters
lgb_baseTuned = lgb.LGBMClassifier(boosting_type=boostingtype[1], num_leaves=boostingtype[6], max_depth=boostingtype[7], learning_rate=boostingtype[0], colsample_bytree=boostingtype[5],objective=boostingtype[2], metric=boostingtype[3], sub_feature=boostingtype[4], random_state=0) #boosting_type=bestParams[1], num_leaves=bestParams[6], max_depth= bestParams[7], learning_rate=bestParams[0], colsample_bytree=bestParams[5]

acc = []
accT = []
precisionL = []
aucs = []
precisionLT = []
aucsT = []
Mrecall = np.linspace(0, 1, 100)
MrecallT = np.linspace(0, 1, 100)
x = 0

# 20 times 5 fold corss evaluation 
for i in range(20):
    for j in range(5):
        for k in range(5):
            lgb_base.fit(allTrainX[j][k], allTrainY[j][k], eval_set=(X_valid, y_valid), eval_metric=scoring)
            lgb_baseTuned.fit(allTrainX[j][k], allTrainY[j][k], eval_set=(X_valid, y_valid), eval_metric=scoring)
            y_score=lgb_base.predict(X_valid)
            y_scoreT=lgb_baseTuned.predict(X_valid)
            accuracy=accuracy_score(y_valid, y_score)
            accuracyT=accuracy_score(y_valid, y_scoreT)
            precision, recall, thresholds = precision_recall_curve(y_valid, y_score)
            precisionT, recallT, thresholds = precision_recall_curve(y_valid, y_scoreT)
            acc.append(accuracy)
            accT.append(accuracyT)
            precisionL.append(interp(Mrecall, precision, recall))
            prAuc = auc(recall, precision)
            aucs.append(prAuc)
            precisionLT.append(interp(MrecallT, precisionT, recallT))
            TprAuc = auc(recallT, precisionT)
            aucsT.append(TprAuc)
            plt.plot(recall, precision, lw=3, alpha=0.5, label='Fold %d (AUCPR = %0.2f)' % (x+1, prAuc))
            plt.plot(recallT, precisionT, lw=3, alpha=0.5, label='Fold %d (AUCPRT = %0.2f)' % (x+1, TprAuc))
            x += 1
        


Mprecision = np.mean(precisionL, axis=0)
Mauc = auc(Mrecall, Mprecision)
std_auc = np.std(aucs)

bestAUC = 0
bestAUCT = 0
for i in range(len(aucs)):
    if bestAUC < aucs[i]:
        bestAUC = aucs[i]
    if bestAUCT < aucsT[i]:
        bestAUCT = aucsT[i]

acc = np.array(acc)
accT = np.array(accT)
meanAcc = np.mean(acc)
meanAccT = np.mean(accT)
STDacc = np.std(acc)
STDaccT = np.std(accT)

print("Average Accuracy:", meanAcc)
print("Average Tunned Accuracy:", meanAccT)
print("STD of Accuracy:", STDacc)
print("Tunned STD of Accuracy:", STDaccT)
print("Percision Recall AUC:", bestAUC)
print("Tunned Percision Recall AUC:", bestAUCT)


print("")
print(bestParams)
































