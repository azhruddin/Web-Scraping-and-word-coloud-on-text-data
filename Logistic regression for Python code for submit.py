### Logistic regression Assignment
## Sol for 1st question
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

## Loading the dataset
Affairs = pd.read_csv("C:\\Users\\Dell\\Desktop\\Assighnments\\M-9-Logestic_reg\\Handson\\Affairs.csv")

Affairs.isna().sum() ## No NA's present in the dataset

Affairs.describe()
Affairs = Affairs.drop(["Unnamed: 0"], axis = 1)  ## Removing the columns which is not related dataset
Affairs.colnames()

## Converting the output variable into binary format
Affairs['naffairs'] = (Affairs.naffairs > 0).astype(int)

## bulding the model using the logistic regression
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = Affairs).fit()
model.summary()
model.summary2()
## Prediction for model
pred = model.predict(Affairs.iloc[ :, 1: ])

## Finding cutoff
from sklearn import metrics
fpr, tpr, thresholds = roc_curve(Affairs.naffairs, pred)
optimal_idx = np.argmax(tpr - fpr)
cutoff = thresholds[optimal_idx]
cutoff
## cut off = 0.252157

## Ploting ROC curve
import pylab as pl
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

## Finding Arreaundcurve
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
## Area under the ROC curve : 0.720880

# filling all the cells with zeroes
Affairs["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
Affairs.loc[pred > cutoff, "pred"] = 1

## Confision Matrics
confusion_matrix = pd.crosstab(Affairs.pred, Affairs['naffairs'])
confusion_matrix

## Accuracy for the model
accuracy_test = (318 + 98)/(601) 
accuracy_test
## Accuracy = 0.692179

# classification report
classification = classification_report(Affairs["pred"], Affairs["naffairs"])
classification



### Splitting the data into train and test data 

train_data, test_data = train_test_split(Affairs, test_size = 0.4) # 40% test data

# Model building 
# import statsmodels.formula.api as sm
model_t = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model_t.summary2() # for AIC
model_t.summary()

# Prediction on Test data set
test_pred = model_t.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(241)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > cutoff, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (109 + 40)/(241) 
accuracy_test
## Accuracy = 0.6182572

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

## Area under curve
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
## Area under curve =  0.64597

# prediction on train data
train_pred = model_t.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(360)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > cutoff, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (185 + 64)/(360)
print(accuracy_train)
## accuracy_train = 0.69166

## The train data accuracy and test data accuracy is similar so we approving the model.

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\
## Sol for 2nd question

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


bank = pd.read_csv("C:/Users/Dell/Desktop/Assighnments/M-9-Logestic_reg/Handson/bank_data.csv")

bank.isna().sum() ## No NA's presnt in the dataset
bank.columns
## Renaming columns names
bank.columns = ['age', 'default', 'balance', 'housing', 'loan', 'duration', 'campaign', 'pdays', 'previous', 'poutfailure', 'poutother', 'poutsuccess', 'poutunknown', 'con_cellular', 'con_telephone', 'con_unknown', 'divorced', 'married', 'single', 'joadmin', 'jobluecollar', 'joentrepreneur', 'johousemaid', 'jomanagement', 'joretired', 'joselfemployed', 'joservices', 'jostudent', 'jotechnician', 'jounemployed', 'jounknown', 'y']

bank['y'] = (bank.y).astype(int)

## Rearranging columns names
bank = bank.iloc[:, [31, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]]

## Buielding the model using logistic regression
model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + joselfemployed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank).fit()
model.summary2() ## for AIS
model.summary

## prediction for model
pred = model.predict(bank.iloc[ :, 1: ])
pred

## Finding cutoff
from sklearn import metrics
fpr, tpr, thresholds = roc_curve(bank.y, pred)
optimal_idx = np.argmax(tpr - fpr)
cutoff = thresholds[optimal_idx]
cutoff
## cut off = 0.1147422078

## Ploting ROC curve
import pylab as pl
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

## Finding Arreaundcurve
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
## Area under the ROC curve : 0.890843

# filling all the cells with zeroes
bank["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
bank.loc[pred > cutoff, "pred"] = 1

## Confision Matrics
confusion_matrix = pd.crosstab(bank.pred, bank.y)
confusion_matrix

## Accuracy for the model
accuracy_test = (32805 + 4303)/(45211) 
accuracy_test
## Accuracy = 0.82077370
# classification report
classification = classification_report(bank.pred, bank.y)
classification
## We build the model and going to perform on train and test data

## partitioning the data
train_data, test_data = train_test_split(bank, test_size = 0.4) 

## model
model_t = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + joselfemployed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = train_data).fit()
model.summary2()
model.summary()

## predicting on test data
pred_test = model_t.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(18085)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[pred_test > cutoff, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data.y)
confusion_matrix

accuracy_test = (12982 + 1715)/(18085) 
accuracy_test
## Accuracy = 0.81266242

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], pred_test)
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

## Area under curve
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
## Area under curve =  0.8915706

# prediction on train data
train_pred = model_t.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(27126)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > cutoff, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (19642 + 2606)/(27126)
print(accuracy_train)
## accuracy_train = 0.820172

### The accuracy of the train and test data is similar, so the model is valid.

####\/\/\//\\\//\//\/\\\\\\//\/\\/\//\\/\/\/\/\//\\//\\//\/\/\\/\/\/\/\//\\\\\/\\//\/\\/\/\/\//\\//\/\\/\\\/\/\
## Sol for 3rd Question

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import patsy as dmatrices
import socketserver
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report


Election = pd.read_csv("C:/Users/Dell/Desktop/Assighnments/M-9-Logestic_reg/Handson/election_data.csv")
Election.describe()
Election.columns

# To drop NaN values
Election = Election.dropna()

Election = Election.drop(["Election-id"], axis = 1)
Election.columns = ['Result', 'Year', 'amount', 'popularity']
Election.isna().sum()

## Filling Na's with mean for continues data
Election.Year = Election.Year.fillna(Election.Year.mean())
Election.amount = Election.amount.fillna(Election.amount.mean())
Election.popularity = Election.popularity.fillna(Election.popularity.mean())

## Filling the NA's with mode for catagorical data
Election.Result = Election.Result.fillna(Election.Result.mode())

Election['Result'] = (Election.Result).astype(int)

X_data = Election[['Year', 'amount', 'popularity']]
y_data = Election['Result']

## Buelding the model using logistic regression
logreg = LogisticRegression()
logreg.fit(X_data,y_data)

logreg.summary()

## Prerdiction on model 
pred = logreg.predict(X_data)
pred

## Finding the error

## Finding cutoff
from sklearn import metrics
fpr, tpr, thresholds = roc_curve(y_data, pred)
optimal_idx = np.argmax(tpr - fpr)
cutoff = thresholds[optimal_idx]
cutoff
## Cutoff is 1


## Ploting ROC curve
import pylab as pl
i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

## Finding Arreaundcurve
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
## Area under the ROC curve : 0.875000

# filling all the cells with zeroes
X_data["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
X_data.loc[pred > cutoff, "pred"] = 1
## Confision Matrics
confusion_matrix = pd.crosstab(pred, y_data)
confusion_matrix

## Accuracy for the model
accuracy_test = (3 + 6)/(10) 
accuracy_test
## Accuracy = 0.9
# classification report
classification = classification_report(pred, y_data)
classification

## partitioning the data
X_train,X_test,y_train,y_test=train_test_split(X_data,y_data,test_size=0.20,random_state=0)

## model
logreg_t = LogisticRegression()
logreg_t.fit(X_train,y_train)

## predicting on test data
pred_test = logreg_t.predict(X_test)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
X_test["pred_test"] = np.zeros(2)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
X_test.loc[pred_test > cutoff, "pred_test"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(X_test.pred_test, y_test)
confusion_matrix

accuracy_test = (0 + 2)/(2) 
accuracy_test
## Accuracy = 1

# classification report
classification_test = classification_report(X_test["pred_test"], y_test)
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(y_test, pred_test)
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

## Area under curve
roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
## Area under curve =  0.8915706

# prediction on train data
pred_train = logreg_t.predict(y_train)


# Creating new column 
# filling all the cells with zeroes
train_data["pred_train"] = np.zeros(8)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > cutoff, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(y_train.train_pred, y_test)
confusion_matrx

accuracy_train = (1 + 6)/(8)
print(accuracy_train)
## accuracy_train = 0.875
### The accuracy of the train and test data is similar, so the model is valid.