# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:21:14 2020

@author: HP LAPTOP

pareto charts and 
learning curve
p-value
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import sklearn
import random

from sklearn.datasets import  make_classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc

def mse_loss(a , b):
    loss = np.average((a-b)*(a-b))
    return loss

def pareto_chart(y , y_predicted , str1):
    plt.figure()
    fig, ax = plt.subplots()
    ax.scatter(y, y_predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.title(str1)
    plt.show()

def regression_score(classfier , x_trst , y_tr):
    y_pred = classfier.predict(x_trst)
    r_score = r2_score(y_tr , y_pred)
    return r_score

def metrics_of_accuracy(classifier , X_train , y_train) :
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    accuracies.mean()
    accuracies.std()
    return accuracies

def learning_curve_plus(classifier , x_trst , y_tr , x_test , y_te):
    interval = 10 ;
    y_tr = y_tr.reshape((np.shape(y_tr)[0],))
    fig_mean = plt.figure(1)
    fig2_std = plt.figure(2)
    fig3_learning = plt.figure(3)
    plt.figure(2) ; plt.title('std v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('std of accuracies');
    plt.figure(1) ; plt.title('mean v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('mean of accuracies');
    plt.figure(3) ; plt.title('Learning Curve - mse v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('MSE');

    mean_mat = np.zeros(interval) ;
    std_mat = np.zeros(interval)
    learning_mat = np.zeros(interval)
    learning_mat_train = np.zeros(interval)
    
    for i in range(1,(interval+1)):
        x_trial = x_trst[0:int(np.shape(x_trst)[0]/interval*i) , :]
        y_trial = y_tr[0:int(np.shape(y_tr)[0]*i/interval)]
        classifier.fit(x_trial , y_trial)
        accuracies = metrics_of_accuracy(classifier , x_trial , y_trial)
        std_mat[i-1] = accuracies.std() ;
        mean_mat[i-1] = accuracies.mean() ;
        learning_mat[i-1] = mse_loss(y_te , classifier.fit(x_trial , y_trial).predict(x_test)) ;
        learning_mat_train[i-1] =  mse_loss(y_trial , classifier.fit(x_trial , y_trial).predict(x_trial)) ;
        
    
    plt.figure(2) ;
    plt.plot(range(interval) , std_mat , c = 'b')
    plt.figure(1) ;
    plt.plot(range(interval) , mean_mat , c = 'b')
    plt.figure(3) ; 
    plt.plot(range(interval) , learning_mat , label = 'test set' , color='b')
    plt.plot(range(interval) , learning_mat_train , label = 'training set' ,  color='r')
    plt.legend()
    fig_mean.show()
    fig2_std.show()
    
    
def learning_curve(classifier , x_trst , y_tr , x_test , y_te):
    interval = 10 ;
    y_tr = y_tr.reshape((np.shape(y_tr)[0],))
    fig3_learning = plt.figure(3)
    plt.figure(3) ; plt.title('Learning Curve - mse v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('MSE');

    learning_mat = np.zeros(interval)
    learning_mat_train = np.zeros(interval)
    
    for i in range(1,(interval+1)):
        x_trial = x_trst[0:int(np.shape(x_trst)[0]/interval*i) , :]
        y_trial = y_tr[0:int(np.shape(y_tr)[0]*i/interval)]
        classifier.fit(x_trial , y_trial.reshape(np.shape(y_trial)[0],))
        learning_mat[i-1] = mse_loss(y_te , classifier.fit(x_trial , y_trial).predict(x_test)) ;
        learning_mat_train[i-1] =  mse_loss(y_trial , classifier.fit(x_trial , y_trial).predict(x_trial)) ;
        
    x_axis =np.array( range(1,interval+1))*100
    plt.figure(3) ; 
    plt.plot(x_axis , learning_mat , label = 'test set' , color='b')
    plt.plot(x_axis , learning_mat_train , label = 'training set' ,  color='r')
    plt.legend()
    plt.show()
    

    

def learning_curve_r2(classifier , x_trst , y_tr , x_test , y_te):
    interval = 10 ;
    y_tr = y_tr.reshape((np.shape(y_tr)[0],))
    fig3_learning = plt.figure(3)
    plt.figure(4) ; plt.title('Learning Curve - r2_score v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('r2_score');

    learning_mat = np.zeros(interval)
    learning_mat_train = np.zeros(interval)
    
    for i in range(1,(interval+1)):
        x_trial = x_trst[0:int(np.shape(x_trst)[0]/interval*i) , :]
        y_trial = y_tr[0:int(np.shape(y_tr)[0]*i/interval)]
        classifier.fit(x_trial , y_trial)
        learning_mat[i-1] = r2_score(y_te , classifier.fit(x_trial , y_trial).predict(x_test)) ;
        learning_mat_train[i-1] =  r2_score(y_trial , classifier.fit(x_trial , y_trial).predict(x_trial)) ;
        
    x_axis =np.array( range(1,interval+1))*100
    plt.figure(4) ; 
    plt.plot(x_axis , learning_mat , label = 'test set' , color='b')
    plt.plot(x_axis , learning_mat_train , label = 'training set' ,  color='r')
    plt.legend()
    plt.show()
    
def learning_curve_r2(classifier , x_trst , y_tr , x_test , y_te):
    interval = 10 ;
    y_tr = y_tr.reshape((np.shape(y_tr)[0],))
    fig3_learning = plt.figure(3)
    plt.figure(4) ; plt.title('Learning Curve - r2_score v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('r2_score');

    learning_mat = np.zeros(interval)
    learning_mat_train = np.zeros(interval)
    
    for i in range(1,(interval+1)):
        x_trial = x_trst[0:int(np.shape(x_trst)[0]/interval*i) , :]
        y_trial = y_tr[0:int(np.shape(y_tr)[0]*i/interval)]
        classifier.fit(x_trial , y_trial)
        learning_mat[i-1] = r2_score(y_te , classifier.fit(x_trial , y_trial).predict(x_test)) ;
        learning_mat_train[i-1] =  r2_score(y_trial , classifier.fit(x_trial , y_trial).predict(x_trial)) ;
        
    x_axis =np.array( range(1,interval+1))*100
    plt.figure(4) ; 
    plt.plot(x_axis , learning_mat , label = 'test set' , color='b')
    plt.plot(x_axis , learning_mat_train , label = 'training set' ,  color='r')
    plt.legend()
    plt.show()

  
def learning_curve_r2_all(LinReg , SVM , DTReg , XGB , x_trst , y_tr , x_test , y_te):
    interval = 10 ;
    classifiers = [LinReg , SVM , DTReg , XGB]
    y_tr = y_tr.reshape((np.shape(y_tr)[0],))
    fig3_learning = plt.figure(3)
    plt.figure(4) ; plt.title('Learning Curve - r2_score v/s training data size') ; plt.xlabel('training data size'); plt.ylabel('r2_score');
    learning_mat = np.zeros(interval)
    learning_mat_train = np.zeros(interval)
    names = ['Linear Regression' , 'SVM' , 'Decision Tree' , 'XGBoost']
    x_axis =np.array( range(1,interval+1))*100
    for i in range(4):
        classifier = classifiers[i] ;
        str1 = names[i]
        for i in range(1,(interval+1)):
            x_trial = x_trst[0:int(np.shape(x_trst)[0]/interval*i) , :]
            y_trial = y_tr[0:int(np.shape(y_tr)[0]*i/interval)]
            classifier.fit(x_trial , y_trial)
            learning_mat[i-1] = r2_score(y_te , classifier.fit(x_trial , y_trial).predict(x_test)) ;
        plt.figure(4) ;
        plt.plot(x_axis , learning_mat , label = str1 , alpha = 0.7)
        
    plt.legend()
    plt.show()
    

def roc_cur_multiclass(y_te , y_p_xg_1):
    plt.figure()
    y_p_xg_1 = np.around(y_p_xg_1)
    fpr, tpr, _ = roc_curve(y_te <4   , y_p_xg_1 <4)
    roc_auc = auc(fpr , tpr)
    plt.plot(fpr , tpr , label="ROC curve (area = '{0}') for no. of snaking = <4".format( round(roc_auc,2)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    fpr, tpr, _ = roc_curve( np.logical_and((y_te <8),  (y_te>3))   , np.logical_and((y_p_xg_1 <8) , (y_p_xg_1 >3)) )
    roc_auc = auc(fpr , tpr)
    plt.plot(fpr , tpr , label="ROC curve (area = '{0}') for no. of snaking = 3-7".format( round(roc_auc,2) ))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    fpr, tpr, _ = roc_curve(y_te >7   , y_p_xg_1 >7)
    roc_auc = auc(fpr , tpr)
    plt.plot(fpr , tpr , label="ROC curve (area = '{0}') for no. of snaking = >7".format( round(roc_auc,2)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# getting the datset
train_df = pd.read_csv('Snaking_June_2020 (1).csv') ;
train_df = train_df.iloc[: , 1:8 ]
x = train_df.iloc[: , 0:4].values
y1 = train_df.iloc[: , 4].values
y2 = train_df.iloc[: , 5].values
y3 = train_df.iloc[: , 6].values

# making an train_test split
from sklearn.model_selection import train_test_split
x_tr , x_te , y_tr , y_te = train_test_split(x , y1 , test_size = 0.2 , random_state = random.randint(10,100) )
output = 1 ;


'''
x_trst = x_tr
x_test = x_te
y_tr = y1
'''
#standardization
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler() ;
sc_X = sc_X.fit(x_tr[: , :]) ;
x_trs = sc_X.transform(x_tr[: , :]) ;
x_tes = sc_X.transform(x_te[: , :]) ;

x_trst = x_trs
x_test = x_tes


# trying it on linear reg
from sklearn.linear_model import LinearRegression
LinR = LinearRegression()
LinR.fit(x_trst , y_tr)
y_lr = LinR.predict(x_trst)
y_p_lr = LinR.predict(x_test)

'''
if (output == 1):    
    roc_cur_multiclass(y_te , y_p_lr )
else:
    pareto_chart(y_te , y_p_lr , "Linear Regression")
    
drawing a lurning curve for the linear model
learning_curve_r2(LinR , x_trst , y_tr , x_test , y_te)
learning_curve(LinR , x_trst , y_tr , x_test , y_te)
'''

l_lr = mse_loss(y_te , y_p_lr)
acc_rf = metrics_of_accuracy(LinR , x_trst , y_tr)




# making an SVM classifier
sc_y = StandardScaler()
y_tr = sc_y.fit_transform(y_tr.reshape(-1,1))
from sklearn.svm import SVR
SVReg = SVR(C =  7, gamma = 0.01, kernel=  'rbf' )






#Applying grid search 
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
classifier = SVReg ;
parameters = [
              {'C': [  1,3,5,7 ], 
               'kernel': ['rbf'], 
               'gamma': [0.001,0.005,0.01]
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = regression_score,
                           cv = 10,
                           n_jobs = 4)
grid_search = grid_search.fit(x_trst , y_tr )
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

SVReg.parameters = best_parameters
SVReg.fit(x_trs , y_tr)
y_sv = sc_y.inverse_transform( SVReg.predict(x_trs))
y_p_sv = sc_y.inverse_transform(SVReg.predict(x_tes))
y_tr = sc_y.inverse_transform(y_tr)

l_sv = mse_loss(y_te , y_p_sv)
l_t_sv = mse_loss(y_tr , y_sv)

'''
if (output == 1):
    roc_cur_multiclass(y_te , np.around(y_p_sv) )
else:
    pareto_chart(y_te , y_p_sv , "SVM")


drawing a learning curve for the support vector machines
learning_curve(SVReg , x_trst , y_tr , x_test , y_te)
learning_curve_r2(SVReg , x_trst , y_tr , x_test , y_te)
'''

#xgboost
import xgboost as xgb
XGBReg=xgb.XGBRegressor(max_depth = 2 , learning_rate = 0.05 ,  n_estimators = 100 , reg_lambda = 10 , reg_alpha = 10)


from sklearn.model_selection import GridSearchCV
classifier = XGBReg ;
parameters = [
              {'max_depth': [  2,3,4 ], 
               'n_estimators': [100], 
               'learning_rate': [0.05,0.07,0.09],
               'reg_alpha': [0.5 ,1,5],
               'reg_lambda':[5,10,50]
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = regression_score,
                           cv = 5,
                           n_jobs = 4)
grid_search = grid_search.fit(x_trst , y_tr)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

XGBReg.parameters = best_parameters 
XGBReg.fit(x_trst, y_tr)
y_xg = XGBReg.predict(x_trst)
y_p_xg = XGBReg.predict(x_test)
l_xg_2 = mse_loss(y_te , y_p_xg)
l_t_xg_2 = mse_loss(y_tr , y_xg)


'''
if (output == 1):
    roc_cur_multiclass(y_te, y_p_xg)
else:
    pareto_chart(y_te , y_p_xg , "Number of Snaking - Pareto Chart")


learning_curve(XGBReg , x_trst , y_tr , x_test , y_te)
learning_curve_r2(XGBReg , x_trst , y_tr , x_test , y_te)
'''

#xgboost classifier
import xgboost as xgb
XGBCla=xgb.XGBClassifier(max_depth = 5)


from sklearn.model_selection import GridSearchCV
classifier = XGBCla ;
parameters = [
              {'max_depth': [  2,3,5 ]
              
               
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = regression_score,
                           cv = 3,
                           n_jobs = 4)
grid_search = grid_search.fit(x_trst , y_tr)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

XGBCla.parameters = best_parameters 
XGBCla.fit(x_trst, y_tr)
y_xg_1 = XGBCla.predict(x_trst)
y_p_xg_1 = XGBCla.predict(x_test)
l_xg_1 = mse_loss(y_te , y_p_xg)
l_t_xg_1 = mse_loss(y_tr , y_xg)

'''
if (output == 1):
    plt.figure()
    for i in range(15):
        fpr, tpr, _ = roc_curve(y_te == i   , y_p_xg_1 == i)
        roc_auc = auc(fpr , tpr)
        plt.plot(fpr , tpr , label="ROC curve (area = '{0}') for no. of snaking = '{1}'".format( round(roc_auc,2) , i))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.show()
    
elif (output == 1):
    
else:
    pareto_chart(y_te , y_p_xg , "Number of Snaking - Pareto Chart- XGBoost CLassifier")



learning_curve(XGBCla , x_trst , y_tr , x_test , y_te)
learning_curve_r2(XGBCla , x_trst , y_tr , x_test , y_te)
'''

















#DEcision tree regressor
from sklearn.tree import DecisionTreeRegressor
DTReg = DecisionTreeRegressor(max_depth = 4 , criterion='mse'  )

'''
#learning_curve(DTReg , x_trst , y_tr , x_test , y_te)
#learning_curve_r2(DTReg , x_trst , y_tr , x_test , y_te)
'''
# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
classifier = DTReg ;
parameters = [
              {'max_depth': [  3,4,5,6 ]
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = regression_score,
                           cv = 10,
                           n_jobs = 4)
grid_search = grid_search.fit(x_trst , sc_y.transform(y_tr).reshape(np.shape(y_tr)[0],))
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

DTReg.fit(x_tr , y_tr)
y_dt = DTReg.predict(x_tr)
y_p_dt = DTReg.predict(x_te)
l_dt_1 = mse_loss(y_te , y_p_dt)
l_t_dt_1 = mse_loss(y_tr , y_dt)

'''
if (output == 1):
    pareto_chart(y_te , np.around(y_p_dt) , "Decision tree")
else:
    pareto_chart(y_te , y_p_dt , "Decision tree")
'''

'''
from sklearn import tree
fig , axes = plt.subplots( dpi =1200)
tree.plot_tree(DTReg)
plt.savefig('tree1.jpg');
'''













#DEcision tree CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn import utils

DTCla = DecisionTreeClassifier(max_depth = 4 , criterion='gini')

#learning_curve(DTCla , x_trst , y_tr , x_test , y_te)
#learning_curve_r2(DTCla , x_trst , y_tr , x_test , y_te)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
classifier = DTCla ;
parameters = [
              {'max_depth': [  3,4,5,6 ]
               }]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = regression_score,
                           cv = 10)
grid_search = grid_search.fit(x_tr , y_tr.astype(int))
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

DTCla.fit(x_trst , y_tr.astype(int))
y_dt = DTCla.predict(x_tr)
y_p_dt = DTCla.predict(x_te)
l_dt_1 = mse_loss(y_te , y_p_dt)
l_t_dt_1 = mse_loss(y_tr , y_dt)

'''
if (output == 1):
    plt.figure()
    for i in range(15):
        fpr, tpr, _ = roc_curve(y_te == i   , y_p_dt == i)
        roc_auc = auc(fpr , tpr)
        plt.plot(fpr , tpr , label="ROC curve (area = '{0}') for no. of snaking = '{1}'".format( round(roc_auc,2) , i))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.show()
    
else:
    pareto_chart(y_te , y_p_dt , "Decision tree")

'''


    
'''
corelation analysis 
'''

'''
import scipy as sp
import scipy.stats as sps

corr_rel_pear = np.zeros(shape = (4,3) , dtype='float')
corr_rel_kend = np.zeros(shape = (4,3) , dtype='float')
corr_rel_spear = np.zeros(shape = (4,3) , dtype='float')

p_val= np.zeros(shape = (4,3) , dtype='float')
for i in range(0,4):
    corr_rel_pear [i , 0] = sp.stats.pearsonr(x[:,i] , y1)[0]
    corr_rel_pear [i , 1] = sp.stats.pearsonr(x[:,i] , y2)[0]
    corr_rel_pear [i , 2] = sp.stats.pearsonr(x[:,i] , y3)[0]
    
    corr_rel_kend [i , 0] = sp.stats.pearsonr(x[:,i] , y1)[0]
    corr_rel_kend [i , 1] = sp.stats.pearsonr(x[:,i] , y2)[0]
    corr_rel_kend [i , 2] = sp.stats.pearsonr(x[:,i] , y3)[0]
    
    corr_rel_spear [i , 0] = sp.stats.pearsonr(x[:,i] , y1)[0]
    corr_rel_spear [i , 1] = sp.stats.pearsonr(x[:,i] , y2)[0]
    corr_rel_spear [i , 2] = sp.stats.pearsonr(x[:,i] , y3)[0]
    
    p_val [i , 0] = sp.stats.pearsonr(x[:,i] , y1)[1]
    p_val [i , 1] = sp.stats.pearsonr(x[:,i] , y2)[1]
    p_val [i , 2] = sp.stats.pearsonr(x[:,i] , y3)[1]
    

# converting into beaituful dataframes
df1 = np.concatenate(( np.ones( (4,1)) , corr_rel_kend) , axis = 1)

df_cor_ken = pd.DataFrame(df1 , columns = [ 'Input Variable' , 'No. of Snaking' , 'Snaking Length' , 'Snaking Amplitude' ]) ;
df_cor_ken.iloc[0,0] = 'Inner Modulus'
df_cor_ken.iloc[1,0] = 'Outer Modulus'
df_cor_ken.iloc[2,0] = 'Axial Load'
df_cor_ken.iloc[3,0] = 'Friction Coefficient'

dfp = np.concatenate(( np.ones( (4,1)) , p_val) , axis = 1)

df_pval = pd.DataFrame(dfp , columns = [ 'Input Variable' , 'No. of Snaking' , 'Snaking Length' , 'Snaking Amplitude' ]) ;
df_pval.iloc[0,0] = 'Inner Modulus'
df_pval.iloc[1,0] = 'Outer Modulus'
df_pval.iloc[2,0] = 'Axial Load'
df_pval.iloc[3,0] = 'Friction Coefficient'

'''

'''
3D rendering of the results
'''

'''
Interpretability
'''

'''
eli5
'''

#POISSON REGRESSION
import statsmodels.api as sm

poisson_training_results = sm.GLM(y_tr , x_trst , family=sm.families.Poisson()).fit()
print(poisson_training_results.summary())

poisson_predictions = poisson_training_results.get_prediction(x_test)

#summary_frame() returns a pandas DataFrame
predictions_summary_frame = poisson_predictions.summary_frame()
print(predictions_summary_frame)

predicted_counts=predictions_summary_frame['mean']
actual_counts = y_te
fig = plt.figure()
fig.suptitle('Predicted versus actual Snaking counts')
predicted, = plt.plot(np.r_[1:250], predicted_counts[1:250], 'go-', label='Predicted counts')
actual, = plt.plot(np.r_[1:250], actual_counts[1:250], 'ro-', label='Actual counts')
plt.legend(handles=[predicted, actual])
plt.show()


#Show scatter plot of Actual versus Predicted counts
plt.clf()
fig = plt.figure()
fig.suptitle('Scatter plot of Actual versus Predicted counts')
plt.scatter(x=predicted_counts, y=actual_counts, marker='.')
plt.xlabel('Predicted counts')
plt.ylabel('Actual counts')
plt.show()

#ZIP regression
zip_training_results = sm.ZeroInflatedPoisson(endog=y_tr, exog=x_tr, exog_infl=x_tr, inflation='logit').fit()
print(zip_training_results.summary())

zip_predictions = zip_training_results.predict(x_te,exog_infl=x_te)
predicted_counts=np.round(zip_predictions)
actual_counts = y_te
print('ZIP RMSE='+str(np.sqrt(np.sum(np.power(np.subtract(predicted_counts,actual_counts),2)))))



#Let’s plot the predicted versus actual fish counts:
fig = plt.figure()
fig.suptitle('Predicted versus actual counts using the ZIP model')
predicted, = plt.plot(np.r_[1:251], predicted_counts, 'go-', label='Predicted')
actual, = plt.plot(np.r_[1:251], actual_counts, 'ro-', label='Actual')
plt.legend(handles=[predicted, actual])
plt.show()

r2_score(predicted_counts , actual_counts)
