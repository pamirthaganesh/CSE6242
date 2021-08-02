from sklearn.feature_selection import SelectKBest, mutual_info_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error,confusion_matrix, precision_score,recall_score,f1_score, plot_confusion_matrix


'''
input: school dataset
function: seperate rows based on response variable whether it is NaN or not
output: 
      * 1. top_portion: portion with response variable not NaN
      * 2. bot_portion: portion with response variable as NaN
      * 3. X = predictors to train the model
      * 4. y = response variable to train the model
      # 4. x_predict: preditors of school data that has response variable as NaN which is equivalent to no quality info given

'''
def seperate_modelandtest(data):
    top_portion = data[data['Rate'].notnull()]
    bot_portion = data[data['Rate'].isnull()]
    top_portion = top_portion.reset_index(drop=True)
    bot_portion = bot_portion.reset_index(drop=True)
    
    X = top_portion.drop(columns=['Rate','dbn','school_name'])
    y = top_portion[['Rate']]
    x_predict = bot_portion.drop(columns=['Rate','dbn','school_name'])

    X = X.astype('float')
    x_predict = x_predict.astype('float')

    return X,y,x_predict,top_portion,bot_portion

'''
input: xtrain and ytrain (train data), threshold 
function: selects which features are influential in classifying
output: columns of most influential feature above threshold
''' 
def select_features(xtrain, ytrain,threshold=0.04):
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    fs.fit(xtrain, np.ravel(ytrain))
    cols = fs.get_support(indices=True)
    newcol = []
    for i in range(len(fs.scores_)):
        if fs.scores_[i]>threshold:
            newcol.append(xtrain.columns[cols][i])
            # print('Selected Feature {} {}: {}'.format(i,X.columns[cols][i], round(fs.scores_[i],4)))
    # plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    # plt.title('Mutual Feature Selection')
    # plt.xlabel('Features')
    # plt.ylabel('Importance')
    # plt.show()
    return newcol

'''
input: predicted value and actual value
function: if predicted value falls within one point difference, we gave it as a correct mark since we do not have sufficient train data
output: new predicted value that falls into the interval of one difference
''' 
def interval(pred,test):
    for i in range(len(pred)):
        if (float(abs(pred[i]-test.iloc[i]))) in range(2):
            pred[i] = test.iloc[i]
    return pred

'''
input: x_train,y_train,x_test,y_test,random_state,n_estimator,max_dep,max_features
function: prefore hyperparameter tuning to fine the optimal parameters for Random Forest
output: Optimal parameters which includes n_estimator, max_dep and max_features
''' 
# Hyperparameter Grid Search
def hyperparameter(x_train,y_train,x_test,y_test,random_state,n_estimator,max_dep,max_features):
    RF = RandomForestClassifier(random_state=random_state)
    RFfit = RF.fit(x_train,np.ravel(y_train))
    param_grid = dict(n_estimators = n_estimator, max_depth = max_dep, max_features = max_features)
    grid = GridSearchCV(estimator = RFfit, param_grid = param_grid, cv = 7)
    grid_result = grid.fit(x_train, np.ravel(y_train))
    pred = grid.predict(x_test).round()
    pred = interval(pred,y_test)
    # grid_accuracy = accuracy_score(y_test,pred)
    # print('Best Params:', grid_result.best_params_)
    # print('Best Score: ', grid_accuracy)
    return grid_result.best_params_['max_depth'], grid_result.best_params_['max_features'], grid_result.best_params_['n_estimators']

'''
input: x_train,y_train,x_test,y_test,x_predict,random_state,n_estimators,max_depth,max_features
function: predict school quality values using Random Forest with the optimal parameters
output: Predicted values for schools that had originally no labeled data
'''
def estimate(x_train,y_train,x_test,y_test,x_predict,random_state,n_estimators,max_depth,max_features):
    RF = RandomForestClassifier(random_state=random_state)
    RFfit = RF.fit(x_train,np.ravel(y_train))
    pred = RFfit.predict(x_test).round()
    pred = interval(pred,y_test)
    accuracy = accuracy_score(y_test,pred)
    qualities = RFfit.predict(x_predict).round()
    return qualities, accuracy

'''
input: xdata,ydata,random_state,max_depth,n_estimators
function: performs 5-fold cross validation, generate train and test accuracy, confusion box, compute macro precision, recall
output: train and test accuracy, confusion box, compute macro precision, recall
'''
def Kfold(xdata,ydata,random_state,max_depth,n_estimators):
    train_accuracy =[]
    test_accuracy = []
    ConfusionBox = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
    cv = StratifiedKFold(n_splits=5,shuffle=False)

    for train,test in cv.split(xdata,ydata):
        xtrain = xdata.iloc[train]
        ytrain = ydata.iloc[train]
        xtest = xdata.iloc[test]
        ytest = ydata.iloc[test]

        RF= RandomForestClassifier(random_state=random_state,max_depth = max_depth,n_estimators=n_estimators)
        RF.fit(xtrain,np.ravel(ytrain))
        ypred = RF.predict(xtest).round()
        ypred = interval(ypred,ytest)
        test_accuracy.append(accuracy_score(ytest,ypred))
        train_accuracy.append(RF.score(xtrain,ytrain))
        
        CM = confusion_matrix(ytest,ypred)
        ConfusionBox += CM

    count = 0
    precision = []
    recall = []
    for row in ConfusionBox:
        precision.append(row[count]/sum(ConfusionBox[:,count]))
        recall.append(row[count]/sum(row))
        count += 1
        macroprecision = sum(precision)/10
        macrorecall = sum(recall)/10
    return np.mean(train_accuracy), np.mean(test_accuracy), ConfusionBox, macroprecision, macrorecall

    





