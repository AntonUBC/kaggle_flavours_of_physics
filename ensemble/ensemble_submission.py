# This script trains two separate models and ensembles their predictions
# to construct the vector of predicted probabilities for the final submission

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from hep_ml.gradientboosting import UGradientBoostingClassifier
from hep_ml.losses import BinFlatnessLossFunction

# load custom modules
from flavours_utils import utils, paths
from wrappers import models


def Model1():
    
# Model 1 is an ensemble of XGBoost, Random Forest and Uniform Gradient Boosting Classifiers
# which are trained using the stacked data    
    
    model = 1    # set the model number for feature engineering
    n_folds = 3 # set the number of folders for generating meta-features
    n_stack = 15  # number of models used for stacking
    
    train, test, features = utils.LoadData(model)  # load data and obtain the list of features for estimation
    
    # Initialize models for stacking
        
    clf1=KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None)
                          
    clf2=KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, 
                              p=2, metric='minkowski', metric_params=None)
                          
    clf3=KNeighborsClassifier(n_neighbors=20, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None)  
                          
    clf4=KNeighborsClassifier(n_neighbors=40, weights='uniform', algorithm='auto', leaf_size=30, 
                              p=2, metric='minkowski', metric_params=None)
                          
    clf5=KNeighborsClassifier(n_neighbors=80, weights='uniform', algorithm='auto', leaf_size=30, 
                              p=2, metric='minkowski', metric_params=None) 

    clf6=KNeighborsClassifier(n_neighbors=160, weights='uniform', algorithm='auto', leaf_size=30,  
                              p=2, metric='minkowski', metric_params=None)

    clf7=KNeighborsClassifier(n_neighbors=320, weights='uniform', algorithm='auto', leaf_size=30,
                              p=2, metric='minkowski', metric_params=None)                          
                          
    clf8=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=5.0, fit_intercept=True,
                            intercept_scaling=1, class_weight=None, random_state=101, solver='lbfgs', 
                            max_iter=200, multi_class='ovr', verbose=0) 
                        
    clf9=GaussianNB()
                 
    clf10=SVC(C=5.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.008, shrinking=True, probability=True, 
              tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=101)
               
    clf11=RandomForestClassifier(n_estimators=250, criterion='gini', max_depth=6, min_samples_split=2, 
                            min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=0.7, 
                            max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=2,
                            random_state=101, verbose=0, warm_start=False, class_weight=None) 
                            
    clf12=ExtraTreesClassifier(n_estimators=250, criterion='gini', max_depth=6, min_samples_split=2,
                     min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=0.7,
                     max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=2, 
                     random_state=101, verbose=0, warm_start=False, class_weight=None)

    clf13=GradientBoostingClassifier(loss='deviance', learning_rate=0.2, n_estimators=450, subsample=0.7, 
                                min_samples_split=2, min_samples_leaf=5, min_weight_fraction_leaf=0.0,
                                max_depth=6, init=None, random_state=101, max_features=None, verbose=0,
                                max_leaf_nodes=None, warm_start=False)
                                
    clf14=SGDClassifier(loss='log', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,
                        n_iter=10, shuffle=True, verbose=0, epsilon=0.1, n_jobs=2, random_state=101, 
                        learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                        average=False) 

    clf15=models.XGBoostClassifier(nthread=2, eta=.2, gamma=0, max_depth=6, min_child_weight=3, max_delta_step=0,
                         subsample=0.7, colsample_bytree=0.7, silent =1, seed=101,
                         l2_reg=1, l1_reg=0, n_estimators=450)
                         
                               
    clfs = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13, clf14, clf15]    
        
    # Construct stacked datasets
    train_blend, test_blend, train_probs, test_probs = utils.StackModels(train[features], test[features], 
                                                                         train.signal.values, clfs, n_folds)                                                                                      
    
    # Construct data for uniform boosting
    columns = ['p%s ' % (i) for i in range(0, n_stack)]
    meta_train = pd.DataFrame({columns[i]: train_probs[:, i] for i in range(0, n_stack)})
    meta_test = pd.DataFrame({columns[i]: test_probs[:, i] for i in range(0, n_stack)})
    train_ugb = pd.concat([train, meta_train], axis=1)
    test_ugb = pd.concat([test, meta_test], axis=1)
    features_ugb = features + columns               # features used for UGB training (original features + meta-features)

    # Initialize models for ensemble
    loss = BinFlatnessLossFunction(['mass'], n_bins=20, power=1, fl_coefficient=3, uniform_label=0)
                                   
    clf_ugb = UGradientBoostingClassifier(loss=loss, n_estimators=275, max_depth=11, min_samples_leaf=3, 
                            learning_rate=0.03, train_features=features_ugb, subsample=0.85, random_state=101)  
                            
    clf_xgb = models.XGBoostClassifier(nthread=6, eta=.0225, gamma=1.225, max_depth=11, min_child_weight=10, 
                                max_delta_step=0, subsample=0.8, colsample_bytree=0.3,  
                                silent =1, seed=101, l2_reg=1, l1_reg=0, n_estimators=1100)
                                
    clf_rf = RandomForestClassifier(n_estimators=375, criterion='gini', max_depth=10, min_samples_split=6, 
                                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=0.6, 
                                max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=4,
                                random_state=101, verbose=0, warm_start=False, class_weight=None)

    # Train models
    print("Training a Uniform Gradient Boosting model")     
    clf_ugb.fit(train_ugb[features_ugb + ['mass']], train_ugb['signal'])   
    preds_ugb = clf_ugb.predict_proba(test_ugb[features_ugb])[:,1]
    
    print("Training a XGBoost model")     
    clf_xgb.fit(train_blend, train['signal'])
    preds_xgb = clf_xgb.predict_proba(test_blend)
        
    print("Training a Random Forest model") 
    clf_rf.fit(train_blend, train['signal'])
    preds_rf = clf_rf.predict_proba(test_blend)[:,1]
        
    # Compute ensemble predictions
    preds = 0.3*(preds_xgb**(0.65))*(preds_rf**(0.35)) + 0.7*preds_ugb
    
    return preds


def Model2():
    
# Model 2 is a single XGBoost classifier "undertrained" to reduce correlation with tau-mass       
        
    model = 2    # set the model number for feature engineering
                                                         
    train, test, features = utils.LoadData(model)    # load data
    
    # Initialize a XGBoost model
    clf_xgb = models.XGBoostClassifier(nthread=6, eta=0.75, gamma=1.125, max_depth=8, min_child_weight=5, 
                                max_delta_step=0, subsample=0.7, colsample_bytree=0.7, silent=1, seed=1, 
                                l2_reg=1, l1_reg=0, n_estimators=50)                                
                              
    # Train a XGBoost model                                                                   
    print("Training a XGBoost model")  
    clf_xgb.fit(train[features], train['signal'])
   
    # Calculate predictions
    preds = clf_xgb.predict_proba(test[features])
    return preds

print("Training Model1")    
preds_model1 = Model1()         # compute predictions of Model1

print("Training Model2")
preds_model2 = Model2()         # compute predictions of Model2

# compute final predictions for submission  
preds_ensemble = (preds_model1**0.585) * (preds_model2**0.415)

#Save the submission file   
utils.save_submission(preds_ensemble)  