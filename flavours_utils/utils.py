# This script contains functions used for data loading, feature engineering, and saving predictions
# It also contains a stacking function, used to obtain meta-features for the second stage

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from flavours_utils import paths


path_train = paths.DATA_TRAIN_PATH
path_test=paths.DATA_TEST_PATH
path_sample_submission=paths.DATA_SUBMISSION_PATH


def add_features(df, model):  # some feature engineering which I adopted from competition's public scripts
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    if (model==2):          # add one more feature to Model2
        df['NEW_IP_dira']=df['IP']*df['dira']
    df['NEW_FD_SUMP']=df['FlightDistance']/(df['p0_p']+df['p1_p']+df['p2_p'])
    df['NEW5_lt']=df['LifeTime']*(df['p0_IP']+df['p1_IP']+df['p2_IP'])/3
    df['p_track_Chi2Dof_MAX'] = df.loc[:, ['p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof']].max(axis=1)
    return df
    

def LoadData(model):  # model = {1,2} is the model namber, since feature engineering is slightly different for model # 2
    print("Load the training/test data")
    train = pd.read_csv(path_train)
    test  = pd.read_csv(path_test)
    train = add_features(train, model)
    test = add_features(test, model)
    # eliminate some features which make the agreement test fail
    filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits','p0_track_Chi2Dof','CDF1',
                  'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p',
                  'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']
    features = list(f for f in train.columns if f not in filter_out)
    return train, test, features                 

def save_submission(predictions):
    test = pd.read_csv(path_test)
    ids = test.id.values
    submission = pd.DataFrame({"id": ids, "prediction": predictions})
    submission.to_csv(path_sample_submission, index=False)


def StackModels(train, test, y, clfs, n_folds): # train data (pd data frame), test data (pd date frame), Target data,
                                                # list of models to stack, number of folders

# StackModels() performs Stacked Aggregation on data: it uses n different classifiers to get out-of-fold 
# predictions for target data. It uses the whole training dataset to obtain signal predictions for test.
# This procedure adds n meta-features to both train and test data (where n is number of models to stack).

    print("Generating Meta-features")
    skf = list(StratifiedKFold(y, n_folds))
    training = train.as_matrix()
    testing = test.as_matrix()
    scaler = StandardScaler().fit(training)
    train_all = scaler.transform(training)
    test_all = scaler.transform(testing)
    blend_train = np.zeros((training.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((testing.shape[0], len(clfs)))   # Number of testing data x Number of classifiers
    
    for j, clf in enumerate(clfs):
        
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = training[tr_index]
            Y_train = y[tr_index]
            X_cv = training[cv_index]
            scaler=StandardScaler().fit(X_train)
            X_train=scaler.transform(X_train)
            X_cv=scaler.transform(X_cv)
                                  
            clf.fit(X_train, Y_train)
            pred = clf.predict_proba(X_cv)
            
            if pred.ndim==1:  # XGBoost produces ONLY probabilities of success as opposed to sklearn models
                 
                 blend_train[cv_index, j] = pred
                 
            else:
                
                blend_train[cv_index, j] = pred[:, 1]
        
        print('stacking test data')        
        clf.fit(train_all, y)
        pred = clf.predict_proba(test_all)
        
        if pred.ndim==1 :      # XGBoost produces ONLY probabilities of success as opposed to sklearn models
        
           blend_test[:, j] = pred
           
        else:
            
           blend_test[:, j] = pred[:, 1]

    X_train_blend=np.concatenate((training, blend_train), axis=1)
    X_test_blend=np.concatenate((testing, blend_test), axis=1)
    return X_train_blend, X_test_blend, blend_train, blend_test
