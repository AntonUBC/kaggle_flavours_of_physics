# This is a wrapper for a XGBoost classifier 

from sklearn.base import BaseEstimator
import xgboost as xgb


class XGBoostClassifier(BaseEstimator):
    def __init__(self, nthread, eta,
                 gamma, max_depth, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, silent, seed,
                 l2_reg, l1_reg, n_estimators):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.n_estimators=n_estimators
        self.model = None

    def fit(self, X, y):
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'binary:logistic',
          "eta": self.eta,
          "gamma": self.gamma,
          "max_depth": self.max_depth,
          "min_child_weight": self.min_child_weight,
          "max_delta_step": self.max_delta_step,
          "subsample": self.subsample,
          "silent": self.silent,
          "colsample_bytree": self.colsample_bytree,
          "seed": self.seed,
          "lambda": self.l2_reg,
          "alpha": self.l1_reg}
        self.model = xgb.train(params, sf, self.n_estimators)
        return self

    def predict_proba(self, X):
        X=xgb.DMatrix(X)
        preds = self.model.predict(X)
        return preds