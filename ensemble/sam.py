import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
# def ignore_warn(*args, **kwargs): pass
# ignore annoying warning (from sklearn and seaborn)
# warnings.warn = ignore_warn
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


class StackingAveragedModels(BaseEstmator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # we again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models = [list() for x in self.base_models]
        self.meta_model = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shffle=True, random_state=156)

        # The k-fold method is used for cross-validation, and the results of
        # each validation are treated as new features
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)self.base_models_[i].append(instance)
                instance.fit(X[train_index],  y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 将交叉验证预测出的结果 和 训练集中的标签值进行训练
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # 从得到的新的特征  采用新的模型进行预测  并输出结果
    def predict(self, X):
        meta_features = np.column_stack ([
            np.column_stack([model.predict (X) for model in base_models]).mean (axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
"""
主要过程简述如下:
    1、首先需要几个模型，然后对已有的数据集进行K折交叉验证
    2、K折交叉验证训练集，对每折的输出结果保存，最后进行合并
    3、对于测试集T1的得到，有两种方法。注意到刚刚是2折交叉验证，M1相当于训练了2次，
       所以一种方法是每一次训练M1，可以直接对整个test进行预测，这样2折交叉验证后测试
       集相当于预测了2次，然后对这两列求平均得到T1。
    4、是两层循环，第一层循环控制基模型的数目，第二层循环控制的是交叉验证的次数K，
       对每一个基模型会训练K次，然后拼接得到预测结果P1。
"""

stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), 
    # meta_model=model_lgb)
    meta_model=lasso)

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, 
        train.values, y_train, scoring="neg_mean_squared_error", cv=kf))
return(rmse)

score = rmse_cv(stacked_averaged_models) 
print "Stacking Averaged Models score: {:.4f} ({:.4f})".format(
    score.mean(), score.std())  

rmsle = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
