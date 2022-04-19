import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import joblib
from pathlib import Path

path = Path(__file__).parent

class Training:
    def __init__(self, df):
        # self.ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        # file_path = os.path.join(self.ROOT_DIR, 'instance', 'uploads','creditcard.csv')
        #df = pd.read_csv(file_path)
        self.df = df
        self.df = self.df.sample(50000).reset_index(drop=True)
        # define X, y
        self.target = 'Class'
        self.predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                      'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                      'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                      'Amount']

        self.X = self.df[self.predictors]
        self.y = self.df[self.target]

        # split the dataset
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=10)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test



    def run_lgbm(self):
        dtrain = lgb.Dataset(self.X_train.values,
                             label=self.y_train.values,
                             feature_name=self.predictors)

        dvalid = lgb.Dataset(self.X_test.values,
                         label=self.y_test.values,
                         feature_name=self.predictors)


        params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric': 'auc',
              'learning_rate': 0.05,
              'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
              'max_depth': 4,  # -1 means no limit
              'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
              'max_bin': 100,  # Number of bucketed bin for feature values
              'subsample': 0.9,  # Subsample ratio of the training instance.
              'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
              'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
              'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
              'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
              'nthread': 8,
              'verbose': -1,
              'scale_pos_weight': 150,  # because training data is extremely unbalanced
             }


        evals_results = {}

        model = lgb.train(params,
                      dtrain,
                      valid_sets=[dtrain, dvalid],
                      valid_names=['train', 'valid'],
                      evals_result=evals_results,
                      num_boost_round=1000,
                      early_stopping_rounds=2*50,
                      verbose_eval=False,
                      feval=None)

        y_prob = model.predict(self.X_test)
        y_pred = [np.argmax(line) for line in y_prob]

        return accuracy_score(self.y_test, y_pred), roc_auc_score(self.y_test, y_pred), model

    def run_training(self):
        # import dataset

        #     y_prob = model.predict(X_test)
        #     y_pred = [np.argmax(line) for line in y_prob]
        #     model_dict['lightgbm'] = [accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_pred)]

        model_clf_dict = {"Nearest Neighbors": KNeighborsClassifier(),
                 "Decision Tree":DecisionTreeClassifier(),
                 "Random Forest": RandomForestClassifier(),
                 "LGBM" : self.run_lgbm()
                         }


        model_dict = {}
        best_model_dict = {}
        for name in model_clf_dict:
            print('algorithm : ', name)
            if name == 'LGBM':
                score, roc_score, model = self.run_lgbm()


            else:
                model = model_clf_dict[name]
                # traing
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_prob = model.predict_proba(self.X_test)

                # accuracy
                score = accuracy_score(self.y_test, y_pred)
                roc_score = roc_auc_score(self.y_test, y_prob[:, 1])

            model_dict[name] = [score, roc_score]
            best_model_dict[name] = model

        # Comparison of models based on accuracy, and of predicted class

        model_df = pd.DataFrame(model_dict, index=['Accuracy', 'AUC_ROC_score']).T
        model_df = model_df.sort_values(by=['AUC_ROC_score', 'Accuracy'], ascending=False)
        print(model_df)

        best_model_name = model_df.iloc[0].name
        best_model = best_model_dict[best_model_name]

        model_save_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        model_save_file = os.path.join(model_save_dir, 'app', 'model','best_model.pkl')
        joblib.dump(best_model, model_save_file)
        return model_df.to_dict()
