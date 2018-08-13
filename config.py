# Put in our parameters for said classifiers
# Random Forest parameters
import numpy as np

rf_params = {
    'n_jobs': -1,
    'n_estimators': 110,
    # 'warm_start': True,
    # 'max_features': 0.2,
    'max_depth': 5,
    'min_samples_split': 2,
    'max_features': 'sqrt',
    'verbose': 0,
}

# 网格搜索示例
rf_grid_params = {'n_estimators': np.arange(30, 200, 20),
                  'max_depth': np.arange(5, 10),
                  'min_samples_split': np.arange(2, 5)
                  }

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators': 110,
    # 'max_features': 0.5,
    'max_depth': 6,
    'min_samples_split': 3,
    'verbose': 0
}

et_grid_params = {
    'n_estimators': np.arange(30, 200, 20),
    'max_depth': np.arange(5, 10),
    'min_samples_split': np.arange(2, 5)
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate': 0.75
}

ada_grid_params = {
    'n_estimators': np.arange(30, 200, 20),
    'learning_rate': np.arange(0.01, 1, 0.05)
}

# lr parameters
lr_params = {
    'C': 100,
    'penalty': 'l1'
}

lr_grid_params = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 70,
    # 'max_features': 0.2,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'verbose': 0,
    'learning_rate': 0.01
}

gb_grid_params = {
    'n_estimators': np.arange(50, 200, 20),
    'learning_rate': [0.01, 0.1],
    'max_depth': np.arange(2, 10),
    'min_samples_split': np.arange(2, 5)
}

# Support Vector Classifier parameters
svc_params = {
    'kernel': 'linear',
    'C': 0.025
}

xgb_params = {
    # learning_rate = 0.02,
    'n_estimators': 30,
    'max_depth': 4,
    'min_child_weight': 2,
    # gamma=1,
    'gamma': 0.9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    'nthread': -1,
    'scale_pos_weight': 1
}

xgb_grid_params = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
