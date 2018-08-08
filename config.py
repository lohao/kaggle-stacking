# Put in our parameters for said classifiers
# Random Forest parameters
rf_params = {
    'n_jobs': -1,
    'n_estimators': 110,
     # 'warm_start': True,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_split': 2,
    'max_features': 'sqrt',
    'verbose': 0,
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':50,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

#lr parameters
lr_params = {
    'C': 1e5
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 50,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
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