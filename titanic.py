import pandas as pd
import numpy as np
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import helper
import config

import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

SEED = 0


def main():

    #['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    train_df_ori = pd.read_csv('data/train.csv')
    test_df_ori = pd.read_csv('data/test.csv')

    train_df, test_df = fea_eng(train_df_ori, test_df_ori)

    # colormap = plt.cm.RdBu
    # plt.figure(figsize=(14, 12))
    # plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(train_df.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
    # plt.show()

    # Some useful parameters which will come in handy later on
    rf = helper.SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=config.rf_params)
    et = helper.SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=config.et_params)
    # ada = helper.SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=config.ada_params)
    lr = helper.SklearnHelper(clf=LogisticRegression, seed=SEED, params=config.lr_params)
    gb = helper.SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=config.gb_params)
    svc = helper.SklearnHelper(clf=SVC, seed=SEED, params=config.svc_params)

    y_train = train_df['Survived'].ravel()
    train_df = train_df.drop(['Survived'], axis=1)
    x_train = pd.get_dummies(train_df).values
    x_test = pd.get_dummies(test_df).values

    # Create our OOF train and test predictions. These base results will be used as new features
    et_oof_train, et_oof_test = helper.get_oof(et, x_train, y_train, x_test)  # Extra Trees
    rf_oof_train, rf_oof_test = helper.get_oof(rf, x_train, y_train, x_test)  # Random Forest
    # ada_oof_train, ada_oof_test = helper.get_oof(ada, x_train, y_train, x_test)  # AdaBoost
    lr_oof_train, lr_oof_test = helper.get_oof(lr, x_train, y_train, x_test)  # AdaBoost
    gb_oof_train, gb_oof_test = helper.get_oof(gb, x_train, y_train, x_test)  # Gradient Boost
    svc_oof_train, svc_oof_test = helper.get_oof(svc, x_train, y_train, x_test)  # Support Vector Classifier

    print("Training is complete")

    base_predictions_train = pd.DataFrame({'RandomForest': rf_oof_train.ravel(),
                                           'ExtraTrees': et_oof_train.ravel(),
                                           # 'AdaBoost': ada_oof_train.ravel(),
                                           'GradientBoost': gb_oof_train.ravel()
                                           })

    x_train = np.concatenate((et_oof_train, rf_oof_train, lr_oof_train, gb_oof_train, svc_oof_train), axis=1)
    x_test = np.concatenate((et_oof_test, rf_oof_test, lr_oof_test, gb_oof_test, svc_oof_test), axis=1)

    xgb_helper = helper.SklearnHelper(clf=xgb.XGBClassifier, seed=SEED, params=config.xgb_params)
    xgb_helper.train(x_train, y_train)
    predictions = xgb_helper.predict(x_test)

    # 交叉验证，可以快速得到模型的预测正确率
    scores = cross_val_score(rf.clf, x_train, y_train, cv=5)
    print("rf Accuracy: {}".format(scores))
    scores = cross_val_score(et.clf, x_train, y_train, cv=5)
    print("et Accuracy: {}".format(scores))
    # scores = cross_val_score(ada.clf, x_train, y_train, cv=5)
    # print("ada Accuracy: {}".format(scores))
    scores = cross_val_score(lr.clf, x_train, y_train, cv=5)
    print("lr Accuracy: {}".format(scores))
    scores = cross_val_score(gb.clf, x_train, y_train, cv=5)
    print("gb Accuracy: {}".format(scores))
    scores = cross_val_score(svc.clf, x_train, y_train, cv=5)
    print("svc Accuracy: {}".format(scores))
    scores = cross_val_score(xgb_helper.clf, x_train, y_train, cv=5)
    print("stacking xgb Accuracy: {}".format(scores))

    helper.plot_learning_curve(xgb_helper.clf, 'learn curve', x_train, y_train)

    result = pd.DataFrame({"PassengerId": test_df_ori['PassengerId'],
                           "Survived": predictions
                           })

    result.to_csv("data/submission.csv", index=False)


def fea_eng(train, test):
    combine = [train, test]

    #根据是否有舱位作为一个特征
    for data in combine:
        data['Has_Cabin'] = data['Cabin'].apply(lambda x: 0 if x is not np.nan else 1)

    #将家庭人数做为一个特征
    # Feature engineering steps taken from Sina
    # Create new feature FamilySize as a combination of SibSp and Parch
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    #是否单身作为一个特征
    # Create new feature IsAlone from FamilySize
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    #选择最多的岸口填充null值
    # Remove all NULLS in the Embarked column
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

    #
    # Define function to extract titles from passenger names
    # Create a new feature Title, containing the titles of passenger names
    for dataset in combine:
        dataset['Title'] = dataset['Name'].apply(get_title)
    # Group all non-common titles into one single grouping "Rare"
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Dona', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    #根据Title对应的年龄均值作为填充值
    title_age_dict = combine[0].groupby('Title').Age.mean().to_dict()
    for dataset in combine:
        age_null_list = dataset.loc[np.isnan(dataset['Age']), 'Title'].apply(lambda x: title_age_dict[x])
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_list

    train['CategoricalAge'] = pd.cut(train['Age'], 5)

    #选择使用均值填充null值
    # Remove all NULLS in the Fare column and create a new feature CategoricalFare
    for dataset in combine:
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
    # Create a New feature CategoricalAge
    # for dataset in combine:
    #     age_avg = dataset['Age'].mean()
    #     age_std = dataset['Age'].std()
    #     age_null_count = dataset['Age'].isnull().sum()
    #     age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    #     dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    #     dataset['Age'] = dataset['Age'].astype(int)


    for dataset in combine:
        # Mapping Sex
        dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping Embarked
        dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        dataset = helper.get_onehot(dataset, ['Embarked', 'Sex'])

        # Mapping Fare
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # Mapping Age
        dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

        # Feature selection
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Embarked', 'Sex']
    train = train.drop(drop_elements, axis=1)
    train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
    test = test.drop(drop_elements, axis=1)

    return train, test


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


if __name__ == '__main__':
    main()