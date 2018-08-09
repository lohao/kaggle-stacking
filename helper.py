from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
import pandas as pd
import threading
import time


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x, y)

    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_


class GridSearchThread(threading.Thread):
    def __init__(self, clf, params, X, y):
        threading.Thread.__init__(self)
        self.clf = clf
        self.params = params
        self.X = X
        self.y = y

    def run(self):
        import warnings
        warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

        start_time = time.time()

        grid_clf = GridSearchCV(estimator=self.clf, param_grid=self.params, cv=5)

        grid_clf.fit(self.X, self.y)

        print(grid_clf.best_params_)
        print('{} spend time is {}s'.format(str(self.clf).split('(')[0], (time.time()-start_time)))
        print('='*40)

def get_oof(clf, x_train, y_train, x_test):
    """获取stacking第二层训练器的输入数据"""

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    SEED = 0  # for reproducibility
    NFOLDS = 5  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=True)

    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(np.arange(ntrain))):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_tr_test = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_tr_test)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


import pydotplus  # you can install pydotplus with: pip install pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz


def print_graph(clf, feature_names):
    """Print decision tree. not support random forest"""
    graph = export_graphviz(
        clf,
        label="root",
        proportion=True,
        impurity=False,
        out_file=None,
        feature_names=feature_names,
        class_names={0: "D", 1: "R"},
        filled=True,
        rounded=True
    )
    graph = pydotplus.graph_from_dot_data(graph)
    return Image(graph.create_png())


import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"train samples")
        plt.ylabel(u"scores")
        # plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"scores of train sample")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"scores of cv sample")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        # plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def get_onehot(df, columns):
    for column in columns:
        onehot = pd.get_dummies(df[column], prefix=column)
        df = pd.concat([df, onehot], axis=1)

    return df


def add_missing_dummy_columns(d, columns):
    missing_cols = set(columns) - set(d.columns)
    for c in missing_cols:
        d[c] = 0


#增加和column一样的到test数据中，多余的column为0
def fix_columns(d, columns):
    add_missing_dummy_columns(d, columns)

    # make sure we have all the columns we need
    assert (set(columns) - set(d.columns) == set())

    extra_cols = set(d.columns) - set(columns)
    if extra_cols:
        print("extra columns:{}".format(extra_cols))

    d = d[columns]
    return d


if __name__ == '__main__':
    n_cols = 4
    n_rows = 5

    columns = ["col_{}".format(x) for x in range(n_cols)]

    # create the "new" set of columns
    new_columns = columns[:]  # copy
    new_columns.pop()
    new_columns.append('col_new')

    # create the "new" dataframe
    n = np.random.random((n_rows, n_cols))
    d = pd.DataFrame(n, columns=new_columns)

    fixed_d = fix_columns(d.copy(), columns)

    assert (list(fixed_d.columns) == columns)
