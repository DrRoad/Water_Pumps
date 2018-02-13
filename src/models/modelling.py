# Hide deprecation warnings
import warnings
warnings.filterwarnings('ignore')

# Common imports
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing

# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# To format floats
from IPython.display import display
pd.set_option('display.float_format', lambda x: '%.5f' % x)

def model_selection(X_train, X_test, df_labels):
    y_train = df_labels.status_group.values

    # Compare models without optimization
    models = {
    "Dumb Model": AlwaysFunctionalClassifier(),
    "SGD Classifier": SGDClassifier(),
    "Random Forests": RandomForestClassifier(),
    "k-Nearest Neighbors": KNeighborsClassifier(),
    "Softmax Regression": LogisticRegression(multi_class="multinomial",solver="lbfgs"),
    "SVM": SVC(decision_function_shape="ovr"),
    "Decission Trees": DecisionTreeClassifier(),
    "AdaBoost":AdaBoostClassifier(algorithm="SAMME.R"),
    "Gradient Boost":GradientBoostingClassifier()
    }

    results = []
    names = []

    for k, v in models.items():
        cv_scores = cross_val_score(estimator=v, X=X_train, y=y_train, cv=10, n_jobs=1, scoring='accuracy')

        results.append(cv_scores)
        names.append(k)

        print(k)
        print('CV accuracy: %.3f +/- %.3f' % (np.mean(cv_scores), np.std(cv_scores)))
        print('----------------')

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    # Let's try to optimize some of this models
    # Random Forests

    # Initial performance
    forest_clf = RandomForestClassifier()
    cross_val_score(forest_clf, X_train, y_train, cv=3, scoring="accuracy")

    # Random Forests Confusion Matrix
    y_train_pred = cross_val_predict(forest_clf, X_train, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_mx, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            perc = str(round((conf_mx[i, j]/conf_mx.sum())*100,2)) + "%"
            ax.text(x=j, y=i, s=str(conf_mx[i, j])+"\n\n"+perc, va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

    param_grid = [{'max_depth': [ 30, 60],
                   'n_estimators': [ 80, 300],
                   'max_features': [5, 10],
                   'min_samples_leaf': [1, 10],
                   'n_jobs': [-1]}]

    grid_search_rf = GridSearchCV(forest_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    cvres = grid_search_rf.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

    print(grid_search_rf.best_params_)

    cv_results = cross_validate(RandomForestClassifier(**grid_search_rf.best_params_), \
                                X_train, y_train, cv = 3, scoring="accuracy")

    print(cv_results['test_score'].mean())

    # SGD Classifier
    # Initial performance
    sgd_clf = SGDClassifier()
    cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

    # SGD Confusion Matrix
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_mx, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            perc = str(round((conf_mx[i, j]/conf_mx.sum())*100,2)) + "%"
            ax.text(x=j, y=i, s=str(conf_mx[i, j])+"\n\n"+perc, va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

    param_grid = [{'penalty': ['none', 'l2','l1','elasticnet'],
                   'alpha': [ 0.00001,  0.0001,  0.001,  0.01],
                   'loss': ['log'],
                   'n_jobs': [-1]}]

    grid_search_sgd = GridSearchCV(sgd_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_sgd.fit(X_train, y_train)

    cvres = grid_search_sgd.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

    print(grid_search_sgd.best_params_)

    cv_results = cross_validate(SGDClassifier(**grid_search_sgd.best_params_), \
                                X_train, y_train, cv = 3, scoring="accuracy")

    print(cv_results['test_score'].mean())

    # K Nearest Neighbors
    # Initial performance

    knn_clf = KNeighborsClassifier()
    cross_val_score(knn_clf, X_train, y_train, cv=3, scoring="accuracy")

    # KNN Confusion Matrix
    y_train_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(conf_mx, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            perc = str(round((conf_mx[i, j]/conf_mx.sum())*100,2)) + "%"
            ax.text(x=j, y=i, s=str(conf_mx[i, j])+"\n\n"+perc, va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')

    plt.tight_layout()
    plt.show()

    param_grid = [{'n_neighbors': [ 3, 5, 10],
                   'weights': [ 'uniform',  'distance'],
                   'n_jobs': [-1]}]

    grid_search_knn = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_knn.fit(X_train, y_train)

    cvres = grid_search_knn.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

    print(grid_search_knn.best_params_)

    cv_results = cross_validate(KNeighborsClassifier(**grid_search_knn.best_params_), \
                                X_train, y_train, cv = 3, scoring="accuracy")

    print(cv_results['test_score'].mean())

    # Classification with XGBoost

    param_grid = [{'max_depth': [ 3, 10],
                   'n_estimators': [ 80, 300],
                   'learning_rate': [0.01, 0.1, 0.3]}]

    gbm = xgb.XGBClassifier()
    grid_search_xgb = GridSearchCV(gbm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_xgb.fit(X_train, y_train)

    cvres = grid_search_xgb.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(mean_score, params)

    print(grid_search_xgb.best_params_)

    cv_results = cross_validate(xgb.XGBClassifier(**grid_search_xgb.best_params_), \
                                X_train, y_train, cv = 3, scoring="accuracy")

    print(cv_results['test_score'].mean())

    # Just a bit better than Random Forests, but the best so far nevertheless.

    # Ensembling
    # Let's put together all the models shown above to see if we get a better result.
    sgd_clf = SGDClassifier(**grid_search_sgd.best_params_)
    rnd_clf = RandomForestClassifier(**grid_search_rf.best_params_)
    knn_clf = KNeighborsClassifier(**grid_search_knn.best_params_)
    log_clf = LogisticRegression(multi_class="multinomial",solver="lbfgs",C= 30, n_jobs=-1)
    # We'll skip SVM as they slow down too much the modelling times
    # svm_clf = SVC(C= 1, gamma= 0.1, decision_function_shape="ovr", n_jobs=-1)
    dtr_clf = DecisionTreeClassifier(max_depth= 20, min_samples_split= 10)
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=200, algorithm="SAMME.R",
                                 learning_rate=0.5)
    gbrt_clf = GradientBoostingClassifier(max_depth=5, n_estimators=500, learning_rate=0.5)
    xgb_clf = xgb.XGBClassifier(**grid_search_xgb.best_params_)

    clfs = [sgd_clf,rnd_clf,knn_clf,log_clf,dtr_clf,ada_clf,gbrt_clf,xgb_clf]

    voting_clf_ens_soft = VotingClassifier(
        estimators=[('SGD Classifier', clfs[0]),
                    ('Random Forests', clfs[1]),
                    ('k-Nearest Neighbors', clfs[2]),
                    ('Softmax Regression', clfs[3]),
                    ('Decission Trees', clfs[4]),
                    ('AdaBoost', clfs[5]),
                    ('Gradient Boost', clfs[6]),
                    ('XGBoost', clfs[7])],
                    voting='soft', n_jobs=-1)
    voting_clf_ens_soft.fit(X_train, y_train)

    cv_results = cross_validate(voting_clf_ens_soft, X_train, y_train, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())

    # Although slower, it doesn't seem to be a better model than just Random Forests optimized alone, is it probably the soft voting? Let's see
    voting_clf_ens_hard = VotingClassifier(
        estimators=[('SGD Classifier', clfs[0]),
                    ('Random Forests', clfs[1]),
                    ('k-Nearest Neighbors', clfs[2]),
                    ('Softmax Regression', clfs[3]),
                    ('Decission Trees', clfs[4]),
                    ('AdaBoost', clfs[5]),
                    ('Gradient Boost', clfs[6]),
                    ('XGBoost', clfs[7])],
                    voting='hard', n_jobs=-1)
    voting_clf_ens_hard.fit(X_train, y_train)

    cv_results = cross_validate(voting_clf_ens_hard, X_train, y_train, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())
    # Doesn't change much.

    # Stacking
    # Let's create a new model that decides the final label in a new second layer, taking as input the results of all the previous models.
    print(X_train.shape)
    idx = np.random.permutation(len(X_train))  # create shuffle index

    ## split into three sets
    # training set
    Xtr = X_train[idx[:33000]]
    ytr = y_train[idx[:33000]]
    # validation set
    Xvl = X_train[idx[33000:46200]]
    yvl = y_train[idx[33000:46200]]
    # test set
    Xts = X_train[idx[46200:]]
    yts = y_train[idx[46200:]]

    print(Xtr.shape, Xvl.shape, Xts.shape)
    for i, clf in enumerate(clfs):
        clf.fit(Xtr, ytr)
        print("Fitted {}/{}".format(i+1,len(clfs)))

    # run individual classifiers on val set
    yhat = {}
    for i, clf in enumerate(clfs):
        yhat[i] = clf.predict(Xvl)
        print("Predicted {}/{}".format(i+1,len(clfs)))

    # create new training set from predictions
    # combine the predictions into vectors using a horizontal stacking
    Xblend = np.c_[[preds for preds in yhat.values()]].T

    #Transform labels into codes
    le = preprocessing.LabelEncoder()
    Xblend = le.fit_transform(Xblend.reshape(13200*8)).reshape(13200,8)

    # train a random forest classifier on Xblend using yvl for target labels
    rf_blend = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf_blend.fit(Xblend, yvl)

    cv_results = cross_validate(rf_blend, Xblend, yvl, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())

    # Let's see how this behaves with an unseen dataset
    # run individual classifiers on test set
    yhatts = {}
    for i, clf in enumerate(clfs):
        yhatts[i] = clf.predict(Xts)
        print("Predicted {}/{}".format(i+1,len(clfs)))

    Xblendts = np.c_[[preds for preds in yhatts.values()]].T

    Xblendts = le.transform(Xblendts.reshape(13200*8)).reshape(13200,8)

    cv_results = cross_validate(rf_blend, Xblendts, yts, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())

    # Finally, in this exercise, nothing beats Random Forests and XGBoost.

    # Ensembling RF and XGB
    rnd_clf = RandomForestClassifier(**grid_search_rf.best_params_)
    xgb_clf = xgb.XGBClassifier(**grid_search_xgb.best_params_)

    clfs = [rnd_clf,xgb_clf]
    voting_clf_ens_rfxgb = VotingClassifier(
        estimators=[('Random Forests', clfs[0]),
                    ('XGBoost', clfs[1])],
                    voting='soft', n_jobs=-1)
    voting_clf_ens_rfxgb.fit(X_train, y_train)

    cv_results = cross_validate(voting_clf_ens_rfxgb, X_train, y_train, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())
    # This is the best result so far!

    # Stacking RF and XGB
    # We have to be specially careful here to not overfit the RF classifier.
    idx = np.random.permutation(len(X_train))  # create shuffle index

    ## split into three sets
    # training set
    Xtr = X_train[idx[:33000]]
    ytr = y_train[idx[:33000]]
    # validation set
    Xvl = X_train[idx[33000:46200]]
    yvl = y_train[idx[33000:46200]]
    # test set
    Xts = X_train[idx[46200:]]
    yts = y_train[idx[46200:]]

    print(Xtr.shape, Xvl.shape, Xts.shape)

    for i, clf in enumerate(clfs):
        clf.fit(Xtr, ytr)
        print("Fitted {}/{}".format(i+1,len(clfs)))

    # run individual classifiers on val set
    yhat = {}
    for i, clf in enumerate(clfs):
        yhat[i] = clf.predict(Xvl)
        print("Predicted {}/{}".format(i+1,len(clfs)))

    # create new training set from predictions
    # combine the predictions into vectors using a horizontal stacking
    Xblend = np.c_[[preds for preds in yhat.values()]].T

    #Transform labels into codes
    le = preprocessing.LabelEncoder()
    Xblend = le.fit_transform(Xblend.reshape(13200*2)).reshape(13200,2)

    # train a random forest classifier on Xblend using yvl for target labels
    rf_blend = RandomForestClassifier(n_estimators=300, n_jobs=-1)
    rf_blend.fit(Xblend, yvl)

    cv_results = cross_validate(rf_blend, Xblend, yvl, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())

    # Let's see how this behaves with an unseen dataset
    # run individual classifiers on test set
    yhatts = {}
    for i, clf in enumerate(clfs):
        yhatts[i] = clf.predict(Xts)
        print("Predicted {}/{}".format(i+1,len(clfs)))

    Xblendts = np.c_[[preds for preds in yhatts.values()]].T

    Xblendts = le.transform(Xblendts.reshape(13200*2)).reshape(13200,2)

    cv_results = cross_validate(rf_blend, Xblendts, yts, cv = 3, scoring="accuracy")
    print(cv_results['test_score'].mean())

    # Finally, it seems that the best result were obtained with an RF and XGBoost ensemble. Let's use this model to make the final predictions and submission file creation.
    return voting_clf_ens_rfxgb



# Let's create a dumb model that always predict the most common label
class AlwaysFunctionalClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return ['functional'] * len(X)
