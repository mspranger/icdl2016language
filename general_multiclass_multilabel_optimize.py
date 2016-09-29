# -*- coding: utf-8 -*-
import numpy
import pickle
import sklearn.base
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.grid_search
import sklearn.metrics
import sklearn.mixture
import sklearn.multiclass
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree
import warnings

import description_game
import nearest_centroid

def optimize( clf, param_grid, X_train, X_test, y_train, y_test, n_jobs = 16, scoring = 'f1_samples'):
    # Split the dataset in two equal parts
    print("# Tuning hyper-parameters for %s with %s" % (clf, (param_grid, n_jobs, scoring)))
    print()

    search = sklearn.grid_search.GridSearchCV( clf, param_grid, scoring = scoring, n_jobs = n_jobs)
    search.fit( X_train, y_train)

    print("Best parameters set found on development set")
    print()
    print( search.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in search.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % ( mean_score, scores.std() * 2, params))
    print()
    y_true, y_pred = y_test, search.predict( X_test)
    print( "Best estimator test: %s" % description_game.compute_f_scores( y_true, y_pred))
    # print( sklearn.metrics.classification_report(y_true, y_pred))
    # print( sklearn.metrics.precision_recall_fscore_support( y_true, y_pred, average = "samples"))
    # returns the best estimator
    
    return( sklearn.base.clone( search.best_estimator_), search, description_game.compute_f_scores( y_true, y_pred))

def optimize_all( nr_words = 100, 
                 nr_words_per_utterance = 5, 
                 nr_dimensions = 17, 
                 nr_samples = 4532,
                 X_train = None,
                 y_train = None,
                 X_test = None,
                 y_test = None,
                 n_jobs = 16,
                 file_name = "optimized_classifiers.pickle"):
    if X_train == None:
        X = numpy.random.uniform( size = ( nr_samples, nr_dimensions))
        y_cat, y_bin = description_game.compute_tutor_weighted( X, nr_words, nr_words_per_utterance)
        y = y_bin
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split( X, y, test_size = 0.5, random_state = 0)
    
    tasks = [ 
    # optimize SGD
    ( "SGD", sklearn.multiclass.OneVsRestClassifier( sklearn.linear_model.SGDClassifier( eta0 = 0.1)),
    { 'estimator__loss' : ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
      'estimator__penalty' : ['none', 'l2', 'l1', 'elasticnet'],
      'estimator__learning_rate' : ['constant', 'invscaling', 'optimal'],
      'estimator__eta0' : [ 0.001, 0.01, 0.1, ],
      'estimator__learning_rate' : ['constant', 'invscaling', 'optimal'], 
      'estimator__alpha' : [  0.00001, 0.0001, 0.001,  0.01,  0.1],
      'estimator__fit_intercept' : [ True, False],
      'estimator__epsilon' : [ .001, .01, .1],
      'estimator__class_weight' : ['auto', None]}),
    # optimize PassiveAggressive
    ( "PassiveAgressive", sklearn.multiclass.OneVsRestClassifier( sklearn.linear_model.PassiveAggressiveClassifier()),
     { 'estimator__C' : [ 0.01, 0.1, 1.0],
       'estimator__fit_intercept' : [ True, False],
       'estimator__n_iter' : [ 5, 10, 50, 100]}),
    # optimize LogisticRegression
    ( "LogisticRegression", sklearn.multiclass.OneVsRestClassifier( sklearn.linear_model.LogisticRegression()),
     { 'estimator__penalty' : [ 'l2', 'l1' ],
       'estimator__C' : [ 0.01, 0.1, 1.0],
       'estimator__class_weight' : ['auto', None]}),
    #     NEVER optimize Bagging it is too slow in execution
    #    ( "Bagging", sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.BaggingClassifier( sklearn.neighbors.KNeighborsClassifier())),
    #     { 'estimator__n_estimators' : [ 10, 50, 100],
    #       'estimator__max_samples' : [ 0.1, 0.5, 1.0],
    #       'estimator__max_features' : [ 0.1, 0.5, 1.0],
    #       'estimator__bootstrap' : [ True, False],
    #       'estimator__bootstrap_features' : [ True, False],
    #       'estimator__n_jobs' : [16]}),
    # optimize RandomForest
    ( "RandomForest", sklearn.ensemble.RandomForestClassifier(),
     { 'n_estimators' : [ 10, 50, 100],
       'criterion' : ["gini", "entropy"],
       'max_features' : [ "auto", 0.1, 0.5, 1.0],
       'max_depth' : [ None, 10, 50, 100, 1000],
       'class_weight' : ['auto', 'subsample', None]}),
    # optimize ExtraTrees
    ( "ExtraTrees", sklearn.ensemble.ExtraTreesClassifier(),
     { 'n_estimators' : [ 10, 50, 100],
       'criterion' : ["gini", "entropy"],
       'max_features' : [ "auto", 0.1, 0.5, 1.0],
       'max_depth' : [ None, 10, 50, 100, 1000],
       'class_weight' : ['auto', 'subsample', None]}),
    # optimize AdaBoost
    ( "AdaBoost", sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.AdaBoostClassifier()),
     { 'estimator__n_estimators' : [ 10, 50, 100, 1000],
       'estimator__learning_rate' : [ .1, .5, 1.0]}),
    # optimize GradientBoosting
    ( "GradientBoosting", sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.GradientBoostingClassifier()),
     { 'estimator__loss' : [ 'deviance', 'exponential'],
       'estimator__learning_rate' : [ .01, .1, .5, 1.0],
       'estimator__n_estimators' : [ 10, 50, 100, 1000],
       'estimator__max_depth' : [ 1, 2, 3, 5, 10]}),
    # optimize MultinomialNB
    ( "MultinomialNB", sklearn.multiclass.OneVsRestClassifier( sklearn.naive_bayes.MultinomialNB()),
     { 'estimator__alpha' : [ 0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
       'estimator__fit_prior' : [ True, False]}),
    # optimize GMM
    ( "GMM", sklearn.multiclass.OneVsRestClassifier( sklearn.mixture.GMM()),
     { 'estimator__n_components' : [ 1, 5, 10, 20, 50, 100],
       'estimator__covariance_type' : [ "spherical", "tied", "diag", "full"]})
       ]

    best_classifiers = {}
    try: 
        best_classifiers = pickle.load( open( file_name, "rb"))
    except:
        pass
    
    # add GaussianNB, nearest neighbor and neares centroid without optimization
    best_classifiers["NearestNeighbor"] = sklearn.neighbors.KNeighborsClassifier()
    best_classifiers["NearestCentroid"] = nearest_centroid.NearestCentroidOvR()
    best_classifiers["GaussianNB"] = sklearn.multiclass.OneVsRestClassifier( sklearn.naive_bayes.GaussianNB())
    pickle.dump( best_classifiers, open( file_name, "wb"))

    for name, clf, param_grid in tasks:
        try:
            best_clf = optimize( clf, param_grid, X_train, X_test, y_train, y_test, n_jobs = n_jobs)[0]
            best_classifiers = {}
            try: 
                best_classifiers = pickle.load( open( file_name, "rb"))
            except:
                pass
            best_classifiers[ name] = best_clf
            pickle.dump( best_classifiers, open( file_name, "wb"))
        except:
            pass
    
if __name__ == "__main__":
    import argparse
    warnings.filterwarnings( "ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument( '--optimize-all',
                    action = "store_true",
                    dest = "optimize_all",
                    default = False,
                    help = "")

    parser.add_argument( '--nr-words',
                    type = int,
                    action = "store",
                    dest = "nr_words",
                    default = 100,
                    help = "")
    parser.add_argument( '--nr-words-per-utterance',
                    action = "store",
                    dest = "nr_words_per_utterance",
                    type = int,
                    default = 5,
                    help = "")
    parser.add_argument( '--nr-dimensions',
                    action = "store",
                    dest = "nr_dimensions",
                    type = int,
                    default = 17,
                    help = "")
    parser.add_argument( '--nr-samples',
                    action = "store",
                    dest = "nr_samples",
                    default = 4532,
                    type = int,
                    help = "")
    parser.add_argument( '--file-name',
                    action = "store",
                    dest = "file_name",
                    default = "optimized_classifiers.pickle",
                    help = "")
    parser.add_argument( '--n-jobs',
                    action = "store",
                    dest = "n_jobs",
                    default = 16,
                    type = int,
                    help = "")  
    args = parser.parse_args()

    print( "Running with %s" % args)
    if args.optimize_all:
        optimize_all( nr_words = args.nr_words,
                     nr_words_per_utterance = args.nr_words_per_utterance, 
                     nr_dimensions = args.nr_dimensions, 
                     nr_samples = args.nr_samples,
                     file_name = args.file_name,
                     n_jobs = args.n_jobs)