# -*- coding: utf-8 -*-
import datetime
import multiprocessing
import numpy
import pickle
import sklearn.base
import sklearn.ensemble
import sklearn.metrics
import sklearn.mixture
import sklearn.multiclass
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.svm
import sklearn.cross_validation
import warnings

###########################################

def dist_euclidian_weighted( sample, weights, prototypes):
    "Computes a weighted euclidian distance between sample and weighted prototypes"
    sample_repeated = numpy.reshape( numpy.tile( sample, prototypes.shape[0]), ( prototypes.shape[0], prototypes.shape[1]))
    dist = numpy.sqrt( numpy.sum( weights * numpy.power( ( prototypes - sample_repeated), 2), axis = 1))
    return dist
    
def compute_tutor_weighted( X, nr_words, nr_words_per_utterance, p = 0.5):
    """ Initializes the tutor / nr_words = nr_prototypes
        Returns words per X and words per X binarized"""
    nr_dimensions = X.shape[1]
    tutor = [ numpy.random.binomial( 1, p, ( nr_words, nr_dimensions)),
             numpy.random.uniform( size = ( nr_words, nr_dimensions))]
    all_zeros = numpy.where( numpy.logical_not( tutor[0].any( axis = 1)))[0]
    tutor[0][ all_zeros, numpy.random.choice( nr_dimensions, size = len( all_zeros))] = 1.0
        
    assert( tutor[0].any( axis = 1).all())
    # take the first nr_words_per_utterance of the minimum distance indices
    y_cat = [numpy.argsort( dist_euclidian_weighted( x, tutor[0], tutor[1]))[:nr_words_per_utterance].tolist() for x in X]
    return y_cat, sklearn.preprocessing.MultiLabelBinarizer().fit_transform( y_cat)

###########################################

def compute_f_scores( y_true, y_pred):
    return numpy.array( [sklearn.metrics.precision_recall_fscore_support( y_true, y_pred, average = "micro")[2],
                         sklearn.metrics.precision_recall_fscore_support( y_true, y_pred, average = "macro")[2],
                         sklearn.metrics.precision_recall_fscore_support( y_true, y_pred, average = "samples")[2],
                         sklearn.metrics.precision_recall_fscore_support( y_true, y_pred, average = "weighted")[2]])

###########################################

def test_clf( X_train, y_train, X_test, y_test, classifier):
    """ Computes the learner on train, tests on test and train returns prec/rec/f-score
        This one recreates the classifier every time """
    print( "%s: Running test_clf %s " % (datetime.datetime.now(), classifier))
    if hasattr( classifier, "clone"):
        clf = classifier.clone()
    else:
        clf = sklearn.base.clone( classifier)
    clf.fit( X_train, y_train);
    y_test_pred = clf.predict( X_test)
    y_train_pred = clf.predict( X_train)
    test, train = compute_f_scores( y_test, y_test_pred), compute_f_scores( y_train, y_train_pred)
    print( "%s: Finished test_clf %s" % (datetime.datetime.now(), classifier))
    return test, train

def apply_test_clf( args):
    return test_clf( *args)

def test_clf_ascending( X_train, y_train, X_test, y_test, classifier):
    """ Repeatedley runs test_clf on X_train from X_train[:1] to X_train[:-1]
        Recreates the classifier every time
        Returns test and train precision, recall of classifier with shape == (X_train.shape[0], 3) """
    print( "%s: Running test_clf_ascending %s"  % ( datetime.datetime.now(), classifier))
    results_test = []
    results_train = []

    for i in range( 1, X_train.shape[0]):
        print( "%s: Running test_clf_ascending %s: %i"  % (datetime.datetime.now(), classifier, i))
        try:
            result_test, result_train = test_clf( X_train[:i], y_train[:i], X_test, y_test, classifier)
        except:
            print( "%s Running test_clf_ascending %s/%i Exception ignored" % (datetime.datetime.now(), classifier, i))
            result_test, result_train = numpy.array([ 0,0,0]), numpy.array([ 0,0,0])
        results_test.append( result_test)
        results_train.append( result_train)
        print( "%s: Finished test_clf %s ascending: %i" % (datetime.datetime.now(), classifier, i))
    print( "%s: Finished test_clf_ascending %s" % (datetime.datetime.now(), classifier))
    return numpy.array( results_test), numpy.array( results_train)
    
def apply_test_clf_ascending( args):
    return test_clf_ascending( *args)

###########################################

def test_clfs_cross_validate(  X, y, classifiers, n_folds = 3, cv = None, ascending = False, workers = None, save_results = None, verbose = True):
    """ """
    ## Prepare training/test data -- same for all
    if cv == None:
        cv = sklearn.cross_validation.KFold( y.shape[0], n_folds = n_folds)
    train_test = list( cv)
    if len( set( [ len(t[0]) for t in train_test])) != 1:
        warnings.warn( "test_clfs_cross_validate different training set size across various trials", UserWarning)
        print( [ len(t[0]) for t in train_test])
            
    # load previous results
    all_results = {}
    if save_results:
        try:
            all_results = pickle.load( open( save_results, "rb"))
        except:
            pass

    if workers:
        args = []
        for classifier in classifiers:
            for train, test in train_test:
                args.append( ( X[train], y[train], X[test], y[test], classifier))
        pool = multiprocessing.Pool( workers);
        if ascending:
            results = pool.map( apply_test_clf_ascending, args)
        else:
            results = pool.map( apply_test_clf, args)
        pool.close(); pool.join()
        for i, classifier in enumerate( classifiers):
            classifier_results = results[i*len(train_test):i*len(train_test)+len(train_test)]
            results_test = numpy.add.reduce( [r[0] for r in classifier_results]) / float( n_folds)
            results_train = numpy.add.reduce( [r[1] for r in classifier_results]) / float( n_folds)
            
            all_results[str(classifier)] = ( results_test, results_train)
            if save_results:
                print( "Saving %s" % classifier)
                pickle.dump( all_results, open( save_results, "wb"))
    else:
        # no workers sequential
        for classifier in classifiers:
            print( "%s: Running %s" % (datetime.datetime.now(), classifier))
            if ascending:
                results = [ test_clf_ascending( X[train], y[train], X[test], y[test], classifier) for train, test in train_test]
            else:
                results = [ test_clf( X[train], y[train], X[test], y[test], classifier) for train, test in train_test]
            precision_recall_f_score_test = numpy.add.reduce( [r[0] for r in results]) / float( n_folds)
            precision_recall_f_score_train = numpy.add.reduce( [r[1] for r in results]) / float( n_folds)
            all_results[str(classifier)] = ( precision_recall_f_score_test, precision_recall_f_score_train)
            print( "%s: Finished %s" % (datetime.datetime.now(), classifier))
            if save_results:
                print( "Saving %s" % classifier)
                pickle.dump( all_results, open( save_results, "wb"))

    return all_results

