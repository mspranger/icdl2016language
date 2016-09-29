# -*- coding: utf-8 -*-

import numpy
import sklearn

class NearestCentroidOvR():
    """ Multilabel, multiclass NearestCentroid, trains one vs rest"""
    estimators = None
    
    def __init__( self, metric = "euclidean", shrink_threshold = None):
        self.metric = metric
        self.shrink_threshold = shrink_threshold
        self.estimators = []

    def __str__( self):
        return "NearestCentroidOvR(metric=%s,shrink_treshold=%s)" % (self.metric, self.shrink_threshold)

    def clone( self):
        return NearestCentroidOvR( metric = self.metric, shrink_threshold = self.shrink_threshold)

    def fit( self, X, y):
        """ Expects y.shape[1] to be number labels """
        for i in range( y.shape[1]):
            nc = sklearn.neighbors.NearestCentroid( metric = self.metric, shrink_threshold = self.shrink_threshold)
            try:
                nc.fit( X, y[:,i])
                self.estimators.append( nc)
            except:
                self.estimators.append( None)
        
    def predict( self, X):
        y_pred = numpy.zeros( (X.shape[0], len( self.estimators)))
        for i,e in enumerate( self.estimators):
            if e == None:
                y_pred[:,i] = numpy.zeros( ( X.shape[0], ))
            else:
                y_pred[:,i] = e.predict( X)
        return y_pred
