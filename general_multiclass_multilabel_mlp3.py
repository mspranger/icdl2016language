# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import numpy
import pickle
import keras.models
import keras.layers
import keras.optimizers

import description_game

class MLP3:
    clf = None;
    batch_size = 20
    nb_epoch = 100
    
    
    def __init__( self, layers, dropout, activation, input_dim = 17, output_dim = 100):
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.input_dim = input_dim # can be different from actual
        self.output_dim = output_dim # can be different from actual
        
        
    def __str__( self):
        return "MLP3(layers=%s,dropout=%f,activation=%s,input_dim=%i,output_dim=%i)" % ( self.layers, self.dropout, self.activation, self.input_dim, self.output_dim)
    def fit( self, X, y, verbose = 1):
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        # create model        
        clf = keras.models.Sequential()
        for idx, l in enumerate( self.layers):
            if idx == 0:
                clf.add( keras.layers.Dense( l, input_dim = input_dim, activation = self.activation))
            else:
                clf.add( keras.layers.Dense( l, activation = self.activation))
            clf.add( keras.layers.Dropout( self.dropout))
        clf.add( keras.layers.Dense( output_dim, activation = 'sigmoid'))
        clf.compile( optimizer = keras.optimizers.Adam(), loss = 'binary_crossentropy')
        # fit
        self.clf = clf
        self.clf.fit( X, y, batch_size = self.batch_size, nb_epoch = self.nb_epoch, verbose = verbose)
        
    def predict( self, X):
        y_pred = self.clf.predict( X)
        y_pred[y_pred >= .5] = 1
        y_pred[y_pred < .5] = 0
        return y_pred
    def clone( self):
        return MLP3( self.layers, self.dropout, self.activation, self.input_dim, self.output_dim)
                    
def create_mlp3_from_hyper_optimize( optimization_result, input_dim = 17, output_dim = 100):
    return MLP3( layers = optimization_result["layers"], 
               dropout = optimization_result["dropout"], 
               activation = optimization_result["activation"],
               input_dim = input_dim,
               output_dim = output_dim)

def hyper_optimize( nr_samples = 4532,
    nr_dimensions = 17,
    nr_words = 100,
    nr_words_per_utterance = 5,
    batch_size = 20,
    nb_epoch = 100,
    file_name = "mlp3_optimize_results_17.pickle"):

    print( "hyper_optimize MLP3 %s" % [nr_samples, nr_dimensions, nr_words, nr_words_per_utterance, batch_size, nb_epoch])
    
    X = numpy.random.uniform( size = ( nr_samples, nr_dimensions))
    y_cat, y_bin = description_game.compute_tutor_weighted( X, nr_words, nr_words_per_utterance)
    
    X_train, y_train, X_test, y_test = X[:3399], y_bin[:3399], X[3399:], y_bin[3399:]
    
    best_result = { "f-score" : 0.0 }
    for layer_size in [64,128,256,512,1024]:
        for nr_layers in range( 1, 3):
            for dropout in numpy.arange( 0.1, 1.0, 0.2):
                for activation in ["relu"]: #, "tanh", "sigmoid"
                    layers = [layer_size for i in range(nr_layers)]
                    print( "\n-----\nMLP3 %s with %s layers and dropout %.2f" % ( activation, layers, dropout))
                    try:
                        clf = MLP3( layers, dropout, activation,  X_train.shape[1], y_train.shape[1])
                        clf.fit( X_train, y_train)
                        
                        y_pred = clf.predict( X_train)
                        # print( sklearn.metrics.f1_score( y_train, y_pred, average = "samples"))
                        f_score_train = description_game.compute_f_scores( y_train, y_pred)
                        print( f_score_train)
                            
                        y_pred = clf.predict( X_test)
                        # print( sklearn.metrics.f1_score( y_test, y_pred, average = "samples"))
                        f_score_test = description_game.compute_f_scores( y_test, y_pred)
                        result = { "layers" : layers,
                                  "dropout" : dropout, 
                                  "activation" : activation, 
                                  "f-score" : f_score_test[2],
                                  "result-train" : f_score_train,
                                  "result-test" : f_score_test }
                        
                        print( f_score_test)
                        if result["f-score"] > best_result["f-score"]:
                            best_result = result
                            
                        print("\n\n Best so far %s" % best_result)
                        all_results = []
                        try:
                            all_results = pickle.load( open( file_name, "rb"))
                        except:
                            pass
                        all_results.append( result)
                        pickle.dump( all_results, open( file_name, "wb"))
                    except:
                        print( "ERROR processing MLP3 %s layers, dropout %.2f, activation %s" % ( layers, dropout, activation))
    return best_result

def load_optimized( nr_dimensions = 17):
    all_results = pickle.load( open( "mlp3_optimize_results_%i.pickle" % nr_dimensions, "rb"))
    all_results = sorted( all_results, key = lambda r: r["result-test"][2], reverse = True)
    return all_results[0]

if __name__ == "__main__": 
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument( '--experiment-1',
                    action = "store_true",
                    dest = "experiment_1",
                    default = False,
                    help = "")
    parser.add_argument( '--experiment-scaling-dimensions',
                    action = "store",
                    dest = "experiment_scaling_dimensions",
                    default = None,
                    type = int,
                    help = "")
    parser.add_argument( '--experiment-scaling-p',
                    action = "store",
                    dest = "experiment_scaling_p",
                    default = None,
                    type = float,
                    help = "")
    parser.add_argument('--hyper-optimize',
                    action = "store_true",
                    dest = "hyper_optimize",
                    default = False,
                    help= "grid search for best number of layers, activations and dropout")
    args = parser.parse_args()

    # optimize
    if args.hyper_optimize:
        hyper_optimize()
        for nr_dimensions in [ 10, 100, 1000, 10000]:
            hyper_optimize( nr_dimensions = nr_dimensions, file_name = "mlp3_optimize_results_%i.pickle" % nr_dimensions)

    # experiment robot data
    if args.experiment_1:
        clf = create_mlp3_from_hyper_optimize( load_optimized())

        import general_multiclass_multilabel
        for optimized_str in ["optimized","non-optimized"]:
            general_multiclass_multilabel.run_simulated_data_clfs( [clf], 
                            save_results = "gmm-results-simulated-data-%s.pickle" % optimized_str,
                            workers = None)
            general_multiclass_multilabel.run_robot_data_all_clfs( [clf],
                                    save_results = "gmm-results-robot-data-all-%s.pickle" % optimized_str,
                                    workers = None)
            general_multiclass_multilabel.run_robot_data_clfs( [clf],
                                save_results = "gmm-results-robot-data-%s.pickle" % optimized_str,
                                workers = None)

    # scaling dimensions
    if args.experiment_scaling_dimensions:
        clf = create_mlp3_from_hyper_optimize( load_optimized())
        import general_multiclass_multilabel
        for optimized_str in ["optimized","non-optimized"]:
            general_multiclass_multilabel.run_simulated_data_clfs( [clf],
                                    nr_dimensions = args.experiment_scaling_dimensions,
                                    save_results = "gmm-results-simulated-data--nr-dimensions=%i-%s.pickle" % (args.experiment_scaling_dimensions, optimized_str),
                                    workers = None)
    # scaling p
    if args.experiment_scaling_p:
        clf = create_mlp3_from_hyper_optimize( load_optimized())
        import general_multiclass_multilabel
        for optimized_str in ["optimized","non-optimized"]:
            general_multiclass_multilabel.run_simulated_data_clfs( [clf],
                                    nr_dimensions = 10,
                                    p = args.experiment_scaling_p,
                                    save_results = "gmm-results-simulated-data--p=%.2f-%s.pickle" % (args.experiment_scaling_p, optimized_str),
                                    workers = None)   