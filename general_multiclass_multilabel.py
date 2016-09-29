# -*- coding: utf-8 -*-
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import numpy
import pickle
import sklearn.base
import sklearn.cross_validation
import sklearn.ensemble
import sklearn.grid_search
#import sklearn.discriminant_analysis
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
import robot_data

def run_simulated_data_clfs( classifiers,
                            save_results = "gmm-results-simulated-data.pickle",
                            nr_words = 100,
                            nr_words_per_utterance = 5, # utter best n utterances
                            nr_dimensions = 17,
                            nr_samples = 4532, 
                            p = 0.5,
                            n_folds = 4,
                            ascending = False,
                            workers = None, verbose = True):

    if verbose:
        print( "Prepare data: nr_words=%i; nr_dimensions=%i; nwpu=%i; nr_samples=%i; n_folds=%i" % ( nr_words, nr_dimensions, nr_words_per_utterance, nr_samples, n_folds))
    numpy.random.seed( seed = None)
    # topics
    X = numpy.random.uniform( size = ( nr_samples, nr_dimensions))
    y_cat, y_bin = description_game.compute_tutor_weighted( X, nr_words, nr_words_per_utterance, p = p)
        
    if verbose:
        print( "Run cross validation: nr_words=%i; nr_dimensions=%i; nwpu=%i; nr_samples=%i; n_folds=%i" % ( nr_words, nr_dimensions, nr_words_per_utterance, nr_samples, n_folds))

    results = description_game.test_clfs_cross_validate( X, y_bin, classifiers, 
                                                        n_folds = n_folds, 
                                                        workers = workers, 
                                                        save_results = save_results, 
                                                        ascending = ascending)

    if verbose:
        print( "Finished cross validation: nr_words=%i; nr_dimensions=%i; nwpu=%i; nr_samples=%i; n_folds=%i" % ( nr_words, nr_dimensions, nr_words_per_utterance, nr_samples, n_folds))
    return results

#print( run_simulated_data_clfs( workers = None))
    
def run_robot_data_all_clfs( classifiers,
                   save_results = "gmm-results-robot-data-all.pickle",
                   nr_words = 100,
                   nr_words_per_utterance = 5, # utter best n utterances
                   n_folds = 4,
                   data_sets = robot_data.OBJECT_DATA_SETS,
                   channels = None,
                   ascending = False,
                   workers = None, verbose = True):

    if verbose:
        print( "Prepare data: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))
    # topics
    data = robot_data.load_data( data_sets, suffix = "-object-features-all-numerical-channels.csv")
    if channels:
        data = data[:,channels]
    X = sklearn.preprocessing.MinMaxScaler( feature_range = (0, 1), copy=True).fit_transform( data)
    y_cat, y_bin = description_game.compute_tutor_weighted( X, nr_words, nr_words_per_utterance)

    if verbose:
        print( "Run cross validation: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))

    if verbose:
        print( "Run cross validation: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))
    
    results = description_game.test_clfs_cross_validate( X, y_bin, classifiers, 
                                                        n_folds = n_folds, 
                                                        workers = workers, 
                                                        save_results = save_results, 
                                                        ascending = ascending)

    if verbose:
        print( "Finished cross validation: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))
    return results

#run_robot_data_all_clfs( workers = 16)

def run_robot_data_clfs( classifiers,
                   save_results = "gmm-results-robot-data.pickle",
                   nr_words = 100,
                   nr_words_per_utterance = 5, # utter best n utterances
                   nr_dimensions = 17,
                   n_folds = 4,
                   data_sets = robot_data.OBJECT_DATA_SETS,
                   channels = None,
                   ascending = False,
                   workers = None, verbose = True):
    """ multi classifier """
    if verbose:
        print( "Prepare data: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))

    data_a = robot_data.load_data( robot_data.OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "a")
    data_b = robot_data.load_data( robot_data.OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "b")
    
    scaler = sklearn.preprocessing.MinMaxScaler( feature_range = (0, 1)).fit( numpy.concatenate( [data_a, data_b], axis = 0))
    
    X_a = scaler.transform( numpy.concatenate( [data_a,data_b], axis = 0))
    X_b = scaler.transform( numpy.concatenate( [data_b,data_a], axis = 0))
    y_cat, y_bin = description_game.compute_tutor_weighted( X_a, nr_words, nr_words_per_utterance)

    if verbose:
        print( "Run cross validation: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))
    
    results = description_game.test_clfs_cross_validate( X_b, y_bin, classifiers, 
                                                        n_folds = n_folds, 
                                                        workers = workers, 
                                                        save_results = save_results)

    if verbose:
        print( "Finished cross validation: nr_words=%i; nwpu=%i; n_folds=%i" % ( nr_words, nr_words_per_utterance, n_folds))
    return results

#run_robot_data_all_clfs( workers = 16)

def plot_f_scores_ascending( results, title = "f-scores", output_file = "graphs/temp.html", training_data = False):
    """ if training_data == True then we plot f-scores for training data """
    keys = sorted( results.keys())
    legends = []
    # shorten keys to names 
    for k in keys:
        if k.find("(") > 0:
            legends.append( k[:k.find("(")])
        else:
            legends.append( k)

    bokeh.plotting.output_file( output_file)
    p = bokeh.plotting.figure( title = title, 
                              tools = [ #bokeh.models.HoverTool(),
                                        bokeh.models.WheelZoomTool(), 
                                        bokeh.models.PanTool()], 
                                width = 800, 
                                height = 600)
    colors = bokeh.palettes.brewer["Spectral"][len(keys)]
    for idx, k in enumerate( keys):
        if training_data:
            result = results[k][1]
        else:
            result = results[k][0]
        p.line( numpy.arange( result.shape[0]), result[:,2], color = colors[idx], legend = legends[idx], line_width = 2)
    bokeh.plotting.show( p)
    
def plot_f_scores_pylab_ascending( results, title = "f-scores", output_file = "graphs/temp.html", training_data = False):
    """ if training_data == True then we plot f-scores for training data """
    import matplotlib.pylab
    matplotlib.pylab.ion() # interactive mode on
    matplotlib.pylab.figure()
    
    keys = sorted( results.keys())
    labels = []
    # shorten keys to names 
    for k in keys:
        if k.find("(") > 0:
            labels.append( k[:k.find("(")])
        else:
            labels.append( k)
    
    colors = bokeh.palettes.brewer["Spectral"][len(keys)]
    for idx, k in enumerate( keys):
        if training_data:
            result = results[k][1]
        else:
            result = results[k][0]
            matplotlib.pylab.plot( result[:,2], color = colors[idx], label = labels[idx])
    matplotlib.pylab.ylabel('f-score')
    matplotlib.pylab.xlabel('interactions')
    matplotlib.pylab.legend(loc=0, borderaxespad=0.)
    matplotlib.pylab.show()

def print_results( results, print_train = False):
    """
results = pickle.load( open( "gmm-results-simulated-data.pickle", "rb"))
print_results( results)
    """
    # shorten keys to names
    for k in sorted( results.keys()):
        result = results[k]
        if k.startswith( "OneVsRestClassifier(estimator="):
            k = k[30:]
        if k.find("(") > 0:
            k = k[:k.find("(")]
        if k.endswith( "Classifier"):
            k = k[:-10]
        if print_train:
            print( "%s: %.2f, %.2f, %.2f (%.2f, %.2f, %.2f)" % (k, 100 * result[0][0], 100 * result[0][1], 100 * result[0][2], 100 * result[1][0], 100 * result[1][1], 100 * result[1][2]))
        else:
            print( "%s: %.2f, %.2f, %.2f" % (k, 100 * result[0][0], 100 * result[0][1], 100 * result[0][2]))

def get_names( keys):
    """ Transforms full classifier strings into names"""
    names = []
    # shorten keys to names
    for k in keys:
        if k.startswith( "OneVsRestClassifier(estimator="):
            k = k[30:]
        if k.find("(") > 0:
            k = k[:k.find("(")]
        if k.endswith( "Classifier"):
            k = k[:-10]
        names.append( k)
    return names
    
def print_results_tex( results):
    """
results = [pickle.load( open( "gmm-results-simulated-data-optimized.pickle", "rb")), pickle.load( open( "gmm-results-robot-data-all-optimized.pickle", "rb")),pickle.load( open( "gmm-results-robot-data-optimized.pickle", "rb"))]
print_results_tex( results)

results = [pickle.load( open( "gmm-results-simulated-data-non-optimized.pickle", "rb")), pickle.load( open( "gmm-results-robot-data-all-non-optimized.pickle", "rb")),pickle.load( open( "gmm-results-robot-data-optimized.pickle", "rb"))]
print_results_tex( results)

results = [pickle.load( open( "gmm-results-simulated-data--nr-dimensions=10--nr-samples=4532-non-optimized.pickle", "rb")), pickle.load( open( "gmm-results-simulated-data--nr-dimensions=100--nr-samples=9064-non-optimized.pickle", "rb")), pickle.load( open( "gmm-results-simulated-data--nr-dimensions=1000--nr-samples=13596-non-optimized.pickle", "rb"))]
print_results_tex( results)

results = [pickle.load( open( "gmm-results-simulated-data--nr-dimensions=10--nr-samples=4532-optimized.pickle", "rb")), pickle.load( open( "gmm-results-simulated-data--nr-dimensions=100--nr-samples=9064-optimized.pickle", "rb")), pickle.load( open( "gmm-results-simulated-data--nr-dimensions=1000--nr-samples=13596-optimized.pickle", "rb"))]
print_results_tex( results)

    """
    names = [ dict( zip( get_names(list(r.keys())), list(r.keys()))) for r in results]
    for name in sorted( set([ n for d in names for n in d.keys()])):
        f_scores = [ "%.2f"  % (100.0 * r[n[name]][0][2]) if name in n else "" for r, n in zip(results, names)]
        print( "%s & %s \\\\\\hline" % (name, " & ".join(f_scores)))

def plot_results_scaling( results, scales = [ 10, 100, 1000]):
    """ Results should be a [{ classifier_name : [[precision,recall,f],[precision,recall,f]]}] each entry corresponding to scale
    Example
scales = [10,100,1000,10000]
results = [pickle.load( open( "gmm-results-simulated-data--nr-dimensions=%i-non-optimized.pickle" % dim, "rb")) for dim in scales]
plot_results_scaling( results, scales)

scales = [10,100,1000]
results = [pickle.load( open( "gmm-results-simulated-data--nr-dimensions=%i-optimized.pickle" % dim, "rb")) for dim in scales]
plot_results_scaling( results, scales)

ps = [0.1,0.25,0.5,0.75,1.0]
results = [pickle.load( open( "gmm-results-simulated-data--p=%.2f-non-optimized.pickle" % p, "rb")) for p in ps]
plot_results_scaling( results, ps)

ps = [0.1,0.25,0.5,0.75,1.0]
results = [pickle.load( open( "gmm-results-simulated-data--p=%.2f-optimized.pickle" % p, "rb")) for p in ps]
plot_results_scaling( results, ps)

scales = [10,100,1000]
results = [pickle.load( open( "gmm-results-simulated-data--nr-dimensions=%i--nr-samples=%i-non-optimized.pickle" % (scale, numpy.log10( scale)) * 4532, "rb")) for scale in scales]
plot_results_scaling( results, scales)
    """
    import matplotlib.pylab
    keys = list(set.union( *[ set(r.keys()) for r in results]))
    names_results = {}
    for k in keys:
        name = get_names( [k])[0]
        names_results[name] = [100 * r[k][0][2] for r in results]
    bar_width = (1. - 0.1) / len( names_results)
    if len(names_results) in bokeh.palettes.brewer["Spectral"].keys():
        colors = bokeh.palettes.brewer["Spectral"][len(names_results)]
    else:
        colors = (bokeh.palettes.Spectral11 + bokeh.palettes.RdYlGn11)[:len(names_results)]
        
    
    indices_x = numpy.arange( len( scales))
    fig, ax = matplotlib.pylab.subplots()
    rects = []
    for idx, k in enumerate( sorted( names_results.keys())):
        rects.append( ax.bar( indices_x + idx * bar_width, names_results[k], bar_width, color = colors[idx]))
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel( 'F-Scores')
    ax.set_ybound( [0,100])
    ax.set_title('F-Scores Scaling')
    ax.set_xticks( indices_x + len( names_results) * bar_width / 2)
    ax.set_xticklabels( scales)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend( ( r[0] for r in rects), sorted( names_results.keys()), loc='center left', bbox_to_anchor=(1, 0.5))
    matplotlib.pylab.show()

def plot_results_ascending( results, testing = True):
    """ Plots results ascending 
    Example
results = pickle.load( open( "gmm-results-ascending-simulated-data.pickle", "rb"))    
plot_results_ascending( results)
    """
    import matplotlib.pylab
    names_results = {}
    for k, value in results.items():
        name = get_names([k])[0]
        if testing:
            names_results[name] = value[0]
        else:
            names_results[name] = value[1]
    fig, ax = matplotlib.pylab.subplots()
    colors = bokeh.palettes.brewer["Spectral"][ len( names_results)]
    for idx, k in enumerate( sorted( names_results.keys())):
        ax.plot( 100 * names_results[k][:,2], color = colors[idx], label = k)
    ax.set_ylabel('f-score')
    ax.set_ybound( [0,100])
    ax.set_xlabel('interactions')
    ax.legend( loc=0, borderaxespad=0.)
    matplotlib.pylab.show()


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
    parser.add_argument( '--experiment-scaling-dimensions-and-samples',
                    action = "store",
                    dest = "experiment_scaling_dimensions_and_samples",
                    default = None,
                    type = int,
                    help = "")
    parser.add_argument( '--experiment-scaling-dimensions-adaboost',
                    action = "store",
                    dest = "experiment_scaling_dimensions_adaboost",
                    default = None,
                    type = int,
                    help = "")
    parser.add_argument( '--experiment-scaling-p',
                    action = "store",
                    dest = "experiment_scaling_p",
                    default = None,
                    type = float,
                    help = "")    
    parser.add_argument( '--experiment-ascending',
                    action = "store_true",
                    dest = "experiment_ascending",
                    default = False,
                    help ="")
    parser.add_argument('--optimized-classifiers',
                    action = "store_true",
                    dest = "optimized_classifiers",
                    default = False,
                    help= "whether to load classifiers from optimized_classifiers.pickle")

    parser.add_argument('--workers',
                    action = "store",
                    dest = "workers",
                    default = 16,
                    type = int,
                    help= "how many workers")
    args = parser.parse_args()

    warnings.filterwarnings( "ignore")

    # regression: sklearn.linear_model.LinearRegression(), sklearn.linear_model.Ridge(), 
    # others = [ sklearn.discriminant_analysis.LinearDiscriminantAnalysis(), sklearn.tree.DecisionTreeClassifier(), sklearn.neighbors.NearestCentroid()]

    linear = [ sklearn.multiclass.OneVsRestClassifier( sklearn.linear_model.SGDClassifier()), ## STANDARD SCALE DATA
              sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.PassiveAggressiveClassifier()), 
              sklearn.multiclass.OneVsRestClassifier(sklearn.linear_model.LogisticRegression())]
    ensemble = [ # sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.BaggingClassifier()),
                sklearn.ensemble.RandomForestClassifier(),
                sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.ExtraTreesClassifier()),
                sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.AdaBoostClassifier()),
                sklearn.multiclass.OneVsRestClassifier( sklearn.ensemble.GradientBoostingClassifier())]
    bayesian = [ sklearn.multiclass.OneVsRestClassifier( sklearn.naive_bayes.GaussianNB()), 
                sklearn.multiclass.OneVsRestClassifier( sklearn.naive_bayes.MultinomialNB()),
                sklearn.multiclass.OneVsRestClassifier( sklearn.mixture.GMM( n_components = 20))]
    # svm = [sklearn.svm.SVC( kernel = 'linear'), sklearn.svm.SVC( kernel = 'rbf'), sklearn.svm.SVC( kernel = 'poly'), sklearn.svm.SVC( kernel = 'sigmoid')]

    neighbors = [ sklearn.neighbors.KNeighborsClassifier(), nearest_centroid.NearestCentroidOvR() ]
    
    classifiers = linear + ensemble + bayesian + neighbors
    

    optimized_str = "non-optimized"
    if args.optimized_classifiers:
        print( "Using optimized classifiers") 
        classifiers = list( pickle.load( open( "optimized_classifiers.pickle", "rb")).values())
        optimized_str = "optimized"
    
    if args.experiment_1:
        run_simulated_data_clfs( [classifiers[8]], 
                                save_results = "gmm-results-simulated-data-%s.pickle" % optimized_str,
                                workers = args.workers)
        run_robot_data_all_clfs( classifiers, 
                                save_results = "gmm-results-robot-data-all-%s.pickle" % optimized_str,
                                workers = args.workers)
        run_robot_data_clfs( [classifiers[8]], 
                            save_results = "gmm-results-robot-data-%s.pickle" % optimized_str,
                            workers = args.workers)
        
    # scaling dimensions
    if args.experiment_scaling_dimensions:
        run_simulated_data_clfs( classifiers,
                                nr_dimensions = args.experiment_scaling_dimensions,
                                save_results = "gmm-results-simulated-data--nr-dimensions=%i-%s.pickle" % (args.experiment_scaling_dimensions, optimized_str),
                                workers = args.workers)
    # scaling p
    if args.experiment_scaling_p:
        run_simulated_data_clfs( classifiers,
                                nr_dimensions = 10,
                                p = args.experiment_scaling_p,
                                save_results = "gmm-results-simulated-data--p=%.2f-%s.pickle" % (args.experiment_scaling_p, optimized_str),
                                workers = args.workers)                                    
    # ascending
    if args.experiment_ascending:
        run_simulated_data_clfs( classifiers, 
                                save_results = "gmm-results-ascending-simulated-data-%s.pickle" % optimized_str,
                                ascending = True, 
                                workers = args.workers)
        # run_robot_data_all_clfs( classifiers, save_results = "gmm-results-ascending-robot-data-all.pickle", ascending = True, workers = args.workers)
        # run_robot_data_clfs( classifiers, save_results = "gmm-results-ascending-robot-data.pickle", ascending = True, workers = args.workers)
    if args.experiment_scaling_dimensions_and_samples:
        nr_samples = int( numpy.log10( args.experiment_scaling_dimensions_and_samples) * 4532)
        run_simulated_data_clfs( classifiers,
                                nr_dimensions = args.experiment_scaling_dimensions_and_samples,
                                nr_samples = nr_samples,
                                save_results = "gmm-results-simulated-data--nr-dimensions=%i--nr-samples=%i-%s.pickle" % (args.experiment_scaling_dimensions_and_samples, nr_samples, optimized_str),
                                workers = args.workers)

    if args.experiment_scaling_dimensions_adaboost:
            classifiers = [pickle.load( open( "optimized_classifiers_%i.pickle" % args.experiment_scaling_dimensions_adaboost, "rb"))["AdaBoost"]]
            run_simulated_data_clfs( classifiers,
                                    nr_dimensions = args.experiment_scaling_dimensions_adaboost,
                                    save_results = "gmm-results-simulated-data--nr-dimensions=%i-adaboost.pickle" % args.experiment_scaling_dimensions_adaboost,
                                    workers = args.workers)
        

