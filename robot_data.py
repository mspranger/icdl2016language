# -*- coding: utf-8 -*-
import csv
import os
import math
import numpy
import warnings

ROBOT_DATA_DIR = os.getcwd() + "/../data-qrio-objects"
OBJECT_DATA_SETS = ["objects-1", "objects-1a", "objects-1b", "objects-2", "objects-3", "qrio-1", "qrio-2"]
SPACE_DATA_SETS = [ "space-game-%i" % i for i in range(2, 19)]

######################################################
## functions for crawling robot data
######################################################

def scene_dirs( directory):
    return [ directory + os.sep + scene_dir for scene_dir in filter( lambda d: d.startswith( "scene-"), os.listdir( directory))]

######################################################
## functions for dealing with s-expr
######################################################

def find_feature( name, obj):
    for f in obj[2]:
        if f[0].value() == name:
            return f
    return None;

def feature_to_list_w_name( feature):
    """takes [Symbol('stdev-v'), [Symbol('real'), 0.00139156]] and returns ['stdev-v' 0.00139156]"""
    value = feature[1][1]
    if feature[1][0].value() == "symbol":
        value = feature[1][1].value()
    elif feature[1][0].value() == "cyclic":
        # (* (- value 0.5) 2 pi)
        value = (value - 0.5) * math.pi
    return [feature[0].value(), value]

def feature_name( feature):
    return feature[0].value()

def feature_type( feature):
    return feature[1][0].value()
    
def feature_value( feature):
    """takes [Symbol('stdev-v'), [Symbol('real'), 0.00139156]] and returns 0.00139156"""
    value = feature[1][1]
    if feature[1][0].value() == "symbol":
        value = feature[1][1].value()
    elif feature[1][0].value() == "cyclic":
        # (* (- value 0.5) 2 pi)
        value = (value - 0.5) * math.pi
    return value
    
def object_id( obj):
    return obj[1]
    
#file_name = "/Users/spranger/Projects/robotdata/objects-1/scene-3397637287/a.lisp"
#def load_wm( file_name):
#    data = sexpdata.load( open(file_name, "rt"))
#    data_list = []
#    for obj in data:
#        data_list.append( [ feature_to_list(feature) for feature in obj[2]])

######################################################
## count data
######################################################

def count_scenes( data_sets = OBJECT_DATA_SETS,robot_data_dir = ROBOT_DATA_DIR):
    return len( [scene_dir for data_set in data_sets for scene_dir in scene_dirs( robot_data_dir + os.sep + data_set) ])
    
def load_scenes( data_sets = OBJECT_DATA_SETS, suffix = "-color-channels.csv", robot_data_dir = ROBOT_DATA_DIR, robot = [ "a", "b"]):
    "Load data sets from csv - loads everything into one array"
    if isinstance( robot, str):
        robot = [robot]
    scenes = [[] for r in robot]

    for data_set in data_sets:
        for scene_dir in scene_dirs( robot_data_dir + os.sep + data_set):
            file_names = [ scene_dir + os.sep + r + suffix for r in robot]
            file_exists = [ os.path.isfile( file_name) for file_name in file_names]
            if numpy.all( file_exists):
                for r in range( len( robot)): 
                    with open( file_names[r], "rt") as f:
                        reader = csv.reader( f, delimiter = ',')
                        data = []
                        for row in reader:
                            data.append( [ float(el) for el in row])
                        scenes[r].append( numpy.array( data))
            else:
                warnings.warn( "scene %s does not have data for all robots. Skipping!" % scene_dir, UserWarning)
#    numpy.all( [ len(s1) == len( s2) for s1, s2 in zip( scenes[0], scenes[1])])
    return scenes

def shuffle_scenes( scenes_a, scenes_b, percentage = 0.5, copy = True):
    "Shuffles scenes between a and b"
    assert( len( scenes_a) == len( scenes_b))
    if copy:
        scenes_a = scenes_a[:]
        scenes_b = scenes_b[:]
    switch = numpy.random.choice( numpy.arange( len( scenes_a)), len( scenes_a) * percentage)
    for s in switch:
        d_a = scenes_a[s][:]
        scenes_a[s] = scenes_b[s][:]
        scenes_b[s] = d_a
    return scenes_a, scenes_b

######################################################
## functions for loading csv data
######################################################
    
def load_data( data_sets = OBJECT_DATA_SETS, suffix = "-color-channels.csv", robot_data_dir = ROBOT_DATA_DIR, robot = [ "a", "b"]):
    "Load data sets from csv - loads everything into one array"
    if isinstance( robot, str):
        robot = [robot]
    data = []

    for data_set in data_sets:
        for scene_dir in scene_dirs( robot_data_dir + os.sep + data_set):
            for r in robot:
                file_name = scene_dir + os.sep + r + suffix
                if os.path.isfile( file_name):
                    with open( file_name, "rt") as f:
                        reader = csv.reader( f, delimiter = ',')
                        for row in reader:
                            data.append( [ float(el) for el in row])
    return numpy.array( data)
    
def shuffle_data( data_a, data_b, percentage = 0.5, copy = True):
    """Switches random entries in a and b
       Assuming that a and b are equally long, this algorithm might 
       replace A[i] with B[i] and B[i] with A[i]"""
    assert( data_a.shape == data_b.shape)
    if copy:
        data_a = numpy.copy( data_a)
        data_b = numpy.copy( data_b)
    switch = numpy.random.choice( numpy.arange( len( data_a)), len( data_a) * percentage)
    for s in switch:
        d_a = numpy.copy( data_a[s])
        data_a[s] = numpy.copy( data_b[s])
        data_b[s] = d_a
    return data_a, data_b
    
if __name__ == "__main__":
    data = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv")
    data_a = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "a")
    data_b = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "b")
    color_data = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv")
    color_data_a = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "a")
    color_data_b = load_data( OBJECT_DATA_SETS, suffix = "-object-features-all-numerical-channels.csv", robot = "b")

