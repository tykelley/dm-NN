from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
#from tensorflow.contrib.metrics.python.ops import metric_ops
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

import csv
import itertools
import pandas as pd
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)




"""

NEURAL NETWORK V6: PREDICTING HALO MASS FROM GALAXY PROPERTIES


CURRENT INPUTS:     Black Hole Mass
                    Gas Metallicity
                    Star Metallicity
                    Stellar Mass
                    Halo Velocities (X, Y, Z)

OUPUT:              Halo Mass

DATA:               Illustris-1 (filtered)

"""


# Exact same column / feature names as pre-processed Illustris_1 CSV
COLUMNS = ['New_Index',
            'Pre_Random_Index',
            'SubhaloBHMass',
            'SubhaloBHMdot',
            'SubhaloGasMetallicity',
            'SubhaloGasMetallicityHalfRad',
            'SubhaloGasMetallicityMaxRad',
            'SubhaloGasMetallicitySfr',
            'SubhaloGasMetallicitySfrWeighted',
            'SubhaloGrNr',
            'SubhaloHalfmassRad',
            'SubhaloIDMostbound',
            'SubhaloLen',
            'SubhaloMass',
            'SubhaloMassInHalfRad',
            'SubhaloMassInMaxRad',
            'SubhaloMassInRad',
            'SubhaloParent',
            'SubhaloSFR',
            'SubhaloSFRinHalfRad',
            'SubhaloSFRinMaxRad',
            'SubhaloSFRinRad',
            'SubhaloStarMetallicity',
            'SubhaloStarMetallicityHalfRad',
            'SubhaloStarMetallicityMaxRad',
            'SubhaloStellarPhotometricsMassInRad',
            'SubhaloStellarPhotometricsRad',
            'SubhaloVelDisp',
            'SubhaloVmax',
            'SubhaloVmaxRad',
            'SubhaloWindMass',
            'SubhaloStellarPhotometricsU',
            'SubhaloStellarPhotometricsB',
            'SubhaloStellarPhotometricsV',
            'SubhaloStellarPhotometricsK',
            'SubhaloStellarPhotometricsg',
            'SubhaloStellarPhotometricsr',
            'SubhaloStellarPhotometricsi',
            'SubhaloStellarPhotometricsz',
            'SubhaloVelX',
            'SubhaloVelY',
            'SubhaloVelZ']

# Column / feature names of inputs
FEATURES = ['SubhaloBHMass',                            ## 0.775 corr. with Halo Mass
            'SubhaloGasMetallicity',                    ## 0.113 corr. with Halo Mass
            #'SubhaloGasMetallicitySfr',                ## 0.108 corr. with Halo Mass
            #'SubhaloSFR',                              ## 0.070 corr. with Halo Mass
            'SubhaloStarMetallicity',                   ## 0.170 corr. with Halo Mass
            'SubhaloStellarPhotometricsMassInRad',      ## 0.746 corr. with Halo Mass
            #'SubhaloStellarPhotometricsRad',           ## 0.380 corr. with Halo Mass
            #'SubhaloStellarPhotometricsU',             ## -0.153 corr. with Halo Mass
            #'SubhaloStellarPhotometricsB',             ## -0.161 corr. with Halo Mass
            #'SubhaloStellarPhotometricsV',             ## -0.167 corr. with Halo Mass
            #'SubhaloStellarPhotometricsK',             ## -0.172 corr. with Halo Mass
            #'SubhaloStellarPhotometricsg',             ## -0.163 corr. with Halo Mass
            #'SubhaloStellarPhotometricsr',             ## -0.169 corr. with Halo Mass
            #'SubhaloStellarPhotometricsi',             ## -0.171 corr. with Halo Mass
            #'SubhaloStellarPhotometricsz',             ## -0.172 corr. with Halo Mass
            'SubhaloVelX',                              ## -0.00168 corr. with Halo Mass
            'SubhaloVelY',                              ## 0.00215 corr. with Halo Mass
            'SubhaloVelZ'                               ## 0.00147 corr. with Halo Mass
            ]

# Column / feature name of output
LABEL =     'SubhaloMass'


# Custom pipeline for easy input feature manipulation,
# returns input columns and output column as TensorFlow constants
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    #labels = tf.cast(tf.constant(data_set[LABEL].values), tf.float32)
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


def main(unused_argv):
    # Read in Illustris pre-processed CSV's as dataframes, renaming columns to COLUMNS
    training_set = pd.read_csv("Illustris_1_Training.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("Illustris_1_Test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)


    # Standardize Training Set (see SciKit Learn), write into new dataframe
    training_set_scaled = {j: preprocessing.StandardScaler().fit_transform(training_set[j].values)
                            for j in FEATURES}

    training_set_scaled_df = pd.DataFrame(training_set_scaled,
                                            index = training_set.index,
                                            columns=training_set.columns)

    training_set_scaled_df['SubhaloMass'] = training_set['SubhaloMass'].values


    # Standardize Test Set (see SciKit Learn), write into new dataframe
    test_set_scaled = {j: preprocessing.StandardScaler().fit_transform(test_set[j].values)
                        for j in FEATURES}

    test_set_scaled_df = pd.DataFrame(test_set_scaled,
                                        index = test_set.index,
                                        columns=test_set.columns)

    test_set_scaled_df['SubhaloMass'] = test_set['SubhaloMass'].values






    # Convert CSV feature inputs to TF columns
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]





    # Set up a 4 layer deep neural network with 7, 6, 5, 4 units, respectively
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                                hidden_units=[7,6,5,4],
                                                model_dir="/Users/aaron/Documents/Research/MLprograms/DM/dm-NN/DNN/Model",
                                                  #config=tf.contrib.learn.RunConfig
                                                        #save_checkpoints_steps=1000
                                                        #save_checkpoints_secs=None
                                                        #save_summary_steps=20,

                                                #optimizer=tf.train.ProximalAdagradOptimizer,
                                                #learning_rate=0.01,
                                                #l1_regularization_strength=0.001,
                                                #activation_fn=tf.nn.crelu
                                              )


    # Train the neural network for 10,000 steps, using the
    # pre-processed and standardized Training Set
    regressor.fit(input_fn=lambda: input_fn(training_set),
                    steps = 2000)


    # Test the neural network on the pre-processed and standardized Test Set
    # and calculate the Mean Squared Error
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set),
			                     steps=1
                    			    #metrics={
                    				#'Predicted vs True (TEST DATA) Pearson Correlation':
                    					#MetricSpec(
                    						#metric_fn=metric_ops.streaming_pearson_correlation,
                    						#prediction_key="scores"
                    						#)
                    				     #}
                            )

    loss_score = ev["loss"]


    # Print the Mean Squared Error for Test Set
    print("(TEST DATA) Loss: {0:f}".format(loss_score))


    # Print the exact Halo Mass predictions from the Test Set
    y = regressor.predict_scores(input_fn=lambda: input_fn(test_set))
    predictions = list(itertools.islice(y, 6000))
    print("Predictions: {}".format(str(predictions)))

    # Write Test Set Halo Mass predictions to a CSV
    #df = pd.DataFrame(np.array(list(predictions)))
    #print (df)
    #df.to_csv('predicted_halo_masses_smallNN_alt.csv')

    # To call Tensorboard: 'tensorboard --logdir=...'


if __name__ == "__main__":
    tf.app.run()
