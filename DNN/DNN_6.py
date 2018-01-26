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


# All lables for the CSV containing filtered and randomized Illustris data
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

FEATURES = ['SubhaloBHMass',                        ## 0.775 corr. with Mass
            'SubhaloGasMetallicity',                ## 0.113 corr. with Mass
            #'SubhaloGasMetallicitySfr',            # 0.108 corr. with Mass
            #'SubhaloSFR',                          # 0.070 corr. with Mass
            'SubhaloStarMetallicity',               ## 0.170 corr. with Mass
            'SubhaloStellarPhotometricsMassInRad',  ## 0.746 corr. with Mass
            #'SubhaloStellarPhotometricsRad',       # 0.380 corr. with Mass
            #'SubhaloStellarPhotometricsU',         # -0.153 corr. with Mass
            #'SubhaloStellarPhotometricsB',         # -0.161 corr. with Mass
            #'SubhaloStellarPhotometricsV',         # -0.167 corr. with Mass
            #'SubhaloStellarPhotometricsK',         # -0.172 corr. with Mass
            #'SubhaloStellarPhotometricsg',         # -0.163 corr. with Mass
            #'SubhaloStellarPhotometricsr',         # -0.169 corr. with Mass
            #'SubhaloStellarPhotometricsi',         # -0.171 corr. with Mass
            #'SubhaloStellarPhotometricsz',         # -0.172 corr. with Mass
            'SubhaloVelX',                          ## -0.00168 corr. with Mass
            'SubhaloVelY',                          ## 0.00215 corr. with Mass
            'SubhaloVelZ'                           ## 0.00147 corr. with Mass
            ]

LABEL =     'SubhaloMass'


# Custom pipeline for easy input feature manipulation, which
# returns input columns and output column as TF constants
def input_fn(data_set):
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    #labels = tf.cast(tf.constant(data_set[LABEL].values), tf.float32)
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols, labels


def main(unused_argv):
    # Read CSV's without headers and white space in between items
    training_set = pd.read_csv("Illustris_1_Training.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("Illustris_1_Test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)



    # Normalize specified input features, but not output
    #mm = MinMaxScaler()


    # Training set normalization
    #training_set_scaled = {j: mm.fit_transform(training_set[j].values) for j in FEATURES}
    training_set_scaled = {j: preprocessing.StandardScaler().fit_transform(training_set[j].values) for j in FEATURES}

    training_set_scaled_df = pd.DataFrame(training_set_scaled, index = training_set.index, 							columns=training_set.columns)

    training_set_scaled_df['SubhaloMass'] = training_set['SubhaloMass'].values


    # Test set normalization
    #test_set_scaled = {j: mm.fit_transform(test_set[j].values) for j in FEATURES}
    test_set_scaled = {j: preprocessing.StandardScaler().fit_transform(test_set[j].values) for j in FEATURES}

    test_set_scaled_df = pd.DataFrame(test_set_scaled, index = test_set.index, 						columns=test_set.columns)

    test_set_scaled_df['SubhaloMass'] = test_set['SubhaloMass'].values



    # Convert CSV feature inputs to TF columns
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]


    # Set up a 5 layer deep neural network with 18, 36, 72, 36, 18 units, respectively
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[7,6,5,4],
                                              model_dir="/home/aaron/MLprograms/DM/DNN #6/DNN #6 Model",
                                              #config=tf.contrib.learn.RunConfig(
							#save_checkpoints_steps=1000,
							#save_checkpoints_secs=None,
							#save_summary_steps=20
							#),
					      #optimizer=tf.train.ProximalAdagradOptimizer(
      						#learning_rate=0.01,
      						#l1_regularization_strength=0.001
    						#),
					      #optimizer=tf.train.FtrlOptimizer,
                                              activation_fn=tf.nn.crelu
                                              )
                       # good units for 7 feats: 7, 14, 28, 54, 28, 14, 7


    # Train the NN over 10,000 steps on the Training Set
    regressor.fit(input_fn=lambda: input_fn(training_set_scaled_df),
                  steps = 2000)


    # Test the NN on the Test Set and print MSE + Pearson Correlation b/w predicted vs true
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set_scaled_df),
			    steps=1,
			    #metrics={
				#'Predicted vs True (TEST DATA) Pearson Correlation':
					#MetricSpec(
						#metric_fn=metric_ops.streaming_pearson_correlation,
						#prediction_key="scores"
						#)
				     #}
			   )


    # Run MSE metric on Test Set again
    loss_score = ev["loss"]


    # Print out MSE metric for Test Set
    print("(TEST DATA) Loss: {0:f}".format(loss_score))



    # Make predictions, using Test Set (which is fine temporarily since it is unseen by NN)
    y = regressor.predict_scores(input_fn=lambda: input_fn(test_set_scaled_df))


    # Print Predictions Array through TF's preferred method
    predictions = list(itertools.islice(y, 6000))
    print("Predictions: {}".format(str(predictions)))

    # Make a new Pandas DF from Predictions Array in order to write to CSV
    df = pd.DataFrame(np.array(list(predictions)))
    print (df)
    #df.to_csv('predicted_halo_masses_smallNN_alt.csv')


    # To call Tensorboard: 'tensorboard --logdir=/home/aaron/MLprograms/DM/DNN\ #6/DNN\ #6\ Model/'


if __name__ == "__main__":
    tf.app.run()
