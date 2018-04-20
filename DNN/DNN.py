from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools

import tensorflow as tf
import numpy as np
import pandas as pd

import dm_data


"""
NEURAL NETWORK: PREDICTING HALO MASS FROM GALAXY PROPERTIES

ACTIVE INPUTS - "FEATURES":     Black Hole Mass
                                Gas Metallicity
                                Gas Metallicity SFR
                                SFR
                                Star Metallicity
                                Stellar Mass
                                'PhotometricsRad'
                                3D Velocities (X, Y, Z)

OUPUT - "LABEL":                Halo Mass

DATA:                           Illustris-1
"""



# Column names of values to be used as input features
#   ## signifies results from corelation matrix
CHOSEN_FEATURES = [#'SubhaloBHMass',                            ## 0.775 corr. with Halo Mass
                    'SubhaloGasMetallicity',                    ## 0.113 corr. with Halo Mass
                    #'SubhaloGasMetallicitySfr',                ## 0.108 corr. with Halo Mass
                    'SubhaloSFR',                              ## 0.070 corr. with Halo Mass
                    #'SubhaloStarMetallicity',                   ## 0.170 corr. with Halo Mass
                    'SubhaloStellarPhotometricsMassInRad',      ## 0.746 corr. with Halo Mass
                    'SubhaloStellarPhotometricsRad',           ## 0.380 corr. with Halo Mass
                    'SubhaloStellarPhotometricsU',             ## -0.153 corr. with Halo Mass
                    'SubhaloStellarPhotometricsB',             ## -0.161 corr. with Halo Mass
                    'SubhaloStellarPhotometricsV',             ## -0.167 corr. with Halo Mass
                    'SubhaloStellarPhotometricsK',             ## -0.172 corr. with Halo Mass
                    'SubhaloStellarPhotometricsg',             ## -0.163 corr. with Halo Mass
                    'SubhaloStellarPhotometricsr',             ## -0.169 corr. with Halo Mass
                    'SubhaloStellarPhotometricsi',             ## -0.171 corr. with Halo Mass
                    'SubhaloStellarPhotometricsz',             ## -0.172 corr. with Halo Mass
                    #'SubhaloVelX',                              ## -0.00168 corr. with Halo Mass
                    #'SubhaloVelY',                              ## 0.00215 corr. with Halo Mass
                    #'SubhaloVelZ'                               ## 0.00147 corr. with Halo Mass
                    ]



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',
                        default=8192,
                        type=int,
                        help='batch size')
parser.add_argument('--train_steps',
                        default=1000,
                        type=int,
                        help='number of training steps')


def from_dataset(ds):
    return lambda: ds.make_one_shot_iterator().get_next()


def main(argv):
    args = parser.parse_args(argv[1:])

    # Load Illustris_1 data (pre-filtered, not randomized) and split into train, test sets
    # Fetch a pair of dataframes for training set, testing set
    #    features dataframe: all feature columns except label column {SubhaloMass}
    #    labels dataframe: only label column {SubhaloMass}
    (train_features, train_label), (test_features, test_label) = dm_data.load_data()


    # TODO: look deeper into TF Docs on how this works at a low level
    #           https://www.tensorflow.org/get_started/datasets_quickstart

    # Use tf.Dataset API to create a "TF Dataset" which basically splits Pandas dataframes
    #   into slices of Numpy arrays. This allows it to be piped into the neural network
    #   more efficiently.
    # Shuffle data each time so it is always randomized. Shuffle buffer size must be
    #   bigger than dataset (>130,000)
    # Batch data into batches of 1024 halos, which means NN will train on each batch
    #   and then adjust parameters before training on the next batch. One epoch is when
    #   all batches have been run through (130,000 halos in train set / 1024 = 127 steps).
    train = (dm_data
                .make_dataset(train_features, train_label)
                .shuffle(30000)
                .batch(args.batch_size)
                .repeat()
            )

    # Same as above, but do not shuffle Test set because neural network is not training
    #   on this data. This data is only used for testing and evaluation, after the
    #   neural net has trained on the Train set.
    test = (dm_data
                .make_dataset(test_features, test_label)
                .batch(args.batch_size)
            )

    # Sets up TF "feature columns" which each identify feature name, type, and input processing.
    # These feature columns are empty tensors, they just set up the input framework for the Estimator.
    # For more details: https://www.tensorflow.org/get_started/feature_columns
    # TODO: think about enablabling normalization here
    feature_cols = [tf.feature_column.numeric_column(key=k) for k in CHOSEN_FEATURES]



    # Define the structure of the neural network and the destination for the checkpoint files.
    #   This version of the neural network has 3 hidden layers, each with 10 neurons. It uses the
    #   Adagrad optimizing algorithm and uses a relu (Rectified Linear Unit) activation function
    #   for each neuron. These are the hyperparameters and will be tuned via Google Cloud Platform
    #   HyperTune (https://cloud.google.com/ml-engine/docs/hyperparameter-tuning-overview).
    regressor = tf.estimator.DNNRegressor(
                        feature_columns=feature_cols,
                        hidden_units=[12,64,256,64,12],
                        model_dir="/Users/aaron/Documents/Research/MLprograms/DM/dm-NN/DNN/Model"
                        )

    # Train the neural network on Training set
    regressor.train(
                input_fn=from_dataset(train),
                steps=args.train_steps
                )

    # Evaluate the neural network on Test set
    eval_result = regressor.evaluate(
                                input_fn=from_dataset(test)
                                )

    average_loss = eval_result["average_loss"]

    print("\nMSE for Test Set: {0:f}".format(average_loss))




    # Generate predictions from the neural network on the Test Set
    # Returns the predictions as a dict
    predictor = regressor.predict(from_dataset(test))

    # Convert the dict of predictions to a dataframe
    predictions_df = pd.DataFrame.from_dict(predictor, dtype=float)

    print("\nNeural Network Predictions from Test Set:")
    print(predictions_df)

    # This corresponds to the predictions above because the test set is
    #    randomly selected before the prediction is made.
    print("\nReal Values from Test Set:")
    print(test_label)



    # Write a local CSV of neural network predicted halo mass (from Test Set)
    predictions_df.to_csv('NN_halo_mass_test_set36.csv')
    #print("\nTESTING PREDICTIONS")
    #print()
    #print(predictions_df.head())

    # Write a local CSV of true halo mass (from Test Set)
    test_label.to_csv('true_halo_mass_test_set36.csv')
    #print("\nTESTING TRUES")
    #print()
    #print(test_label.head())

    # Write a local CSV of all input features (from Test Set)
    test_features.to_csv('features_test_set36.csv')
    #print("\nTESTING FEATURES")
    #print()
    #print(test_features.head())

    # To call Tensorboard: 'tensorboard --logdir=...'



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
