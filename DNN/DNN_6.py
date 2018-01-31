from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import dm_data


"""
NEURAL NETWORK: PREDICTING HALO MASS FROM GALAXY PROPERTIES

ACTIVE INPUTS - "FEATURES":     Black Hole Mass
                                Gas Metallicity
                                Star Metallicity
                                Stellar Mass
                                Halo Velocities (X, Y, Z)

OUPUT - "LABEL":                Halo Mass

DATA:                           Illustris-1 (randomized, indices preserved)
"""



# Column names of values to be used as input features
#   ## signifies results from corelation matrix
FEATURES = ['SubhaloBHMass',                            ## 0.775 corr. with Halo Mass
            'SubhaloGasMetallicity',                    ## 0.113 corr. with Halo Mass
            'SubhaloGasMetallicitySfr',                ## 0.108 corr. with Halo Mass
            'SubhaloSFR',                              ## 0.070 corr. with Halo Mass
            'SubhaloStarMetallicity',                   ## 0.170 corr. with Halo Mass
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
            'SubhaloVelX',                              ## -0.00168 corr. with Halo Mass
            'SubhaloVelY',                              ## 0.00215 corr. with Halo Mass
            'SubhaloVelZ'                               ## 0.00147 corr. with Halo Mass
            ]



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1000, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch a  pair of dataframes (features, labels) for training and testing data
    # features_dataframe: all feature columns except label column {subhalo mass}
    # labels_dataframe: only label column {subhalo mass}
    (train_features, train_label), (test_features, test_label) = dm_data.load_data()


    # Sets up TF "feature columns" which each identify feature name, type, and input processing
    # These feature columns are empty tensors, they just set up the input framework for the Estimator
    # For details: https://www.tensorflow.org/get_started/feature_columns
    # *** Still working on enabling normalization here ***
    feature_cols = [tf.feature_column.numeric_column(key=k) for k in FEATURES]


    # Define the structure of the deep neural network and the destination for the model checkpoint files
    regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                            hidden_units=[15,13,10,7,5],
                                            model_dir="/Users/aaron/Documents/Research/MLprograms/DM/dm-NN/DNN/Model"
                                            )


    # Train the neural network for steps specified above and at batch size specificed above, using Training data
    regressor.train(input_fn=lambda: dm_data.train_input_fn(train_features, train_label, args.batch_size),
                    steps=args.train_steps)


    # Evaluate the neural network, using Test data
    eval_result = regressor.evaluate(
        input_fn=lambda:dm_data.eval_input_fn(test_features, test_label, args.batch_size))

    print('\nTest Data MSE: {0:f}\n'.format(eval_result))


    """
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

    # Write first 6,000 of Test Set Halo Mass predictions to a CSV
    #df = pd.DataFrame(np.array(list(predictions)))
    #print (df)
    #df.to_csv('predicted_halo_masses_smallNN_alt.csv')

    # To call Tensorboard: 'tensorboard --logdir=...'

    """


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
