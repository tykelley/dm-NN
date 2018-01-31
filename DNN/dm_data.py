import pandas as pd
import tensorflow as tf


# Exact same column / feature names as pre-processed Illustris_1 CSV
# For future reference, no need to randomize before reading into here
CSV_COLUMN_NAMES = ['New_Index',
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



def load_data(feature_name='SubhaloMass'):
    # Fetches pair of dataframes (features, labels) for training and testing data
    # features_dataframe: all feature columns except label column {subhalo mass}
    # labels_dataframe: only label column {subhalo mass}

    train = pd.read_csv("Illustris_1_Training.csv",
                            names=CSV_COLUMN_NAMES,     # assign new header names
                            header=0                    # ignore old header names
                           )

    train_features, train_label = train, train.pop('SubhaloMass')


    test = pd.read_csv("Illustris_1_Test.csv",
                            names=CSV_COLUMN_NAMES,     # assign new header names
                            header=0                    # ignore old header names
                           )

    test_features, test_label = test, test.pop('SubhaloMass')

    return (train_features, train_label), (test_features, test_label)


def train_input_fn(features, labels, batch_size):
    # This function relies on the TF Dataset API
    # For better explanation see: https://www.tensorflow.org/get_started/get_started_for_beginners#the_program_itself

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(2000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    # This function relies on the TF Dataset API
    # For better explanation see: https://www.tensorflow.org/get_started/get_started_for_beginners#the_program_itself

    """Everything below is straight from TF"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
