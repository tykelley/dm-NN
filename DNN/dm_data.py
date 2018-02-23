import numpy as np
import pandas as pd
import tensorflow as tf


# List of column names to be assigned when the Illustris CSV is read in
# This exists as a reminder of data column names
CSV_COLUMN_NAMES = ['ignore_this',
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


def raw_dataframe():
    # Illustris_1.csv pre-filtering: remove halos with 0 stellar mass 
    #       and halos with <200 particles

    # Load data from Illustris_1.csv into Dataframe and reassign column names
    df = pd.read_csv("Illustris_1.csv",
                        header=0,                    # ignore old header names
                        names=CSV_COLUMN_NAMES       # assign new header names
                    )

    return df


def load_data(label_name='SubhaloMass', train_fraction=0.8, seed=None):
    # Loads Illustris_1 data (prefiltered, not randomized)
    # Splits Illustris)_1 data randomly into a training set and a test set
    # Fetches a pair of dataframes for training set, testing set
    #    features dataframe: all feature columns except label column {SubhaloMass}
    #    labels dataframe: only label column {SubhaloMass}

    data = raw_dataframe()

    # TODO: consider applying more filtering here (i.e. remove halos with 0 black hole mass)
    # Filter data if necessary
    #data = data.dropna()

    np.random.seed(seed)

    # Split Illustris_1 data randomly into 80% for training and 20% for testing
    train_features = data.sample(frac=train_fraction, random_state=seed)
    test_features = data.drop(train_features.index)

    # Remove label column {SubhaloMass} from Illustris_1 data and assign into new DataFrame
    train_label = train_features.pop(label_name)
    test_label = test_features.pop(label_name)

    return (train_features, train_label), (test_features, test_label)


def make_dataset(features, label=None):
    # Create a slice Dataset from DataFrame and labels (see TF.Dataset API Docs)
    # TODO: look deeper into TF Docs on how this works at a low level

    features = dict(features)

    # Convert pd.Series to np.arrays
    for key in features:
        features[key] = np.array(features[key])

    items = [features]

    if label is not None:
        items.append(np.array(label, dtype=np.float32))

    # Create a Dataset of slices
    return tf.data.Dataset.from_tensor_slices(tuple(items))



########################## ANOTHER VERSION ###############################
"""
def train_input_fn(features, labels, batch_size):
    # This function relies on the TF Dataset API

    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(100000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    # This function relies on the TF Dataset API

    #Everything below is straight from TF#
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

"""
