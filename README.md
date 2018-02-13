
# Neural Networks and Dark Matter

This repository contains a deep neural network that predicts the mass
of dark matter halos in Illustris, given only "observable" properties as inputs.
The  goal is to have the neural network learn the galaxy-halo relationships within
raw simulation data.  

## Data
The data is from the Group Catalog in Snapshot 135 (Redshift 0) in the Illustris-1
hydrodynamical simulation.

## Architecture
The current version takes 10 "observable" properties as inputs. Data from 170,000
halos is passed through the neural network, which consists of 3 hidden layers, each
with 10 neurons. Each neuron has a Rectified Linear Unit activation function. The
Adagrad optimizer is used to compute and optimize the gradients as the neural
network "learns." The output is a single neuron, corresponding to the halo mass.

## Future Plans
Going forward, there are a few different options for addressing concerns of confidence
and uncertainty in the neural network. I will be tuning the hyperparameters (i.e.
number of hidden layers, number of neurons in the hidden layer, activation function, etc.)
 with Google Cloud Platform to utilize its close integration with TensorFlow. Google
 Cloud Platform uses a form a Bayesian optimization to find the best values for the
 hyperparameters. This is essential because hyperparameters are highly sensitive to
 the data and there is no universal formula for determining their best values. Regarding
 the architecture of the neural network, there are a few different types that I could
 build, each with their own pros and cons.  

### Option 1: Deep Neural Network, tuned on Google Cloud Platform

- Pros: Short development time (already done)
- Cons: Less transparency into confidence of results (outputs a single number for each prediction)


### Option 2: Deep Neural Network with Kernel Mixture Layer, tuned on Google Cloud Platform

- Pros: More transparency into confidence of results (outputs a probability density function for each prediction)
- Cons: Longer development time (~2 months), instability during training, does not improve neural network learning ability


###  Option 3: Bayesian Deep Neural Network, tuned on Google Cloud Platform

- Pros: Even more transparency into confidence of results (outputs a probability density function for each prediction), more flexibility in model architecture
- Cons: Very long development time (~6 months), requires more machine learning domain knowledge, does not improve neural network learning ability

## Questions To Consider
- Should we be more concerned with individual predictive ability or overall learning ability?
- Can we say that the neural network is learning the galaxy-halo relationships?
- Is it acceptable to make a deep neural network with a single output, instead of a PDF?
- Is it reasonable to remove all halos with zeros in the observable properties (i.e. Black Hole Mass, 0 Gas Metallicity, 0 Star Formation History)?
- Should I compare this neural network to other machine learning techniques (decision trees, polynomial regression, fast forest quantile regression)?

## Previous Work (Before I Knew About GitHub)
I built the first neural network when TensorFlow was in Release 1.2. At that time, DNNRegressor was part of tf.contrib.learn (playground API, not necessarily stable). I tuned this network by randomly selecting hyperparameters and testing various combinations of input features, which led me to train ~200 variations of the neural network. Eventually I found the optimal hyperparameters and input features. However, I became concerned that the neural network was overfitting the data, instead of learning the general galaxy-halo relationships. To address these concerns, I learned how batching, shuffling, and hyperparameter adjustments could help lower the possibility of overfitting. In subsequent TensorFlow releases (currently Release 1.5), the TF team deprecated all the API's that I was using. I rebuilt the neural network using the tf.Estimator and tf.Dataset API's and a canned estimator. These API's allowed me to implement batching and shuffling. I also was able to construct a new data pipeline where all data pre-processing occurs immediately before training. This minimizes the time needed to create Train and Test sets and allows for easy swapping of new datasets (i.e. Illustris TNG).  
