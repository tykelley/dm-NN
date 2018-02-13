
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
 the data and there is no universal formula for determining their best value. Regarding
 the architecture of the neural network, there are a few different types that I could
 build, each with their own pros and cons.  

### Option 1: Deep Neural Network, tuned on Google Cloud Platform

- Pros: Short development time (already done)
- Cons: Less transparency into confidence of results (outputs a single number for each prediction)


### Option 2: Deep Neural Network with Kernel Mixture Network Layer, tuned on Google Cloud Platform

- Pros: More transparency into confidence of results (outputs a probability density function for each prediction)
- Cons: Longer development time (~2 months), instability during training


###  Option 3: Bayesian Deep Neural Network, tuned on Google Cloud Platform

- Pros: Even more transparency into confidence of results (outputs a probability density function for each prediction), more flexibility in model architecture
- Cons: Very long development time (~6 months), requires more machine learning domain knowledge
