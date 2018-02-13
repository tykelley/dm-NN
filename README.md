
# Overview: Neural Networks and Dark Matter

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



### Future Option 1: Deep Neural Network, tuned on Google Cloud Platform

Pros: Short development time
Cons: Less transparency into confidence of results (outputs a single number for each prediction)


### Future Option 2: Deep Neural Network with Kernel Mixture Network Layer, tuned on Google Cloud Platform

Pros: More transparency into confidence of results (outputs a probability density function for each prediction)
Cons: Longer development time, instability during training


### Future Option 3: Bayesian Deep Neural Network, tuned on Google Cloud Platform

Pros: More transparency into confidence of results (outputs a probability density function for each prediction)
Cons: Very long development time, requires more machine learning domain knowledge
