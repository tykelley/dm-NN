# Overview: Neural Networks and Dark Matter Approximations

<<<<<<< HEAD
This repository contains 2 variations of a neural network to approximate the mass of dark matter halos in Illustris, given only the observable properties. The first variation is a Deep Neural Network (DNN), which outputs a single number for each halo mass prediction. The second variation is a Kernel Mixture Network (KMN), which outputs a probability density function for each halo mass prediction. Please refer to the README.md in each directory for details.  
=======
This repository contains 2 variations of a neural network to approximate the mass
of dark matter halos in Illustris, given only the observable properties. The first
variation is a Deep Neural Network, which outputs a single number for each halo
mass prediction. The second variation is a Kernel Mixture Network, which outputs
a probability density function for each halo mass prediction. Please refer to
the README.md in each directory for details.  


# Option 1: Deep Neural Network, tuned on Google Cloud Platform 

Pros: Short development time
Cons: Less transparency into confidence of results (outputs a single number for each prediction) 


# Option 2: Deep Neural Network with Kernel Mixture Network Layer, tuned on Google Cloud Platform

Pros: More transparency into confidence of results (outputs a probability density function for each prediction) 
Cons: Longer development time, instability during training 


# Option 3: Bayesian Deep Neural Network, tuned on Google Cloud Platform

Pros: More transparency into confidence of results (outputs a probability density function for each prediction)
Cons: Very long development time, requires more machine learning domain knowledge
>>>>>>> 20fd3f475e73e2a2fa416c2bf94cd529c42d37cb
