Code base for normalizing flows using python and tensorflow

Implemented by Siddhartha Saxena, Jaivardhan Kapoor and Shibhansh Dohare. Course project of Bayesian Machine Learning 2016-17 Sem 2. Extending the work of Rezende et al.  

The file normFlow_vae_tensorflow runs a Variational Auto-encoder which has a much more flexible posterior distribution as compared to vanilla vae. It takes MNIST as default-input 

Usage : python normFlow_vae_tensorflow.py (1/0 \[plot_or_not\]) \[number of flows\]

Output : 10 files with latent states of each number, 1 file with combined latent states, and a graph plotting them if plot_or_not=1. Out folder contains samples of generations after each 100
