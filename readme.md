# Normalizing flows to generate MNIST Digits  

<p align="center">
  <img src="https://github.com/siddsax/Normalizing_Flows/blob/master/NF.jpg">
</p>


This code base implements Normalizing Flows as proposed in Rezende et al. to generate MNIST digits using Tensorflow. 

Usage:
```bash
python main.py [plot_or_not] [number of flows]
```

No need to download MNIST, tensorflow does it for you!

Outputs
* 10 files with latent states of each number
* 1 file with combined latent states 
* Graph plotting latent states as in the diagram if plot_or_not=1
* Folder names Out is generated containing samples of generations after interval of 100 iterations

<p float="left">
  <img src="/Original_10Lkh.png" width="400" />
  <img src="/10LkhIter_4Flows.png" width="400" height="300"/> 
  <figcaption>2D Latent Vector Representation, Left is vanilla VAE, Right is with Normalizing flows. As can be seen, the one with normalizing flows has a flexible multi-modal distribution, opposed to unimodal gaussian for vanilla vae</figcaption>
</p>


If you use the code base, please cite us at 

```bash
@article{saxena2017variational,
  title={Variational Inference via Transformations on Distributions},
  author={Saxena, Siddhartha and Dohare, Shibhansh and Kapoor, Jaivardhan},
  journal={arXiv preprint arXiv:1707.02510},
  year={2017}
}
```
