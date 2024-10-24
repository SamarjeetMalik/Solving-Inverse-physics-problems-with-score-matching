# Solving Inverse Physics Problems with Score Matching

This repository contains the ML implementaion for the paper "Solving Inverse Physics Problems with Score Matching". The paper can be found [here](https://arxiv.org/abs/2301.10250).



## Installation and Requirements

The code is written in Python 3.8 and tested with CUDA 11.4:

```bash conda create -n smdp python=3.8```

The majority of the code is based on JAX, which we install first with 

```bash 
conda activate smdp

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
``` 

For each experiment, additional packages can be installed with 

```bash
pip install -r requirements.txt
```

within the experiment's directory.

## Project Structure

This code covers several experiments. Each experiment is located in a separate folder. 

We learn score fields for 1D-processes with simple SDEs. The experiments are located in the folder `toy-example`. The simple setup allows us to compare the score learned by our method with the analytical score and evaluate how well the posterior distribution obtained from our method matches the true posterior distribution.

### Heat Diffusion

In this example, we learn the score field for the stochastic heat diffusion equation. The experiments are located in the folder `heat-diffusion`.  
As the diffusive nature of the equation destroys information over time, small-scale structures need to be created during inference. 
This highlights the advanatages of the SDE version of our method, as noise added to the trajectories can be used to create missing details.

### Buoyancy-driven Flow with Obstacles

This example is located in the folder `buoyancy-flow`. We learn the score field for a buoyancy-driven flow with obstacles. 
What makes this experiment challenging is that it involves non-linear physics and randomly placed obstacles for each simulation. 
This means that the learned score field needs to be able to generalize very well to unseen scenarios.

