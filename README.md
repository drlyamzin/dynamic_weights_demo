This demo is intended for a dataset where a subject makes binary choices (e.g. Left/Right) on many trials, while the brain activity is being recorded. 

This code fits a dynamical psychometric model (probability of choice {0,1} given task conditions X) and a dynamical model of neural activity using state-space approach, i.e. a type of latent variable models with Markovian dynamics and a stochastic observation process. It then links up the inferred neural and psychometric latent variables using a third dynamical model.

**Available demos:**

**demo.ipynb** - simplest version of the demo: fits the dynamic psychometric (behavioral) model, then takes one estimated dynamical weight and looks for its correlates in the neural activity. 

**demo_fit_dynamic_A_k.ipynb** - finds a dynamic projection matrix from the latent psychometric variables to the neural activity. Assumes random walk dynamics on the weights of the projection matrix. Uses a previously fitted dynamical psychometric weights. 
