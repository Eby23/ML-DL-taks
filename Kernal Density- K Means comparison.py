#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from parameters import parameters
from knn import knn
from kde import kde
from gauss1D import gauss1D
import matplotlib.pyplot as plt


# In[2]:


def gauss1D(m, v, N, w):
    pos = np.arange(-w, w - w / N, 2 * w / N)
    insE = -0.5 * ((pos - m) / v) ** 2
    norm = 1 / (v * np.sqrt(2 * np.pi))
    res = norm * np.exp(insE)
    realDensity = np.stack((pos, res), axis=1)
    return realDensity


# In[3]:


def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    # Compute the number of samples created
    N = len(samples)

    # Create a linearly spaced vector
    pos = np.arange(-5, 5.0, 0.1)

    # Estimate the density from the samples using a kernel density estimator
    norm = np.sqrt(2 * np.pi) * h * N
    res = np.sum(np.exp(-(pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2 / (2 * h ** 2)), axis=0) / norm

    # Form the output variable
    estDensity = np.stack((pos, res), axis=1)

    return estDensity


# In[4]:


def knn(samples, k):
    # compute density estimation from samples with KNN
    # Input
    #  samples    : DxN matrix of data points
    #  k          : number of neighbors
    # Output
    #  estDensity : estimated density in the range of [-5, 5]

   
    # Compute the number of the samples created
    N = len(samples)

    # Create a linearly spaced vector
    pos = np.arange(-5, 5.0, 0.1)

    # Sort the distances so that we can choose the k-th point
    dists = np.sort(np.abs(pos[np.newaxis, :] - samples[:, np.newaxis]), axis=0)

    # Estimate the probability density using the k-NN density estimation
    res = (k / (2 * N)) / dists[k - 1, :]

    # Form the output variable
    estDensity = np.stack((pos, res), axis=1)

    
    return estDensity


# In[5]:


def parameters():
    k = 30  # knn neighbors
    h = 0.3  # kde windowsize / radius
    return h, k


# In[6]:


h, k = parameters()

print('Find Kernel/K-Nearest Neighborhood Density Estimators')

# Produce the random samples
samples = np.random.normal(0, 1, 100)

# Compute the original normal distribution
realDensity = gauss1D(0, 1, 100, 5)

# Estimate the probability density using the KDE
estDensity = kde(samples, h)

# plot results
plt.subplot(2, 1, 1)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KDE Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()

# Estimate the probability density using KNN
estDensity = knn(samples, k)

# Plot the distributions
plt.subplot(2, 1, 2)
plt.plot(estDensity[:, 0], estDensity[:, 1], 'r', linewidth=1.5, label='KNN Estimated Distribution')
plt.plot(realDensity[:, 0], realDensity[:, 1], 'b', linewidth=1.5, label='Real Distribution')
plt.legend()
plt.show()


# In[ ]:




