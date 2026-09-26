import numpy as np


def flops(image_size, batch):
  return batch*(17714*image_size**2+313344)

def memory(image_size, batch):
  return 4*(1039968+batch*image_size**2*26+868*batch)

def bytes_moved(image_size,batch):
  return 4*(1039968+batch*image_size**2*49+1636*batch)

def latency(image_size, batch, theta=4.44):
  return np.maximum(flops(image_size, batch)/(8.1*10**12), bytes_moved(image_size, batch)/(320*10**9)) * theta

def energy(image_size, batch, theta_energy):
  alpha, beta = theta_energy
  return alpha * flops(image_size, batch) + beta * bytes_moved(image_size, batch)
