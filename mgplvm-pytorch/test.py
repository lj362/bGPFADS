from turtle import ycor
from pytest import skip
import torch
import numpy as np

t = torch.ones(3, 4, 5)
a = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
nu2 = torch.ones(3, 4, 5)
tensor = torch.empty((3,4,10))


t[...,1,:] = 5
print(torch([1]))

