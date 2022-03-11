import os
from setuptools import setup
from setuptools import find_packages

setup(name='mGPLVM',
      author='Anonymous',
      version='0.0.1',
      description='Pytorch implementation of mGPLVM',
      license='MIT',
      install_requires=['numpy', 'torch==1.7', 'scipy>=1.0.0', 'scikit-learn', 'glmnet_py', 'matplotlib', 'mypy', 'sphinx', 'sphinx-rtd-theme', 'pytest', 'pytest-cov', 'jupyter', 'ipykernel'],
      packages=find_packages())
