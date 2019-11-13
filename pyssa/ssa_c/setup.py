import torch
#from setuptools import setup, Extension 
#from torch.utils import cpp_extension
from distutils.core import setup, Extension

module1 = Extension('demo',
                    sources = ['ssa.cpp'])

setup (name = 'SSA',
       version = '1.0',
       description = 'Simple SSA implementation',
       ext_modules = [module1])