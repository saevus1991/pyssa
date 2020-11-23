# install with 'python setup.py install --record files.txt'
import os
import numpy as np
from setuptools import setup, Extension
import shutil

numpy_dir = np.get_include()

# standard kinetic sssa
module1 = Extension('gillespie',
        include_dirs = [numpy_dir],
        sources = ['gillespie.cpp'])

setup (name = 'Gillespie',
        version = '1.0',
        description = 'Gillespie algorithm for mass action systems',
        ext_modules = [module1])

# ssa with piece-wise constant rate functions
module2 = Extension('gillespie_timedep',
        include_dirs = [numpy_dir],
        sources = ['gillespie_timedep.cpp'])

setup (name = 'Time-dependent Gillespie',
        version = '1.0',
        description = 'Gillespie algorithm for mass action systems. Rate constants vary with time as piece-wise constant functions',
        ext_modules = [module2])

cur_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(cur_dir, 'build')

# copy to above directory
for dirpath, subdirs, filenames in os.walk(build_dir):
    for filename in filenames:
        if ('.so' in filename or '.pyd' in filename) and 'gillespie' in filename:
            file_path = os.path.join(dirpath, filename)
            target_path = os.path.join(cur_dir, filename)
            shutil.copyfile(file_path, target_path)


