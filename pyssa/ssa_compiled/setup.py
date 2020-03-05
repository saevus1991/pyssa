# install with 'python setup.py install --record files.txt'
import os
import numpy as np
from distutils.core import setup, Extension
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
build_dir = cur_dir + '/build'

# copy to above directory
for dirpath, subdirs, filenames in os.walk(build_dir):
    for filename in filenames:
        if '.so' in filename and 'gillespie' in filename:
            file_path = dirpath + '/' + filename
            target_path = cur_dir + '/' + filename
            shutil.copyfile(file_path, target_path)


