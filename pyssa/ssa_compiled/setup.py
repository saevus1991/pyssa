# install with 'python setup.py install --record files.txt'
import os
import numpy as np
from distutils.core import setup, Extension
import shutil

numpy_dir = np.get_include()

module1 = Extension('gillespie',
       include_dirs = [numpy_dir],
                     sources = ['gillespie.cpp'])

setup()

setup (name = 'Gillespie',
       version = '1.0',
       description = 'Gillespie algorithm for mass action systems',
       ext_modules = [module1])

cur_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = cur_dir + '/build'

for dirpath, subdirs, filenames in os.walk(build_dir):
    for filename in filenames:
        if '.so' in filename and 'gillespie' in filename:
            file_path = dirpath + '/' + filename
            target_path = os.path.dirname(cur_dir) + '/' + filename

# copy to above directory
shutil.copyfile(file_path, target_path)
