from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


setup (ext_modules = cythonize('calaculate_dot_plot_cy.pyx') )