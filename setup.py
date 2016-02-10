from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


USE_CYTHON = True
ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension("pyspatial.spatiallib", ["pyspatial/spatiallib" + ext],
        include_dirs = [numpy.get_include()]),
]

setup(
    name = "pyspatial",
    version='0.1.0',
    ext_modules = cythonize(extensions),
    packages=['pyspatial'],
)
