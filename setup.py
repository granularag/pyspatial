import os
import codecs
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

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return codecs.open(os.path.join(rootpath, *parts), 'r').read()


pkg_data = {'': ['templates/*.js',
                 'templates/*.html',
                 'templates/js/*.js',
                 'templates/html/*.html',
                 'templates/css/*.css']}

LICENSE = read('LICENSE.txt')
long_description = '{}\n{}'.format(read('README.md'), read('CHANGES.txt'))
setup(
    name="pyspatial",
    version='0.1.0',
    author="Granular, Inc",
    description='Data structures for working with (geo)spatial data',
    license='New BSD',
    url='https://github.com/granularag/pyspatial',
    ext_modules=cythonize(extensions),
    packages=['pyspatial'],
    package_data=pkg_data,
    classifiers=['Development Status :: 4 - Beta',
                 'Topic :: Scientific/Engineering :: GIS',
                 'License :: OSI Approved :: BSD License'],
    keywords=('spatial raster vector shapefile geojson data visualization '
              'pandas shapely gis geojson geographic geo')
)
