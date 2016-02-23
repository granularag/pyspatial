import os
import codecs
from distutils.core import setup
from distutils.extension import Extension
from pip.req import parse_requirements
import numpy


try:
    from Cython.Build import cythonize
    USE_CYTHON = True
    ext = ".pyx"

except ImportError as e:
    USE_CYTHON = False
    ext = ".c"

extensions = [
    Extension("pyspatial.spatiallib", ["pyspatial/spatiallib" + ext],
              include_dirs = [numpy.get_include()]),
]

if USE_CYTHON:
    extensions = cythonize(extensions)

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return codecs.open(os.path.join(rootpath, *parts), 'r').read()


pkg_data = {'': ['templates/*.js',
                 'templates/*.html',
                 'templates/js/*.js',
                 'templates/html/*.html',
                 'templates/css/*.css']}

long_description = '{}\n{}'.format(read('README.md'), read('CHANGES.txt'))
setup(
    name="pyspatial",
    version='0.1.3',
    author="Granular, Inc",
    maintainer="Aman Thakral",
    description='Data structures for working with (geo)spatial data',
    license='BSD',
    url='https://github.com/granularag/pyspatial',
    ext_modules=extensions,
    packages=['pyspatial'],
    package_data=pkg_data,
    long_description=long_description,
    classifiers=['Development Status :: 4 - Beta',
                 'Topic :: Scientific/Engineering :: GIS',
                 'License :: OSI Approved :: BSD License'],
    keywords=('spatial raster vector shapefile geojson data visualization '
              'pandas shapely gis geojson geographic geo')
)
