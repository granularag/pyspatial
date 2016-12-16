"""
Copyright (c) 2016, Granular, Inc. 
All rights reserved.
License: BSD 3-Clause ("BSD New" or "BSD Simplified")

Redistribution and use in source and binary forms, with or without modification, are permitted 
provided that the following conditions are met: 

  * Redistributions of source code must retain the above copyright notice, this list of conditions 
    and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the 
    following disclaimer in the documentation and/or other materials provided with the distribution. 
  * Neither the name of the nor the names of its contributors may be used to endorse or promote products 
    derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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

if os.environ.get('READTHEDOCS', False) == 'True':
    INSTALL_REQUIRES = []
else:
    extensions = []
    INSTALL_REQUIRES = ['pandas', 'shapely', 'fiona', 'GDAL',
                        'scikit-image', 'RTree']

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
    version='0.2.1',
    author="Granular, Inc",
    maintainer="Aman Thakral",
    description='Data structures for working with (geo)spatial data',
    license='BSD',
    url='https://github.com/granularag/pyspatial',
    ext_modules=extensions,
    packages=['pyspatial'],
    package_data=pkg_data,
    long_description=long_description,
    install_requires=INSTALL_REQUIRES,
    classifiers=['Development Status :: 4 - Beta',
                 'Topic :: Scientific/Engineering :: GIS',
                 'License :: OSI Approved :: BSD License'],
    keywords=('spatial raster vector shapefile geojson data visualization '
              'pandas shapely gis geojson geographic geo')
)
