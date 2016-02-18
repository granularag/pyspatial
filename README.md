# Overview

pyspatial is python package to provide data structures on top of gdal/ogr.  The 3 core data stuctures are:

* VectorLayer: a collection of geometries with pandas like manipulation.  Each geometry is an osgeo.ogr.Geometry object. For an object reference see [http://gdal.org/python/].
* RasterDataset: an abstraction of a spatial raster (both tiled on untiled) to support querying of pixels that intersect with shapes.
* RasterBand: a numpy array representation of a raster with spatial metadata (in memory only, no tiled support).

pyspatial makes it easy to read, analyze, query, and manipulate spatial data in both vector and raster form. It brings the familiarity of pandas to working with vector data, and provides querying capability similar to PostGIS for both vector and raster data.  Since it uses GDAL for much of the computations, the performance is quite good.  Based on the author's experience, the performance has been significantly better than PostGIS, and orders of magnitude faster than similar libraries in R.


# Development

* Python code: pip install -e /path/to/pyspatial
* Cython code: python setup.py build_ext --inplace
* Tests: nosetests -v