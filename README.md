# Overview

pyspatial is python package to provide data structures on top of gdal/ogr. It's core use cases have been around simplifying geospatial data science
workflows in Python.  The 3 core data stuctures are:

* VectorLayer: a collection of geometries with pandas like manipulation.  Each geometry is an osgeo.ogr.Geometry object. For an object reference see [http://gdal.org/python/].
* RasterDataset: an abstraction of a spatial raster (both tiled on untiled) to support querying of pixels that intersect with shapes.
* RasterBand: a numpy array representation of a raster with spatial metadata (in memory only, no tiled support).

pyspatial makes it easy to read, analyze, query, and visualize spatial data in both vector and raster form. It brings the familiarity of pandas to working with vector data, and provides querying capability similar to PostGIS for both vector and raster data.  Since it uses GDAL for much of the computations, the performance is quite good.  Based on the authors' experience, the performance has been significantly better than PostGIS, and orders of magnitude faster than similar libraries in R.



# Examples

http://nbviewer.jupyter.org/github/granularag/pyspatial/tree/master/examples/


# Development

* Python code: pip install -e /path/to/pyspatial
* Cython code: python setup.py build_ext --inplace
* Tests: nosetests -v

# Known Issues

* In ogr, when you get the centroid of a geometry (e.g. cent = my_geometry.Centroid()), cent does not inherit the spatial reference. It needs to be reassigned using cent.AssignSpatialReference(my_geometry.GetSpatialReference())
* VectorLayer object does not support a Float64Index
* If you encouter:
  * "TypeError: object of type 'Geometry' has no len()", most likely you have duplicate values in your index.  Make sure your index is unique.

# Contributors

* Aman Thakral (Lead, github: aman-thakral)
* Josh Daniel (github: joshdaniel)
* Chris Seifert (github: caseifert)
* Ron Potok (github: rpotok)
* Sandra Guteg (github: guetgs)
* Emma Fuller (github: emfuller)
