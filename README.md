# Development

Python code: pip install -e /path/to/ana-spatial
Cython code: python setup.py build_ext --inplace
Tests: nosetests -v

# Overview:

pyspatial is python package to provide data structures on top of gdal/ogr.  The 3 core data stuctures are:

* VectorLayer: a collection of shapes with pandas like manipulation
* RasterDataset: an abstraction of a spatial raster (both tiled on untiled) to support querying of pixels that intersect with shapes.
* RasterBand: a numpy array representation of a raster with spatial metadata

pyspatial makes it easy to read, analyze, query, and manipulate spatial data in both vector and raster form.

# Examples
