# Overview

pyspatial is python package to provide data structures on top of gdal/ogr. Its core use cases have been around simplifying geospatial data science workflows in Python.  The 3 core data stuctures are:

* VectorLayer: a collection of geometries with pandas like manipulation.  Each geometry is an osgeo.ogr.Geometry object. For an object reference [see this link](http://gdal.org/python/).
* RasterDataset: an abstraction of a spatial raster (both tiled on untiled) to support querying of pixels that intersect with shapes.
* TiledWebRaster: an abstraction of a tiled spatial raster typically used for visualization on the web (e.g. openlayers or google maps) (Still in testing)
* RasterBand: a numpy array representation of a raster with spatial metadata (in memory only, no tiled support).

pyspatial makes it easy to read, analyze, query, and visualize spatial data in both vector and raster form. It brings the familiarity of pandas to working with vector data, and provides querying capability similar to PostGIS for both vector and raster data.  Since it uses GDAL for much of the computations, the performance is quite good.  Based on the authors' experience, the performance has been significantly better than PostGIS, and orders of magnitude faster than similar libraries in R.

*Documentation is available [here](http://pyspatial.readthedocs.io/)*

## Library Highlights
  * Battle tested: we use it for our day-to-day work, and for processing all the data behind [AcreValue](https://www.acrevalue.com/).  In fact, all of our PostGIS workflows have been migrated to pyspatial.
  * Read/write both raster and vector data (including support for http/s3/google cloud sources).  Also convert to/from shapely/gdal/ogr/numpy objects seamlessly.
  * Fast spatial queries since it leverages GDAL and libspatialindex/RTree. For extracting vector data from a raster, the library is 60x - 100x faster than R.
  * Integration of vector/raster data structures to make interoperation seamless.
  * Pandas-like API for working with collections of geometries.
  * First class support for spatial projections. The data structures are spatial projection aware, and allow you to easily transform between projections.
  * When performing operations between data sources, the data will automatically be reprojected intelligently.  No more spatial projection management!
  * Integrated interactive visualization within IPython (via Leaflet).  Plots markers, geometries, and choropleths too!

## Docker

```
docker pull amanthakral/pyspatial
```
[See docker/README.md for build/run instructions](docker/README.md)

## Examples

Plese see this [link](http://nbviewer.jupyter.org/github/granularag/pyspatial/tree/master/examples/).

## Questions?

Send us a message on google groups: [pyspatial-users@googlegroups.com](https://groups.google.com/forum/#!forum/pyspatial-users)

## Development

* Python code: pip install -e /path/to/pyspatial
* Cython code: python setup.py build_ext --inplace
* Tests: nosetests -v

## TODOs
* Adjust timed tests.  Currently calibrated to a early Macbook Pro 15" with Core i7.  These tend to fail on many other machines.  Should either remove the @timed decorators, or figure out what a reasonable time is for the tests.

## Known Issues

* In ogr, when you get the centroid of a geometry (e.g. cent = my_geometry.Centroid()), cent does not inherit the spatial reference. It needs to be reassigned using cent.AssignSpatialReference(my_geometry.GetSpatialReference())
* VectorLayer object does not support a Float64Index
* If you encouter:
  * "TypeError: object of type 'Geometry' has no len()", most likely you have duplicate values in your index.  Make sure your index is unique.
* On some environments, calling shape.Intersection(point) for certain shape/point combinations causes python to crash. See [this gist](https://gist.github.com/sandra-granular/c5009e189d842ddf72878c41df77e03c)

## Contributors

* Aman Thakral (Lead, github: aman-thakral)
* Josh Daniel (github: joshdaniel)
* Chris Seifert (github: caseifert)
* Ron Potok (github: rpotok)
* Sandra Guteg (github: guetgs)
* Emma Fuller (github: emfuller)
* Alan Glennon (github: glennon)
* James Russell (github: jamesdrussell)
