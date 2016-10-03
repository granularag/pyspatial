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

import json
import os
import math
import re
from uuid import uuid4
from six import string_types
#Scipy
import numpy as np
from skimage.transform import downscale_local_mean

#Geo
from osgeo import gdal, osr
from osgeo.gdalconst import GA_ReadOnly
from osgeo.osr import SpatialReference
from shapely import wkb, ops
from shapely.affinity import scale
from shapely.geometry import box
from skimage.io import imsave

from PIL import Image, ImageDraw
from pyspatial import fileutils

from pyspatial import spatiallib as slib
from pyspatial.vector import read_geojson, to_geometry, bounding_box
from pyspatial.vector import VectorLayer
from pyspatial.utils import projection_from_epsg
from pyspatial import globalmaptiles

NP2GDAL_CONVERSION = {
    "uint8": 1,
    "uint16": 2,
    "int16": 3,
    "uint32": 4,
    "int32": 5,
    "float32": 6,
    "float64": 7,
    "complex64": 10,
    "complex128": 11,
}

GDAL2NP_CONVERSION = {v: k for k, v in NP2GDAL_CONVERSION.items()}
TILE_REGEX = re.compile('([0-9]+)_([0-9]+)\.tif')


def rasterize(shp, ext_outline=False, ext_fill=True, int_outline=False,
              int_fill=False, scale_factor=4):

    """Convert a vector shape to a raster. Assumes the shape has already
    been transformed in to a pixel based coordinate system. The algorithm
    checks for the intersection of each point in the shape with
    a pixel grid created by the bounds of the shape. Partial overlaps
    are estimated by scaling the image in X and Y by the scale factor,
    rasterizing the shape, and downscaling (using mean), back to the
    bounds of the original shape.

    Parameters
    ----------

    shp: shapely.Polygon or Multipolygon
        The shape to rasterize

    ext_outline: boolean (default False)
        Include the outline of the shape in the raster

    ext_fill: boolean (default True)
        Fill the shape in the raster

    int_outline: booelan (default False)
        Include the outline of the interior shapes

    int_fill: boolean (default False):
        Fill the interior shapes

    scale_factor: int (default 4)
        The amount to scale the shape in X, Y before downscaling. The
        higher this number, the more precise the estimate of the overlap.


    Returns
    -------
    np.ndarray representing the rasterized shape.
    """
    sf = scale_factor

    minx, miny, maxx, maxy = map(int, shp.bounds)
    if minx == maxx and miny == maxy:
        return np.array([[1.]])

    elif maxy > miny and minx == maxx:
        n = maxy - miny + 1
        return np.zeros([n, 1]) + 1./n

    elif maxy == miny and minx < maxx:
        n = maxx - minx + 1
        return np.zeros([1, n]) + 1./n

    if ((maxx - minx + 1) + (maxy - miny + 1)) <= 2*sf:
        sf = 1.0

    shp = scale(shp, xfact=sf, yfact=sf)
    minx, miny, maxx, maxy = shp.bounds
    width = int(maxx - minx + 1)
    height = int(maxy - miny + 1)

    img = Image.new('L', (width, height), 0)
    _shp = shp.geoms if hasattr(shp, "geoms") else [shp]

    ext_outline = int(ext_outline)
    ext_fill = int(ext_fill)
    int_outline = int(int_outline)
    int_fill = int(int_fill)

    for pg in _shp:
        ext_pg = [(x-minx, y-miny) for x, y in pg.exterior.coords]
        ImageDraw.Draw(img).polygon(ext_pg, outline=ext_outline, fill=ext_fill)
        for s in pg.interiors:
            int_pg = [(x-minx, y-miny) for x, y in s.coords]
            ImageDraw.Draw(img).polygon(int_pg, outline=int_outline,
                                        fill=int_fill)

    return downscale_local_mean(np.array(img), (sf, sf))


class RasterBase(object):
    """
    Provides methods and attributes common to both RasterBand and
    RasterDataset, particularly for converting shapes to pixels
    in the raster coordinate space. Stores a coordinate system for a raster.

    Parameters
    ----------
    RasterXSize, RasterYSize: int
        Number of pixels in the width and height respectively.

    geo_transform : list of float
        GDAL coefficients for GeoTransform (defines boundaries and pixel size
        for a raster in lat/lon space).

    proj: osr.SpatialReference
        The spatial projection for the raster.

    Attributes
    ----------
    xsize, ysize: int
        Number of pixels in the width and height respectively.

    geo_transform : list of float
        GDAL coefficients for GeoTransform (defines boundaries and pixel size
        for a raster in lat/lon space).

    min_lon: float
         The minimum longitude in proj coordinates

    min_lat: float
         The minimum latitude in proj coordinates

    max_lat: float
         The maximum latitude in proj coordinates

    lon_px_size: float
         Horizontal size of the pixel

    lat_px_size: float
         Vertical size of the pixel

    proj: osr.SpatialReference
         The spatial projection for the raster.
    """
    def __init__(self, RasterXSize, RasterYSize, geo_transform, proj):
        self.geo_transform = geo_transform
        self.xsize = RasterXSize
        self.ysize = RasterYSize
        self.RasterXSize = self.xsize
        self.RasterYSize = self.ysize
        self.min_lon = self.geo_transform[0]
        self.max_lat = self.geo_transform[3]
        self.min_lat = self.geo_transform[3] + self.geo_transform[5]*self.ysize
        self.lon_px_size = abs(self.geo_transform[1])
        self.lat_px_size = self.geo_transform[5]
        self.pixel_area = abs(self.lon_px_size * self.lat_px_size)
        self.proj = proj

    def _to_pixels(self, lon, lat, alt=None):
        """Convert a point from lon/lat to pixel coordinates.  Note,
        the altitude is currently ignored.

        Parameters
        ----------
        lon: float
            Longitude of point

        lat: float
            Latitude of point

        Returns
        -------
        list of int
            (longitude in pixel space, latitude in pixel space).
            Rounded to the nearest pixel.
        """
        lon_px, lat_px = slib.to_pixels(lon, lat, self.min_lon,
                                        self.max_lat, self.lon_px_size,
                                        self.lat_px_size)
        return int(lon_px), int(lat_px)

    def shape_to_pixel(self, geom):
        """Takes a feature and returns a shapely object transformed into the
        pixel coords.

        Parameters
        ----------
        feat : osgeo.ogr.Geometry
            Feature to be transformed.

        Returns
        -------
        shapely.Polygon
            Feature in pixel coordinates.
        """
        shp = wkb.loads(geom.ExportToWkb())
        return ops.transform(self._to_pixels, shp)

    def to_pixels(self, vector_layer):
        """Takes a vector layer and returns list of shapely geometry
        transformed in pixel coordinates. If the projection of the
        vector_layer is different than the raster band projection, it
        transforms the coordinates first to raster projection.

        Parameters
        ----------
        vector_layer : VectorLayer
            Shapes to be transformed.

        Returns
        -------
        list of shapely.Polygon
            Shapes in pixel coordinates.
        """
        if self.proj.ExportToProj4() != vector_layer.proj.ExportToProj4():
            vector_layer = vector_layer.transform(self.proj)
        return [self.shape_to_pixel(geom) for geom in vector_layer]

    def to_raster_coord(self, pxx, pxy):
        """Convert pixel corrdinates -> raster coordinates"""
        if not (0 <= pxx < self.RasterXSize):
            raise ValueError("Invalid x coordinate: %s" % pxx)

        if not (0 <= pxy < self.RasterYSize):
            raise ValueError("Invalid x coordinate: %s" % pxx)

        # urx, ury are the upper right coordinates
        # xsize, ysize, are the pixel sizes
        urx, xsize, _, ury, _, ysize = self.geo_transform
        return (urx + pxx * xsize, ury + ysize * pxy)

    def to_geometry_grid(self,  minx, miny, maxx, maxy):
        """Convert pixels into a geometry grid. All values should be in
        pixel cooridnates.

        Returns
        -------
        VectorLayer with index a tuple of the upper left corner coordinate
        of each pixel.
        """

        xs = np.arange(minx, maxx+1)
        ys = np.arange(miny, maxy+1)

        x, y = np.meshgrid(xs, ys)
        index = []
        boxes = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x1, y1 = self.to_raster_coord(x[i, j], y[i, j])
                x2, y2 = self.to_raster_coord(x[i, j] + 1, y[i, j] + 1)
                boxes.append(bounding_box((x1, x2, y1, y2), self.proj))
                index.append((int(x[i, j]), int(y[i, j])))

        return VectorLayer(boxes, index=index, proj=self.proj)

    def GetGeoTransform(self):
        """Returns affine transform from GDAL for describing the relationship
        between raster positions (in pixel/line coordinates) and georeferenced
        coordinates.

        Returns
        -------
        min_lon: float
             The minimum longitude in raster coordinates.

        lon_px_size: float
             Horizontal size of each pixel.

        geo_transform[2] : float
            Not used in our case. In general, this would be used if the
            coordinate system had rotation or shearing.

        max_lat: float
             The maximum latitude in raster coordinates.

        lat_px_size: float
             Vertical size of the pixel.

        geo_transform[5] : float
            Not used in our case. In general, this would be used if the
            coordinate system had rotation or shearing.

        References
        ----------
        http://www.gdal.org/gdal_tutorial.html
        """
        return self.geo_transform

    def get_extent(self):
        """Returns extent in raster coordinates.

        Returns
        -------
        xmin : float
            Minimum x-value (lon) of extent in raster coordinates.

        xmax : float
            Maximum x-value (lon) of extent in raster coordinates.

        ymin : float
            Minimum y-value (lat) of extent in raster coordinates.

        ymax : float
            Maximum y-value (lat) of extent in raster coordinates.
        """
        ymax = self.max_lat
        xmin = self.min_lon
        ymin = ymax + self.lat_px_size*self.ysize
        xmax = xmin + self.lon_px_size*self.xsize
        return (xmin, xmax, ymin, ymax)

    def bbox(self):
        """Returns bounding box of raster in raster coordinates.

        Returns
        -------
        ogr.Geometry
            Bounding box in raster coordinates:
                (xmin : float
                    minimum longitude (leftmost)

                 ymin : float
                    minimum latitude (bottom)

                 xmax : float
                    maximum longitude (rightmost)

                 ymax : float
                    maximum latitude (top))
        """
        (xmin, xmax, ymin, ymax) = self.get_extent()
        return to_geometry(box(xmin, ymin, xmax, ymax),
                           proj=self.proj)


class RasterBand(RasterBase, np.ndarray):
    def __new__(cls, ds, band_number=1):
        """
        Create an in-memory representation for a single band in
        a raster. (0,0) in pixel coordinates represents the
        upper left corner of the raster which corresponds to
        (min_lon, max_lat).  Inherits from ndarray, so you can
        use it like a numpy array.

        Parameters
        ----------
        ds: gdal.Dataset

        band_number: int
            The band number to use

        Attributes
        ----------
        data: np.ndarray[xsize, ysize]
             The raster data
        """

        if not isinstance(ds, gdal.Dataset):
            path = fileutils.get_path(ds)
            ds = gdal.Open(path, GA_ReadOnly)

        band = ds.GetRasterBand(band_number)
        if band is None:
            msg = "Unable to load band %d " % band_number
            msg += "in raster %s" % ds.GetDescription()
            raise ValueError(msg)

        gdal_type = band.DataType
        dtype = np.dtype(GDAL2NP_CONVERSION[gdal_type])
        self = np.asarray(band.ReadAsArray().astype(dtype)).view(cls)

        self.gdal_type = gdal_type
        proj = SpatialReference()
        proj.ImportFromWkt(ds.GetProjection())
        geo_transform = ds.GetGeoTransform()

        # Initialize the base class with coordinate information.
        RasterBase.__init__(self, ds.RasterXSize, ds.RasterYSize,
                            geo_transform, proj)

        self.nan = band.GetNoDataValue()

        #self = np.ma.masked_equal(self, band.GetNoDataValue(), copy=False)
        ctable = band.GetColorTable()
        if ctable is not None:
            self.colors = np.array([ctable.GetColorEntry(i)
                                    for i in range(256)],
                                   dtype=np.uint8)
        else:
            self.colors = None

        ds = None
        return self

    def __init__(self, ds, band_number=1):
        pass

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        cols = ['xsize', 'ysize', 'geo_transform',
                'RasterXSize', 'RasterYSize',
                'min_lon', 'min_lat', 'max_lat',
                'lon_px_size', 'lat_px_size',
                'pixel_area', 'proj', 'gdal_type',
                'colors', 'nan']

        for c in cols:
            setattr(self, c, getattr(obj, c, None))

    def to_gdal(self, driver="MEM", path=''):
        """Convert to a gdal dataset."""
        drv = gdal.GetDriverByName(driver)
        ds = drv.Create(path, self.xsize, self.ysize, 1, self.gdal_type)
        ds.SetGeoTransform(self.GetGeoTransform())
        ds.SetProjection(self.proj.ExportToWkt())
        band = ds.GetRasterBand(1)
        if self.colors is not None:
            ctable = gdal.ColorTable()
            for i, c in enumerate(self.colors):
                ctable.SetColorEntry(i, tuple(c))
                band.SetColorTable(ctable)
        band.WriteArray(self)
        band.FlushCache()
        return ds

    def transform(self, proj, size=None, method="nneighbour"):
        """
        A sample function to reproject and resample a GDAL dataset from within
        Python. The idea here is to reproject from one system to another, as well
        as to change the pixel size. The procedure is slightly long-winded, but
        goes like this:

        1. Set up the two Spatial Reference systems.
        2. Open the original dataset, and get the geotransform
        3. Calculate bounds of new geotransform by projecting the UL corners
        4. Calculate the number of pixels with the new projection & spacing
        5. Create an in-memory raster dataset
        6. Perform the projection
        """
        methods = {"mean": gdal.GRA_Average,
                   "bilinear": gdal.GRA_Bilinear,
                   "cubic": gdal.GRA_Cubic,
                   "cubic-spline": gdal.GRA_CubicSpline,
                   "lanczos": gdal.GRA_Lanczos,
                   "mode": gdal.GRA_Mode,
                   "nneighbour": gdal.GRA_NearestNeighbour}

        if method not in methods:
            raise ValueError("methods must be one of: %s" % methods.keys())

        tx = osr.CoordinateTransformation(self.proj, proj)

        geo_t = self.GetGeoTransform()
        if size is None:
            x_size = self.RasterXSize
            y_size = self.RasterYSize
        else:
            x_size, y_size = size

        # Work out the boundaries of the new dataset in the target projection
        (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
        (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1]*x_size,
                                            geo_t[3] + geo_t[5]*y_size)
        x_px_size = (lrx - ulx) * 1./x_size
        y_px_size = (lry - uly) * 1./y_size

        mem_drv = gdal.GetDriverByName('MEM')
        src = self.to_gdal()

        dest = mem_drv.Create('', x_size, y_size, 1, self.gdal_type)
        # Calculate the new geotransform
        new_geo = (ulx, x_px_size, geo_t[2], uly, geo_t[4], y_px_size)
        # Set the geotransform
        dest.SetGeoTransform(new_geo)
        dest.SetProjection(proj.ExportToWkt())

        # Perform the projection/resampling
        res = gdal.ReprojectImage(src, dest, self.proj.ExportToWkt(),
                                  proj.ExportToWkt(),
                                  methods[method])
        assert res == 0
        rb = RasterBand(dest)
        rb.colors = np.copy(self.colors)
        dest = None
        src = None
        return rb

    def to_wgs84(self, method="nneighbour"):
        return self.transform(projection_from_epsg(4326), method=method)

    def save(self, path, format="GTiff"):
        self.to_gdal(format, path)

    def to_rgb(self):
        return slib.create_image_array(self, self.colors)

    def save_png(self, path):
        imsave(path, self.to_rgb())


class RasterQueryResult:
    """
    Container class to hold the result of a raster query.

    Attributes
    ----------
    id : str or int
        The id of the shape in the vector layer

    coordinates : np.ndarray
        The requested raster pixel coordinates

    values: np.ndarray
        The values of the intersected pixels in the raster

    weights: np.ndarray
        The fraction of the polygon intersecting with the pixel
    """
    def __init__(self, id, coordinates, values, weights):
        self.id = id
        self.coordinates = coordinates
        self.values = values
        self.weights = weights


class RasterDataset(RasterBase):
    """
    Raster representation that supports tiled and untiled datasets, and
    allows querying with a set of shapes.

    Raster may be tiled or untiled. A RasterDataset
    object may be queried one or multiple times with a set of shapes (in a
    VectorLayer). We also try to match the attribute and method names of
    gdal.Dataset when possible to allow for easy porting of caller code
    to use this class instead of gdal.Dataset, as this class transparently
    works on an untiled gdal.Dataset, in addition to the added functionality
    to handle tiled datasets.

    Parameters
    ----------
    path_or_ds: str or gdal.Dataset

    xsize: int
         The number of pixels in the X coordinate

    ysize: int
         The number of pixels in the Y coordinate

    geo_transform: tuple

    proj: osr.SpatialReference
        The spa
    grid_size: int (default=None)
        Number of pixels for each tile. Assumes that each tile is square.

    index: dict (default=None)
        Dictionary matching the Geojson spec describing boundary of each file.
        Typically generated by gdaltindex.

    tile_structure: str (default="%d_%d.tif")
        A string describing the file structure and the format of the tiles.
        In case of use of gdal2tiles tiles, this must be set to "%d/%d.png".

    Attributes
    ----------
    path : str
        Path to raster data files.

    grid_size : int
        Number of pixels in width or height of each grid tile. If set to None,
        that indicates this is an untiled raster.

    raster_bands : RasterBand, or
                    dict of (list of int): RasterBand
        Dictionary storing raster arrays that have been read from disk.
        If untiled, this is set at initialization to the whole raster. If
        tiled, these are read in lazily as needed. Index is (x_grid, y_grid)
        where x_grid is x coordinate of leftmost pixel in this tile relative
        to minLon (and is a multiple of grid_size), and y_grid is y coordinate
        of uppermost pixel in this tile relative to maxLat (and is also a
        multiple of grid_size). See notes below for more information on how
        data is represented here.

    shapes_in_tiles : dict of (int, int): set of str
        What shapes are left to be processed in each tile. Key is (minx, maxy)
        of tile (upper left corner), and value is set of ids of shapes. This
        is initially set in query(), and shape ids are removed from this data
        structure for a tile once they have been processed. Tiles can be
        cleared from memory when there are no shapes left in their set.

    Notes
    -----
    Raster representation (tiled and untiled):
    The core functionality of this class is to look up pixel values (for a
    shape or set of shapes) in the raster. To do this, we store the raster in
    a 2D-array of pixels relative to (0,0) being the upper left corner aka
    (min_lon, max_lat) in lon/lat coordinates. We can then convert vector
    shapes into pixel space, and look up their values directly in the raster
    array. For an untiled raster, we can read in the raster directly during
    initialization.

    Tiled representation:
    For a tiled dataset (ie. the data is split into multiple files), we
    still treat the overall raster upper left corner as (0,0), and
    recognize that each tile has a position relative to the overall raster
    pixel array. We store each tile in a 2D array in a dictionary keyed by
    the tile position relative to the overall raster position in pixel space.
    For example, a pixel at (118, 243) in a tiled dataset with grid size = 100
    would be stored in raster_bands[(100, 200)][18][43]. As a memory
    utilization and performance enhancement, we lazily read tiles from disk
    when they are first needed and store them in raster_bands{} (for the
    lifetime of the RasterDataset object). If memory turns out to be a
    problem, it might make sense to store these in a LRU cache instead.

    TODOs
    -----

    * Added band number
    * Add support for color tables and raster attributes
    """

    def __init__(self, path_or_ds, xsize, ysize, geo_transform, proj,
                 tile_regex=TILE_REGEX, grid_size=None, index=None,
                 tile_structure=None, tms_z=None):

        ds = None
        self.band_count = None

        if not isinstance(path_or_ds, gdal.Dataset):
            path = path_or_ds
        else:
            ds = path_or_ds
            path = ds.GetDescription()
            self.band_count = ds.RasterCount

        self.path = path
        self.proj = proj
        self.grid_size = grid_size
        self.index = index

        if tile_structure:
            self.tile_structure = tile_structure
        else:
            self.tile_structure = "%d_%d.tif"

        self.raster_bands = {}
        self.shapes_in_tiles = {}
        self.tile_regex = tile_regex
        self.index = index
        self.grid_size = grid_size
        self.dtype = None

        # Initialize the base class with coordinate information.
        RasterBase.__init__(self, xsize, ysize, geo_transform, proj)

        # Read raster file now if this is an untiled data set.
        if self.grid_size is None:
            if ds is None:
                self.raster_bands = read_vsimem(self.path)
            else:
                if self.band_count == 1:
                    self.raster_bands = RasterBand(ds)
                    self.dtype = self.raster_bands.dtype
                else:
                    self.raster_bands = [RasterBand(ds, band_number=(i+1))
                                         for i in range(self.band_count)]
                    self.dtype = [r.dtype for r in self.raster_bands]

        if self.tile_structure:
            self.tile_regex = self.tile_structure.replace('%d','([0-9]+)').replace('.','\.')
            self.tile_regex = re.compile(self.tile_regex)

        ds = None
        path_or_ds = None

    def _get_value_for_pixel(self, px):
        """Look up value for a pixel in raster space.

        Parameters
        ----------
        px : np.array
            Pixel coordinates for 1 point: [x_coord, y_coord]

        Returns
        -------
        dtype
            Value in raster at pixel coordinates specified by px.
            Type is determined by GDAL2NP_CONVERSION from RasterBand data
            type.

        """
        x_grid, y_grid = self._get_grid_for_pixel(px)

        # Look up the value in the x,y offset in the grid tile we just found
        # or read, and return it.
        x_px = px[0] - x_grid
        y_px = px[1] - y_grid

        # If we haven't already read this grid tile into memory, do so now,
        # and store it in raster_bands for future queries to access.
        if (x_grid, y_grid) not in self.raster_bands:
            filename = self.path + self.tile_structure % (x_grid, y_grid)
            self.raster_bands[(x_grid, y_grid)] = read_vsimem(filename)
            if self.dtype is None:
                self.dtype = self.raster_bands[(x_grid, y_grid)].dtype

        # Look up the grid tile for this pixel.
        raster = self.raster_bands[(x_grid, y_grid)]

        return raster[y_px][x_px]

    def _get_grid_for_pixel(self, px):
        """Compute the min_x, min_y of the tile that contains pixel,
        which can also be used for looking up the tile in raster_bands.

        Parameters
        ----------
        px : np.array
            Pixel coordinates for 1 point: [x_coord, y_coord]

        Returns
        -------
        list of int
            (min_x, min_y) of tile that contains px, in pixel coordinates.

        """
        return slib.grid_for_pixel(self.grid_size, px[0], px[1])

    def get_values_for_pixels(self, pxs):
        """Look up values for a list of pixels in raster space.

        Parameters
        ----------
        pxs : np.array
            Array of pixel coordinates. Each row is [x_coord, y_coord]
            for one point.

        Returns
        -------
        list of dtype
            List of values in raster at pixel coordinates specified in pxs.
            Type is determined by GDAL2NP_CONVERSION from RasterBand data
            type.
        """
        # Untiled case: Use the 1-file raster array we read in at
        # initialization.
        if self.grid_size is None:
            if isinstance(self.raster_bands, list):
                return [r[pxs[:, 1], pxs[:, 0]] for r in self.raster_bands]
            else:
                return self.raster_bands[pxs[:, 1], pxs[:, 0]]
        # Tiled case: Compute the grid tile to read, and the x,y offset in
        # that tile.
        else:
            return np.array([self._get_value_for_pixel(px) for px in pxs], dtype=self.dtype)

    def _key_from_tile_filename(self, filename):
        """Get (x_grid, y_grid) key of upper left corner of tile from filename.

        Parameters
        ----------
        filename : str
            Tile filename. We assume filename is in format given by argument tile_structure,
            which is by default "%d_%d.tif" :
            'arbitrary_path/{x_grid}_{y_grid}.tif'
            e.g. 'data/tiled/2500_2250.tif'
            Path does not matter, but format after last slash is assumed.

        Returns
        -------
        x_grid : int
            minimum value for x for tile in raster pixel coordinates.

        y_grid : int
            minimum value for y for tile in raster pixel coordinates.
        """
        r = self.tile_regex.search(filename)
        if (len(r.groups()) != 2):
            raise ValueError("Tile filenames must comply with the given tile_structure : " + self.tile_structure)
        x_grid = int(r.group(1))
        y_grid = int(r.group(2))
        return x_grid, y_grid

    def _small_pixel_query(self, shp, shp_px):

        grid = self.to_geometry_grid(*shp_px.bounds)
        areas = {}
        for i, b in grid.items():

            if b.Intersects(to_geometry(shp, proj=self.proj)):
                diff = b.Intersection(to_geometry(shp, proj=self.proj))
                areas[i] = diff.GetArea()

        index = areas.keys()
        total_area = sum(areas.values())

        if total_area > 0:
            weights = np.array([areas[k]/self.pixel_area for k in index])
        else:
            weights = np.zeros(len(index))

        values = self.get_values_for_pixels(np.array(index))
        return values, weights

    def query(self, vector_layer, ext_outline=False, ext_fill=True,
              int_outline=False, int_fill=False, scale_factor=4,
              missing_first=False, small_polygon_pixels=4):
        """
        Query the dataset with a set of shapes (in a VectorLayer). The
        vectors will be reprojected into the projection of the raster. Any
        shapes in the vector layer that are not within the bounds of the
        raster will return with values and weights as np.array([]).

        Parameters
        ----------
        vector_layer : VectorLayer
            Set of shapes in vector format, with ids attached to each.
        ext_outline: boolean (default False)
            Include the outline of the shape in the raster
        ext_fill: boolean (default True)
            Fill the shape in the raster
        int_outline: booelan (default False)
            Include the outline of the interior shapes
        int_fill: boolean (default False):
            Fill the interior shapes
        scale_factor: int (default 4)
            The amount to scale the shape in X, Y before downscaling. The
            higher this number, the more precise the estimate of the overlap.

        missing_first: boolean (default false)
            Where the missing values should be at the beginning or the
            end of the results.

        small_polygon_pixels: integer (default 4)
            Number of pixels for the intersection of the polygon with the
            raster to be considered "small".  This is a slow step that computes
            the exact intersection between the polygon and the raster in the
            cooridate space of the raster (not pixel space!).

        Yields
        ------

        RasterQueryResult.  This is 3 attributes: id, values, weights.  The
        values are the pixel values from the raster.  the weights are the fraction
        of the pixel that is occupied by the polgon.
        """

        if self.proj.ExportToProj4() != vector_layer.proj.ExportToProj4():
            # Transform all vector shapes into raster projection.
            vl = vector_layer.transform(self.proj)
        else:
            vl = vector_layer

        # Filter out all shapes outside the raster bounds
        bbox = self.bbox()
        vl = vl.within(bbox)

        ids_to_tiles = None
        tiles_to_ids = None

        # Removing this for now.  It needs more thought!
        # Optimization to minimize memory usage if the RasterDataset contains
        # an index.  This will sort by the upper left corners of all the shapes
        # and process one shape at a time.  It will remove the corresponding
        # entries in self.raster_bands once all references for shapes in a
        # particular tile have been removed.
        #if self.index is not None:
        #    res = {self._key_from_tile_filename(id): set(vl.intersects(f).ids)
        #           for id, f in self.index.items()}

        #    tiles_to_ids = {k: v for k, v in res.items() if len(v) > 0}
        #    ids_to_tiles = defaultdict(set)
        #    for tile, shp_ids in tiles_to_ids.items():
        #        for id in shp_ids:
        #            ids_to_tiles[id].add(tile)

        #    vl = vl.sort()

        missing = vector_layer.index.difference(vl.index)

        if missing_first:
            ids = missing.append(vl.ids)
        else:
            ids = vl.ids.append(missing)

        px_shps = dict(zip(vl.ids, self.to_pixels(vl)))

        for id in ids:
            shp = px_shps.get(id, None)

            if shp is None:
                yield RasterQueryResult(id, np.array([]), [], np.array([]))

            else:
                #Eagerly load tiles
                #if ids_to_tiles is not None:
                #    for key in list(ids_to_tiles[id]):
                #        if key not in self.raster_bands:
                #            filename = self.path + "%d_%d.tif" % key
                #            self.raster_bands[key] = RasterBand(filename)
                #        tiles_to_ids[key].remove(id)

                # Check for small polygon since rasterizing a polygon
                # doesn't work for small polygons
                if vl[id].GetArea() < small_polygon_pixels * self.pixel_area:
                    values, weights = self._small_pixel_query(vl[id], shp)
                    coordinates = + np.array([])
                else:
                    # Rasterize the shape, and find list of all points.
                    mask = rasterize(shp, ext_outline=ext_outline,
                                     ext_fill=ext_fill,
                                     int_outline=int_outline,
                                     int_fill=int_fill,
                                     scale_factor=scale_factor).T

                    minx, miny, maxx, maxy = shp.bounds
                    idx = np.argwhere(mask > 0)

                    if idx.shape[0] == 0:
                        weights = mask[[0]]
                    else:
                        weights = mask[idx[:, 0], idx[:, 1]]

                    coordinates = (idx + np.array([minx, miny])).astype(int)
                    values = self.get_values_for_pixels(coordinates)

                yield RasterQueryResult(id, coordinates, values, weights)

            #if tiles_to_ids is None:
            #   continue

            #Remove raster bands that are empty
            #empty = [k for k, v in tiles_to_ids.items() if len(v) == 0]
            #for e in empty:
            #    del self.raster_bands[e]
            #    del tiles_to_ids[e]


class TiledWebRaster(RasterDataset):
    """
    Raster representation for tiled data sets commonly used on the web
    (e.g OpenLayers, GoogleMaps, etc.).  These have been assumed to be
    produced using the gdal2tiles.py script (found in /scripts dir).

    Assumes that the tiles are projected in Popular Visualisation CRS / Mercator
    (EPSG:3785)

    Parameters
    ----------
    path: str
       The path to the tiled data
    zoom: int
       Zoom to use, typically a value from 6 to 15.

    tile_size: int
       n x n size of each tile in pixels, default is 256

    bands: list of ints
       The bands to use.  Default is [1, 2, 3]

    xy_tile_path: str
       The tile path structure, default="%s/%s.png"

    Attributes
    ----------
    path : str
        Path to raster data files.

    resoltution : float
        Number of meters in both x & y that each pixel represents.
        For example, at a zoom of 11, each pixel is approx 76 m x 76 m.

    Notes
    -----
    Since these datasets are typically png files, there will be a substantial
    performance hit due to the CPU overhead of decompression
    """
    def __init__(self, path, zoom, tile_size=256, bands=None,
                 xy_tile_path="%s/%s.png"):
        self.gm = globalmaptiles.GlobalMercator(tileSize=256)
        self.resolution = self.gm.Resolution(zoom)
        self.zoom = zoom
        self.bands = range(1, 4) if bands is None else bands
        self.tile_size = tile_size
        xsize = 2**zoom*tile_size
        ysize = xsize
        max_lat = 20037508.342789244  # 1/2 the curcumference of the earther in meters
        min_lon = -20037508.342789244
        geo_transform = (min_lon, self.resolution, 0., max_lat, 0., -self.resolution)
        proj = projection_from_epsg(3785)  # Sphereical Mercator
        grid_size = (2*max_lat/self.resolution, 2*max_lat/self.resolution)
        tile_structure = "%d/" % zoom + xy_tile_path
        super(TiledWebRaster, self).__init__(path, xsize, ysize, geo_transform, proj,
                                             grid_size=grid_size, tile_structure=tile_structure)

    def _to_pixels(self, lat, lon, alt=None):
        a, b = super(TiledWebRaster, self)._to_pixels(lat, lon, alt=alt)
        return a, self.ysize - b

    def _get_value_for_pixel(self, px):
        """Look up value for a pixel in raster space.

        Parameters
        ----------
        px : np.array
            Pixel coordinates for 1 point: [x_coord, y_coord]

        """
        x_grid, y_grid = self.gm.PixelsToTile(px[0], px[1])

        # Look up the value in the x,y offset in the grid tile we just found
        # or read, and return it.
        x_px = px[0] - x_grid*self.tile_size
        y_px = px[1] - y_grid*self.tile_size

        # If we haven't already read this grid tile into memory, do so now,
        # and store it in raster_bands for future queries to access.
        if (x_grid, y_grid) not in self.raster_bands:
            filename = self.path + self.tile_structure % (x_grid, y_grid)
            self.raster_bands[(x_grid, y_grid)] = [read_vsimem(filename, b) for b in self.bands]
            if self.dtype is None:
                self.dtype = [r.dtype for r in self.raster_bands[(x_grid, y_grid)]]

        # Look up the grid tile for this pixel.
        raster = self.raster_bands[(x_grid, y_grid)]
        try:
            return [r[y_px][x_px] for r in raster]
        except:
            # TODO: get a default nan value for each dtype
            return [0 for i in raster]

    def get_values_for_pixels(self, pxs):
        """Look up values for a list of pixels in raster space.

        Parameters
        ----------
        pxs : np.array
            Array of pixel coordinates. Each row is [x_coord, y_coord]
            for one point.

        Returns
        -------
        list of dtype
            List of values in raster at pixel coordinates specified in pxs.
            Type is determined by GDAL2NP_CONVERSION from RasterBand data
            type.
        """
        values = [self._get_value_for_pixel(px) for px in pxs]
        n = len(values)
        m = len(self.bands)
        x = [[values[i][j] for i in range(n)] for j in range(m)]
        return [np.array(x[i], dtype=self.dtype[i]) for i in range(m)]


def read_catalog(dataset_catalog_filename_or_handle, workdir=None):
    """Take a catalog file and create a raster dataset

    Parameters
    ----------
    dataset_catalog_filename_or_handle : str or opened file handle
        if str : Path to catalog file for the dataset. May be relative or absolute.

        Catalog files are in json format, and usually represent a type of data
        (e.g. CDL) and a year (e.g. 2014).

    Returns
    -------
    RasterDataset

    See Also
    --------
    scripts/create_catalog.py : How to create a catalog file for a dataset.
    raster_query_test.py : Simple examples of exercising RasterQuery on tiled
        and untiled datasets, and computing stats from results.
    vector.py : Details of VectorLayer."""
    if isinstance(dataset_catalog_filename_or_handle, string_types):
        with open(dataset_catalog_filename_or_handle) as catalog_file:
            decoded = json.load(catalog_file)
    else:
        decoded = json.load(dataset_catalog_filename_or_handle)

    size = [int(x) for x in decoded["Size"]]
    coordinate_system = str(decoded["CoordinateSystem"])
    transform = decoded["GeoTransform"]

    # Get the projection for the raster
    proj = osr.SpatialReference()
    proj.ImportFromWkt(coordinate_system)

    if workdir is None:
        path = decoded["Path"]
    else:
        path = os.path.join(workdir, decoded["Path"])

    grid_size = decoded.get("GridSize", None)
    index = None
    tile_structure = None

    if "Index" in decoded:
        index, index_df = read_geojson(json.dumps(decoded["Index"]),
                                       index="location")
        index = index.transform(proj)

    if "Tile_structure" in decoded:
        tile_structure = decoded["Tile_structure"]

    return RasterDataset(path, size[0], size[1],
                         geo_transform=transform,
                         proj=proj,
                         grid_size=grid_size,
                         index=index,
                         tile_structure=tile_structure)


def read_raster(path, band_number=1):
    """
    Create a raster dataset from a single raster file

    Parameters
    ----------
    path: string
        Path to the raster file.  Can be either local or s3/gs.

    band_number: int
        The band number to use

    Returns
    -------

    RasterDataset
    """

    path = fileutils.get_path(path)
    ds = gdal.Open(path, GA_ReadOnly)
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    proj = SpatialReference()
    proj.ImportFromWkt(ds.GetProjection())
    geo_transform = ds.GetGeoTransform()
    return RasterDataset(ds, xsize, ysize, geo_transform, proj)


def read_band(path, band_number=1):
    """
    Read a single band from a raster into memory.

    Parameters
    ----------
    path: string
        Path to the raster file.  Can be either local or s3/gs.

    band_number: int
        The band number to use

    Returns
    -------

    RasterBand
    """

    path = fileutils.get_path(path)
    ds = gdal.Open(path, GA_ReadOnly)
    return RasterBand(ds, band_number=band_number)


def read_vsimem(path, band_number=1):
    """
    Read a single band into memory from a raster. This method
    does not support all raster formats, only those that are
    supported by /vsimem

    Parameters
    ----------
    path: string
        Path to the raster file.  Can be either local or s3/gs.

    band_number: int
        The band number to use

    Returns
    -------

    RasterBand
    """
    filename = str(uuid4())
    with fileutils.open(path) as inf:
        gdal.FileFromMemBuffer("/vsimem/%s" % filename,
                               inf.read())

    ds = gdal.Open("/vsimem/%s" % filename)
    gdal.Unlink("/vsimem/%s" % filename)
    return RasterBand(ds, band_number=band_number)
