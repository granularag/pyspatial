from urlparse import urlparse
import os
from osgeo import ogr
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly

gdal.SetConfigOption('GDAL_HTTP_UNSAFSSL', 'YES')
gdal.SetConfigOption('CPL_VSIL_ZIP_ALLOWED_EXTENSION', 'YES')
gdal.SetConfigOption('CPL_CURL_GZIP', 'YES')
gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
gdal.SetConfigOption('CPL_VSIL_CURL_USE_HEAD', 'NO')


class PyspatialIOError(Exception):
    pass


def get_path(path, use_streaming=False):
    """Read a shapefile from local or an http source"""
    url = urlparse(path)

    prefix = ""
    if url.path.endswith("zip"):
        prefix += "/vsizip/"

    if "http" in url.scheme:
        curl = "/vsicurl" if prefix == "" else "vsicurl"
        if use_streaming:
            curl += "_streaming"
        prefix = os.path.join(prefix, curl)

    if not path.startswith("/"):
        path = os.path.join(prefix, path)
    else:
        path = prefix+path
    return path


def get_ogr_datasource(path, use_streaming=False):
    path = get_path(path, use_streaming=use_streaming)
    ds = ogr.OpenShared(path, update=False)
    if ds is None:
        raise PyspatialIOError("Unable to read path: %s" % path)
    return ds


def get_gdal_datasource(path):
    path = get_path(path)
    ds = gdal.OpenShared(path, GA_ReadOnly)
    if ds is None:
        raise PyspatialIOError("Unable to read path: %s" % path)
    return ds
