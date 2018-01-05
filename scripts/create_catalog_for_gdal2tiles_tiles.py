
import os
import sys
import json
import math
import pprint
import argparse
import logging
# log = logging.getLogger('geos.py')
# log.propagate = False
log = logging.getLogger(__file__)
# logging.basicConfig(stream = sys.stderr, level=logging.DEBUG, format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')
logging.basicConfig(stream = sys.stderr, level=logging.DEBUG, format='%(lineno)s %(levelname)s:%(message)s')
log.setLevel(logging.INFO)

from osgeo import gdal

import pygeoj

from pyspatial.vector import read_layer, read_geojson
from pyspatial.utils import projection_from_string
from pyspatial import globalmaptiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create json catalog file and geojson file.')
    parser.add_argument('tiles_path', help='The path to a zoom level of some gdal2tiles tiles')

    parser.add_argument('--dest_catalog', dest="catalog_filename",
                        help='The output path for the json catalog file',
                        default=None)

    parser.add_argument('--dest_geojson', dest="geojson_filename",
                        help='The output path for the geojson file',
                        default=None)

    parser.add_argument('--tiles_size', dest='tiles_size', type=int, default=256,
                        help=('Specify the tiles size in pixels '
                              '(assumes both x and y are the same)'))

    args = parser.parse_args()


    ######################## Creating the geojson file
    geojson = pygeoj.new()
    geojson.define_crs(type="name", name="urn:ogc:def:crs:OGC:1.3:CRS84")

    g = globalmaptiles.GlobalMercator()

    for dirname, subdirList, filename_list in os.walk(args.tiles_path):
        for filename in filename_list:
            if filename.endswith('.png'):
                rel_path = '.' + os.sep + os.path.join(dirname, filename)
                tms_z, tms_x, tms_y = rel_path.rstrip('.png').split(os.sep)[-3:]
                tms_z, tms_x, tms_y = list(map(int, (tms_z, tms_x, tms_y)))
                minLat, minLon, maxLat, maxLon = g.TileLatLonBounds(tms_x, tms_y, tms_z)
                # minLat, minLon = lower left corner
                # maxLat, maxLon = upper right corner
                tile_upper_left_corner = minLon, maxLat
                tile_upper_right_corner = maxLon, maxLat
                tile_lower_right_corner = maxLon, minLat
                tile_lower_left_corner = minLon, minLat
                log.debug("tile_upper_left_corner " + str(tile_upper_left_corner) )
                log.debug("tile_upper_right_corner " + str(tile_upper_right_corner) )
                log.debug("tile_lower_right_corner " + str(tile_lower_right_corner) )
                log.debug("tile_lower_left_corner " + str(tile_lower_left_corner) )

                feature = pygeoj.Feature(geometry={"type":"Polygon", "coordinates":[[tile_upper_left_corner, 
                                                                            tile_upper_right_corner, 
                                                                            tile_lower_right_corner, 
                                                                            tile_lower_left_corner, 
                                                                            tile_upper_left_corner]]},
                                    properties={"location": rel_path })
                geojson.add_feature(feature)

    # geojson.add_all_bboxes()
    # geojson.add_unique_id()
    # geojson.save(args.geojson_filename, indent=4) # geojson.update_bbox() inside

    geojson.update_bbox()
    if args.geojson_filename is not None:
        with open(args.geojson_filename, "w+b") as outf:
            outf.write(json.dumps(geojson._data, indent=4, sort_keys=True))
    else:
        print(json.dumps(geojson._data, indent=4, sort_keys=True))


    ######################## Creating the catalog file

    minLon_raster, minLat_raster, maxLon_raster, maxLat_raster = geojson.bbox
    log.debug("geojson.bbox " + str(geojson.bbox) )
    raster_upper_left_corner = minLon_raster, maxLat_raster
    raster_upper_right_corner = maxLon_raster, maxLat_raster
    raster_lower_right_corner = maxLon_raster, minLat_raster
    raster_lower_left_corner = minLon_raster, minLat_raster
                
    log.debug("raster_upper_left_corner " + str(raster_upper_left_corner) )
    log.debug("raster_lower_right_corner " + str(raster_lower_right_corner) )

    # for i in range(len(geojson)):
    #     feature = geojson.get_feature(i)
    #     print feature.geometry.bbox

    minLon_tile, minLat_tile, maxLon_tile, maxLat_tile = geojson.get_feature(0).geometry.bbox
    nb_tiles_x = (minLon_raster - maxLon_raster) / (minLon_tile - maxLon_tile)
    nb_tiles_y = (minLat_raster - maxLat_raster) / (minLat_tile - maxLat_tile)
    log.debug('nb_tiles_x %s => %s' % (nb_tiles_x, round(nb_tiles_x)) )
    log.debug('nb_tiles_y %s => %s' % (nb_tiles_y, round(nb_tiles_y)) )
    nb_tiles_x = int(round(nb_tiles_x))
    nb_tiles_y = int(round(nb_tiles_y))

    raster_size_x = nb_tiles_x * args.tiles_size
    raster_size_y = nb_tiles_y * args.tiles_size

    geotrans_left_value = minLon_raster
    geotrans_delta_x = (maxLon_raster - minLon_raster) / raster_size_x
    geotrans_rotation_x = 0.0
    geotrans_top_value = maxLat_raster
    geotrans_rotation_y = 0.0
    geotrans_delta_y = (minLat_raster - maxLat_raster) / raster_size_y
    geoTransform = (geotrans_left_value, geotrans_delta_x,    geotrans_rotation_x,
                    geotrans_top_value,  geotrans_rotation_y, geotrans_delta_y)
    log.debug("geoTransform\n " + pprint.pformat(geoTransform) )

    assert raster_lower_right_corner[0] == geotrans_left_value + raster_size_x * geotrans_delta_x
    assert raster_lower_right_corner[1] == geotrans_top_value  + raster_size_y * geotrans_delta_y

    coordinate_system_EPSG4326 = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433],
        AUTHORITY["EPSG","4326"]]
    """

    # https://en.wikipedia.org/wiki/Web_Mercator
    coordinate_system_EPSG3857 = """
    PROJCS["WGS 84 / Pseudo-Mercator",
        GEOGCS["WGS 84",
            DATUM["WGS_1984",
                SPHEROID["WGS 84",6378137,298.257223563,
                    AUTHORITY["EPSG","7030"]],
                AUTHORITY["EPSG","6326"]],
            PRIMEM["Greenwich",0,
                AUTHORITY["EPSG","8901"]],
            UNIT["degree",0.0174532925199433,
                AUTHORITY["EPSG","9122"]],
            AUTHORITY["EPSG","4326"]],
        PROJECTION["Mercator_1SP"],
        PARAMETER["central_meridian",0],
        PARAMETER["scale_factor",1],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["metre",1,
            AUTHORITY["EPSG","9001"]],
        AXIS["X",EAST],
        AXIS["Y",NORTH],
        EXTENSION["PROJ4","+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext  +no_defs"],
        AUTHORITY["EPSG","3857"]]
    """

    catalog = { "CoordinateSystem": coordinate_system_EPSG4326.replace(' ','').replace('\n',''),
        "GeoTransform": geoTransform,
        "GridSize": args.tiles_size,
        "Path": args.tiles_path,
        "Size": [ raster_size_x, raster_size_y],
        "Tile_structure": "%d/%d.png",
        "TMS_z": tms_z
    }

    if args.catalog_filename is not None:
        with open(args.catalog_filename, "w+b") as outf:
            outf.write(json.dumps(catalog, indent=4, sort_keys=True))
    else:
        print(json.dumps(catalog, indent=4, sort_keys=True))

