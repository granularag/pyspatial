import os
import argparse
import json
from pyspatial.vector import read_layer, read_geojson
from pyspatial.utils import projection_from_string
import gdal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create json catalog file.')
    parser.add_argument('src', help='The source raster file, whether tiled '
                                    'or untiled')

    parser.add_argument('--dest', dest="dest",
                        help='The output path for the json file',
                        default=None)
    parser.add_argument('--tiles', dest='tile_path', default=None,
                        help='Specify path to tiles')

    parser.add_argument('--index', dest='index_path', default=None,
                        help='Specify the index file for the grid')

    parser.add_argument('--grid', dest='grid_size', type=int, default=None,
                        help=('Specify the grid size in pixels '
                              '(assumes both x and y are the same)'))

    args = parser.parse_args()

    hDataset = gdal.OpenShared(args.src)

    # Get projection.
    proj = hDataset.GetProjectionRef()

    # Dump to json
    catalog = {"Path": args.src,
               "CoordinateSystem": proj,
               "GeoTransform": hDataset.GetGeoTransform()}

    band = hDataset.GetRasterBand(1)
    ctable = band.GetColorTable()

    if ctable is not None:
        colors = [ctable.GetColorEntry(i) for i in range(256)]
        catalog["ColorTable"] = colors

    xsize = hDataset.RasterXSize
    ysize = hDataset.RasterYSize
    catalog["Size"] = (xsize, ysize)

    if args.tile_path is not None:
        if os.path.exists(args.tile_path):
            tiles = os.listdir(args.tile_path)
            if len(tiles) == 0:
                raise ValueError("%s is empty" % args.tile_path)

            tile = os.path.join(args.tile_path, tiles[0])
            ds = gdal.OpenShared(tile)
            if ds is None:
                raise ValueError("Unable to open file: %s" % tile)

            xsize = ds.RasterXSize
            ysize = ds.RasterYSize
            if xsize != ysize:
                raise ValueError("tiles must have same X and Y size")

            catalog["GridSize"] = xsize
            catalog["Path"] = args.tile_path
        else:
            raise ValueError("tiles path does not exist: %s" % args.tile_path)

    if args.index_path is not None:
        read = read_geojson if args.index_path.endswith("json") else read_layer
        index = read(args.index_path)
        catalog["Index"] = index.transform(projection_from_string()).to_dict()

    if args.dest is not None:
        with open(args.dest, "w+b") as outf:
            outf.write(json.dumps(catalog))
    else:
        print json.dumps(catalog)
