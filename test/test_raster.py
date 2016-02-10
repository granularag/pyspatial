import os
import pyspatial.raster as rst

base = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(base, "data/raster/prism.tif")


def test_read_raster():
    rd = rst.read_raster(filename)
    assert isinstance(rd, rst.RasterDataset)


def test_read_vsimem():
    rb = rst.read_vsimem(filename)
    assert isinstance(rb, rst.RasterBand)


def test_read_band():
    rb = rst.read_band(filename)
    assert isinstance(rb, rst.RasterBand)
