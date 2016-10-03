from pyspatial.globalmaptiles import GlobalMercator
from pyspatial.raster import TiledWebRaster

zoom = 13
twr = TiledWebRaster("/tmp/95000_45000_rgb/", zoom)


def test_point():
    zoom = 13
    lat, lng = 38.76479194327964,-89.23713684082031  # lat, lng of Carlyle Lake in MO
    # The color should be blue
    # Testing that pixel coords match
    GM = GlobalMercator()
    mx, my = GM.LatLonToMeters(lat, lng)
    mx, my = GM.LatLonToMeters(lat, lng)
    assert twr._to_pixels(mx, my) == GM.MetersToPixels(mx, my, zoom)
