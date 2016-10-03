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

import os
from pyspatial.vector import read_geojson, to_geometry
from pyspatial.raster import read_raster
import numpy as np
from numpy.testing import assert_array_almost_equal
from test_raster_query import compute_stats

base_dir = os.path.abspath(os.path.dirname(__file__))
get_path = lambda x: os.path.join(base_dir, "data", x)
vl, vldf = read_geojson(get_path("vector/small_polygon.geojson"))
rd = read_raster(get_path("raster/95000_45000.tif"))
shp = vl[0]


def test_small_polygon():
    bboxes = vl.boundingboxes()
    shp_px = rd.to_pixels(bboxes)[0]

    grid = rd.to_geometry_grid(*shp_px.bounds)

    areas = {}
    for i, b in grid.items():
        diff = b.Intersection(to_geometry(shp, proj=rd.proj))
        areas[i] = diff.GetArea()

    total_area = sum(areas.values())

    index = areas.keys()
    if total_area > 0:
        weights = np.array([areas[k]/total_area for k in index])
    else:
        weights = np.zeros(len(index))

    assert abs(total_area - shp.GetArea()) < 1e-8
    assert abs(1 - sum(weights)) < 1e-8
    values = rd.get_values_for_pixels(np.array(index))
    assert_array_almost_equal(values, np.array([121, 176, 176, 176],
                                               dtype=np.uint8))

    classes = np.arange(0, 256)
    exp_stats = [0.114183, 0.885817]
    exp_classes = [121, 176]
    for r in rd.query(vl):
        stats = compute_stats(r.values, r.weights)
        assert_array_almost_equal(exp_classes, classes[stats > 0])
        assert_array_almost_equal(exp_stats, stats[stats > 0])
