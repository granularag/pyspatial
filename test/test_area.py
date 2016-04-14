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
from pyspatial import vector as vt
from nose.tools import assert_almost_equal
from nose.tools import timed

# Conversion sq. m -> sq. mi
M_TO_MI = 3.8610216e-07


class TestVectorLayer:
    @classmethod
    def setup_class(cls):
        cls.base_dir = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(cls.base_dir, "data/vector")
        path = os.path.join(path, "bay_area_counties.geojson")
        cls.counties, cls.counties_df = vt.read_geojson(path, index="NAME")
        clu_path = "data/vector/clu/clu_public_a_il189.shp"
        cls.clus, _ = vt.read_layer(os.path.join(cls.base_dir, clu_path))
        il_path = os.path.join(cls.base_dir, "data/vector/il.geojson")
        cls.il, cls.il_df = vt.read_geojson(il_path, index="name")

    def test_areas(self):
        vl = self.counties[["San Francisco"]]

        # Testing UTM
        assert_almost_equal(vl.areas("utm").sum() * M_TO_MI, 49.6650636)

        # Testing Albers projection
        assert_almost_equal(vl.areas("albers").sum() * M_TO_MI, 49.701871750)

        vl = self.il[["Washington County, IL"]]
        assert_almost_equal(vl.areas("albers").sum() * M_TO_MI, 564.022347393)

    @timed(0.5)
    def test_perf_areas_albers(self):
        assert_almost_equal(self.clus.areas("albers").sum() * M_TO_MI,
                            537.630561662)

    # Currently really slow
    @timed(8)
    def test_perf_areas_utm(self):
        assert_almost_equal(self.clus.areas("utm").sum() * M_TO_MI,
                            537.78827998574002)

if __name__ == "__main__":
    import nose
    nose.main()
