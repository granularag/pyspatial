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
    #@timed(8)
    #def test_perf_areas_utm(self):
    #    assert_almost_equal(self.clus.areas("utm").sum() * M_TO_MI,
    #                        537.78827998574002)

if __name__ == "__main__":
    import nose
    nose.main()
