import os
import numpy as np

from pyspatial.raster import rasterize, read_catalog
from pyspatial.raster import RasterBand
from pyspatial.vector import read_layer


cwd = os.getcwd()
base = os.path.abspath(os.path.dirname(__file__))


def get_path(x):
    return os.path.join(base, "data", x)


class TestRasterDatasetTiled:
    @classmethod
    def setup_class(cls):
        os.chdir(base)
        # Let's add some CLUs for Washington County, IL
        cls.vl, _ = read_layer(get_path("vector/clu/clu_public_a_il189.shp"),
                               index="uniqueid")

        # Figure out which raster files we'll need.
        cls.dataset = read_catalog(get_path("../catalog/cdl_2014.json"))

        # Create a RasterBand for the raster. Raster data is stored
        # and read from there.
        cls.rb = RasterBand(get_path("raster/95000_45000.tif"))

        # Convert CLUs to pixels.
        cls.px_shps = cls.dataset.to_pixels(cls.vl)

    @classmethod
    def teardown_class(cls):
        os.chdir(cwd)

    def test_single_shape_should_have_equivalent_counts(self):

        # Read in raster projection (applies to all raster tiles).
        rast_proj = self.dataset.proj

        # Compute counts for a single shape using single-tile computation for reference.
        shp = self.px_shps[0]
        mask = rasterize(shp, ext_outline=0, int_outline=1).T
        minx, miny, maxx, maxy = shp.bounds
        pts = (np.argwhere(mask>0) + np.array([minx, miny])).astype(int)
        counts_reference = np.bincount(self.rb[pts[:,1],pts[:, 0]], minlength=256)

        # Now, compute counts for a single shape using RasterDataset, and compare against single-tile computation.
        values = np.array(self.dataset.get_values_for_pixels(pts)).astype(np.uint8)
        counts = np.bincount(values, minlength=256)
        assert(np.array_equal(counts, counts_reference))

    # Compare all shapes between single-tile computation and RasterDataset computation.
    def test_all_shapes_should_have_equivalent_counts(self):

        # Read in raster projection (applies to all raster tiles).
        rast_proj = self.dataset.proj

        # All shapes in the layer should already be in pixel coords. Check that we have the right number of them.
        assert (len(self.px_shps) == 23403)

        # Compute counts for a single shape using single-tile computation for reference.
        for i, shp in enumerate(self.px_shps):
            # Rasterize the shape, and find list of all points. This is the same for either case.
            mask = rasterize(shp, ext_outline=0, int_outline=1).T
            minx, miny, maxx, maxy = shp.bounds
            pts = (np.argwhere(mask>0) + np.array([minx, miny])).astype(int)

            # Tiled data (in RasterDataset): Compute counts.
            values_td = np.array(self.dataset.get_values_for_pixels(pts)).astype(np.uint8)
            counts_td = np.bincount(values_td, minlength=256)

            # Single-tile Dataset: Compute counts for reference, and compare for equality.
            counts_ref = np.bincount(self.rb[pts[:,1],pts[:, 0]], minlength=256)
            assert(np.array_equal(counts_td, counts_ref))
