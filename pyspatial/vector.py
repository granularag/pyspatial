from urlparse import urlparse

import requests
import smart_open

import pandas as pd
from rtree import index
from osgeo.osr import CoordinateTransformation, SpatialReference
from osgeo import ogr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import shape
from shapely import wkb
from shapely.geometry import box
from shapely import ops
import pyspatial.utils as ut
from pyspatial.spatiallib import to_utm
from pyspatial.io import get_ogr_datasource


def to_shapely(feat):
    if isinstance(feat, BaseGeometry):
        return feat
    elif isinstance(feat, ogr.Feature):
        return wkb.loads(feat.geometry().ExportToWkb())
    elif isinstance(feat, list) or isinstance(feat, VectorLayer):
        if isinstance(feat[0], ogr.Feature):
            return [wkb.loads(f.geometry().ExportToWkb()) for f in feat]
        elif isinstance(feat[0], ogr.Geometry):
            return [wkb.loads(f.ExportToWkb()) for f in feat]
    elif isinstance(feat, ogr.Geometry):
        return wkb.loads(feat.ExportToWkb())
    else:
        raise ValueError("Unable to convert to shapely object")


def to_geometry(shp, copy=False, proj=None):
    """Convert shp to a ogr.Geometry.

    Parameters
    ----------
    shp: ogr.Geometry, ogr.Feature, or shapely.BaseGeometry
        The shape you want to convert

    copy: boolean (default=False)
        Return a copy of the shape instead of a reference

    proj: str or osr.SpatialReference (default=None)
        The projection of the shape to define (if the shape is
        not projection aware), or transform to (if projection aware).
        If a string is provided, it assumes that it is in PROJ4.

    Returns
    -------
    ogr.Geometry"""

    target_proj = None
    source_proj = None

    # Check shape type
    if isinstance(shp, ogr.Geometry):
        geom = shp

    elif isinstance(shp, ogr.Feature):
        geom = shp.geometry()

    elif isinstance(shp, BaseGeometry):
        geom = ogr.CreateGeometryFromWkb(wkb.dumps(shp))
    else:
        raise ValueError("Unable to convert to ogr.Geometry object")

    # Check projection
    if isinstance(proj, str) or isinstance(proj, unicode):
        target_proj = SpatialReference()
        target_proj.ImportFromProj4(proj)

    elif isinstance(proj, SpatialReference):
        target_proj = proj

    elif proj is None:
        target_proj = geom.GetSpatialReference()
        if target_proj is None:
            raise ValueError("shp does not have a SpatialReference")
    else:
        raise ValueError("Unable to set projction.")

    # Return shapely
    if isinstance(shp, BaseGeometry):
        geom.AssignSpatialReference(proj)
        return geom

    if copy:
        geom_cp = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
        geom_cp.AssignSpatialReference(geom.GetSpatialReference())
        geom = geom_cp

    if proj is not None:
        source_proj = geom.GetSpatialReference()
        if source_proj is None:
            raise ValueError("shp does not have a SpatialReference")
        ct = CoordinateTransformation(source_proj, target_proj)
        geom.Transform(ct)
        geom.AssignSpatialReference(target_proj)

    return geom


def bounding_box(envelope):
    xmin, xmax, ymin, ymax = envelope
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(xmin, ymin)
    ring.AddPoint(xmin, ymax)
    ring.AddPoint(xmax, ymax)
    ring.AddPoint(xmax, ymin)
    ring.AddPoint(xmin, ymin)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def to_feature(shp, fid, proj=None):
    feature_def = ogr.FeatureDefn()
    feature = ogr.Feature(feature_def)
    feature.SetGeometry(to_geometry(shp, proj=proj))
    feature.SetFID(fid)
    return feature


BOOL_PREDICATES = ["intersects", "contains", "within", "crosses", "touches",
                   "equals", "disjoint"]


class VectorLayer(object):
    """
    Parameters
    ----------
    features: ogr.Feature[] or ogr.Geometry[], shapely.BaseGeometry[]

    proj: osr.SpatialReference
         The projection for the features

    index: iterable
        The index to use for the shapes

    Attributes
    ----------

    index: pandas.Index

   _sindex: rtree.index.Index
        The spatial index. Initially None, but can be built with build_sindex()

    features, proj, id: see parameters

    fields: str[]
        The names of the attributes

    """
    def __init__(self, features, proj, index):
        self._sindex = None

        if not hasattr(features, "__iter__"):
            raise ValueError("Features must be iterable.")

        if len(features) > 0:
            f0 = features[0]
            if isinstance(f0, ogr.Geometry) or isinstance(f0, BaseGeometry):
                fids = xrange(len(features))
                self.features = [to_feature(*f) for f in zip(features, fids)]
            elif isinstance(f0, ogr.Feature):
                self.features = features
            else:
                msg = "features of type %s not supported" % type(f0)
                raise ValueError(msg)
        else:
            self.features = features

        [f.geometry().FlattenTo2D() for f in self.features]
        self.proj = proj
        self.index = index

        self._id_to_features = dict(zip(self.index, self.features))
        if len(self.index) > 0:
            self._id_type = type(self.index[0])
            f = self.features[0]
            self.fields = [f.GetFieldDefnRef(i).name for i in
                           xrange(f.GetFieldCount())]
        else:
            self._id_type = None
            self.fields = []

    def __iter__(self):
        for f in self.features:
            yield f

    def __len__(self):
        return len(self.features)

    def _make_ids(self, ids):
        return pd.Index(ids)

    def filter_by_id(self, ids, inplace=False):
        """Return a vector layer with only those shapes with
        id in ids

        Parameters
        ----------
        ids: iterable
            The ids to filter on

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer."""

        assert hasattr(ids, "__iter__"), "ids must be iterable"
        if not isinstance(ids, pd.Index):
            ids = self._make_ids(ids)

        if inplace:
            self.features = self[ids]
            self.index = ids
            self._id_to_features = dict(zip(self.index, self.features))
            self._sindex = None
            self.build_sindex()
        else:
            features = [self._id_to_features[i].Clone() for i in ids]
            proj = SpatialReference()
            proj.ImportFromWkt(self.proj.ExportToWkt())
            vl = VectorLayer(features, proj, ids)
            return vl

    def __getitem__(self, keys):

        if isinstance(keys, slice):
            ids = self.index[keys]
            return self.filter_by_id(ids)

        if self._id_type is None:
            return self.filter_by_id([])

        if hasattr(keys, "__iter__"):
            return self.filter_by_id([k for k in keys])
        else:
            try:
                k = self._id_type(keys)
            except:
                raise ValueError("Invalid key type: %s" % keys)

            if k not in self._id_to_features:
                raise KeyError("Key not found: %s" % k)

            return self._id_to_features[self._id_type(keys)]

    def _get_index_intersection(self, shp):
        if self._sindex is None:
            self.build_sindex()

        shp = to_shapely(shp)

        if isinstance(shp, list):
            raise ValueError("Collections of shapes are not supported!")

        return shp, self._sindex.intersection(shp.bounds, objects="raw")

    def intersects(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that intersect with shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.intersects

        """
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if (self.to_geometry(i).
                                  Intersect(to_geometry(shp)))]

        ids = self._make_ids(ids)

        if index_only:
            return ids
        else:
            return self.filter_by_id(ids, inplace=inplace)

    def iintersects(self, shp):
        """Return an index with only those shapes in the
        vector layer that intersect with shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------

        pandas.Index
        """
        return self.intersects(shp, index_only=True, inplace=False)

    def contains(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that contain shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.contains

        """
        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if self.to_geometry(i).Contains(shp)]
        ids = self._make_ids(ids)

        if index_only:
            return ids
        else:
            return self.filter_by_id(ids, inplace=inplace)

    def icontains(self, shp):
        """Return an index with only those shapes in the
        vector layer that contain  shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------

        pandas.Index
        """
        return self.contains(shp, index_only=True, inplace=False)

    def within(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in
        the vector layer that are within shp.

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.within"""

        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if self.to_geometry(i).Within(shp)]
        ids = self._make_ids(ids)

        if index_only:
            return ids
        else:
            return self.filter_by_id(ids, inplace=inplace)

    def iwithin(self, shp):
        """Return an index with only those shapes in the
        vector layer that are within shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------

        pandas.Index"""
        return self.within(shp, index_only=True, inplace=False)

    def crosses(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that cross shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.crosses"""

        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if self.to_geometry(i).Crosses(shp)]
        ids = self._make_ids(ids)

        if index_only:
            return ids
        else:
            return self.filter_by_id(ids, inplace=inplace)

    def icrosses(self, shp):
        """Return an index with only those shapes in the
        vector layer that crosses shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------

        pandas.Index"""
        return self.crosses(shp, index_only=True, inplace=False)

    def touches(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that touches shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.touches
        """

        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if self.to_geometry(i).Touches(shp)]
        ids = self._make_ids(ids)

        if index_only:
            return ids
        else:
            return self.filter_by_id(ids, inplace=inplace)

    def itouches(self, shp):
        """Return an index with only those shapes in the
        vector layer that touches shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------

        pandas.Index"""
        return self.touches(shp)

    def equals(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that are equal shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#binary-predicates

        """

        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = [i for i in ids if self.to_geometry(i).Equals(shp)]
        ids = self._make_ids(ids)

        if index_only:
            return ids

        return self.filter_by_id(ids, inplace=inplace)

    def iequals(self, shp):
        """Return an index with only those shapes in the
        vector layer that equals shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------
        pandas.Index"""
        return self.equals(shp, index_only=True, inplace=False)

    def disjoint(self, shp, index_only=False, inplace=False):
        """Return a vector layer with only those shapes in the
        vector layer that are disjoint with shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        index_only: boolean (default False)
            Return the ids only (not a new vector layer)

        inplace: boolean (default False)
            Perform this operation in place (do not return a
            new vector layer.

        See Also
        --------
        http://toblerity.org/shapely/manual.html#object.disjoint
        """

        shp = to_geometry(shp)
        _shp, ids = self._get_index_intersection(shp)
        ids = self.index.difference(self._make_ids(ids))

        if index_only:
            return ids

        return self.filter_by_id(ids, inplace=inplace)

    def idisjoint(self, shp):
        """Return an index with only those shapes in the
        vector layer that is disjoint to shp

        Parameters
        ----------
        shp: shapely.BaseGeometry, ogr.Geometry, or ogr.Feature
            The shape to test against

        Returns
        -------
        pandas.Index"""
        return self.disjoint(shp, index_only=True, inplace=False)

    def intersection(self, shp):
        """
        Cut the shapes in the VectorLayer to match the intersection
        specified by shp.

        Parameters
        ----------
        shp: shapely geometry or ogr Feature/Geometry

        Returns
        -------
        VectorLayer interesected by the shape
        """

        vl = self.intersects(shp)
        shp = to_shapely(shp)

        if isinstance(shp, list):
            raise ValueError("Collections of shapes are not supported!")

        for id, feat in vl.iteritems():
            geom_wkb = wkb.dumps(shp.intersection(to_shapely(feat)))
            geom = ogr.CreateGeometryFromWkb(geom_wkb)
            geom.AssignSpatialReference(self.proj)
            feat.SetGeometry(geom)

        return vl

    def unary_union(self):
        return ops.unary_union(self.to_shapely())

    def is_valid(self, index_only=False):
        """
        Get vector layer with valid shapes.
        """
        ids = [i for i in self.index if self.to_geometry(i).IsValid]
        if index_only:
            return ids

        return self.filter_by_id(ids)

    def is_invalid(self, index_only=False):
        """
        Get vector layer with invalid shapes.
        """
        ids = [i for i in self.index if not self.to_geometry(i).IsValid]
        if index_only:
            return ids

        return self.filter_by_id(ids)

    def is_empty(self, index_only=False):
        """
        Get vector layer with the empty shapes
        """
        ids = [i for i in self.index if not self.to_geometry(i).IsEmpty]
        if index_only:
            return ids

        return self.filter_by_id(ids)

    def is_ring(self, index_only=False):
        """
        Get vector layer with the ring shapes
        """
        ids = [i for i in self.index if not self.to_geometry(i).IsRing]
        if index_only:
            return ids

        return self.filter_by_id(ids)

    def transform(self, target_proj, inplace=False):
        ct = CoordinateTransformation(self.proj, target_proj)
        if inplace:
            [f.geometry().Transform(ct) for f in self.features]
            self._sindex = None
            self.build_sindex()
        else:
            feats = [f.Clone() for f in self.features]
            [f.geometry().Transform(ct) for f in feats]
            return VectorLayer(feats, target_proj, self.index)

    def to_wgs84(self):
        """Transform the VectorLayer into WGS84"""
        proj = ut.projection_from_epsg()
        return self.transform(proj)

    def to_shapely(self, ids=None):
        if ids is None:
            s = [to_shapely(f) for f in self.features]
        else:
            if type(ids) in [str, unicode] and self._id_type in [str, unicode]:
                return to_shapely(self[self._id_type(ids)])
            elif hasattr(ids, "__iter__"):
                s = [to_shapely(self[self._id_type(id)]) for id in ids]
            else:
                return to_shapely(self[ids])

        return pd.Series(s, index=self.index)

    def to_geometry(self, ids=None, proj=None):
        if ids is None:
            s = [to_geometry(f, proj=proj, copy=True) for f in self.features]
        else:
            if type(ids) in [str, unicode] and self._id_type in [str, unicode]:
                return to_geometry(self[self._id_type(ids)], proj=proj,
                                   copy=True)
            elif hasattr(ids, "__iter__"):
                s = [to_geometry(self[self._id_type(id)], proj=proj, copy=True)
                     for id in ids]
            else:
                return to_geometry(self[ids], proj=proj, copy=True)

        return pd.Series(s, index=self.index)

    def map(self, f):
        """Apply a function, f, over all the features.  Returns
        A pandas.Series object"""
        data = map(f, self.features)
        return pd.Series(data, index=self.index)

    def areas(self, proj=None):
        """Compute the areas for each of the shapes in the vector
        layer

        Parameters
        ----------
        proj: string or osr.SpatialReference (default=None)
            valid strings are 'albers' or 'utm'. If None, no
            transformation of coordinates.

        Returns
        -------
        pandas.Series"""
        if proj is None:
            return self.to_geometry().map(lambda x: x.GetArea())

        if proj == 'utm':
            if self.proj.ExportToProj4().strip() != ut.PROJ_WGS84:
                vl = self.transform(ut.projection_from_string())
            else:
                vl = self

            shps = vl.to_shapely()
            areas = [ops.transform(to_utm, shp).area for shp in shps]
            s = pd.Series(areas, index=self.index)
            s.name = "area_sqr_m"
            return s

        elif proj == 'albers':
            proj = ut.projection_from_string(ut.ALBERS_N_AMERICA)

        return self.transform(proj).areas()

    def distances(self, shp, proj=None):
        """Compute the euclidean distances for each of the shapes in the vector
        layer. If proj is not none, it will transform shp into proj.

        Note: if shp is a shapely object, it is upto to the user
        to make sure shp is in the correct coordinate system.

        Parameters
        ----------
        proj: string or osr.SpatialReference (default=None)
            valid strings are 'albers' or 'utm'. If None, no
            transformation of coordinates.

        Returns
        -------
        pandas.Series"""

        if proj is None:
            shp = to_geometry(shp)
            return self.to_geometry(proj=proj).map(lambda x: x.Distance(shp))

        if proj == 'utm':
            if not self.proj.ExportToProj4() == ut.PROJ_WGS84:
                vl = self.transform(ut.projection_from_string())
            else:
                vl = self

            _shp = ops.transform(to_utm, to_shapely(shp))
            d = vl.to_shapely() \
                  .map(lambda x: ops.transform(to_utm, x).distance(_shp))
            s = pd.Series(d, index=self.index)
            return s

        elif proj == 'albers':
            proj = ut.projection_from_string(ut.ALBERS_N_AMERICA)

        shp = to_geometry(shp, copy=True, proj=proj)
        return self.to_geometry(proj=proj).map(lambda x: x.Distance(shp))

    def centroids(self, format="VectorLayer"):
        """Get a DataFrame with "x" and "y" columns for the
        centroid of each feature.

        Parameters
        ----------
        format: str (default='VectorLayer')
             Return type of the centroids.  available options are
             'Series', 'DataFrame', or 'VectorLayer'. 'Series'
             will a collection of (x, y) tuples.  'DataFrame' will
             be a DataFrame with columns 'x' and 'y'
        """

        formats = ["DataFrame", "VectorLayer", "Series"]

        if format in ("DataFrame", "Series"):
            data = (f.geometry().Centroid().GetPoints()[0] for f in
                    self.features)
            if format == "Series":
                return pd.Series(data, index=self.index)
            else:
                return pd.DataFrame(data, columns=["x", "y"], index=self.index)
        elif format == "VectorLayer":
            pts = [f.geometry().Centroid() for f in self.features]
            [p.AssignSpatialReference(self.proj) for p in pts]
            return VectorLayer(pts, self.proj, self.index)
        else:
            raise ValueError("format must be in %s" % formats)

    def envelopes(self):
        """The the envelope of each shape as xmin, xmax, ymin, ymax.
        Returns a pandas.Series."""
        data = (f.geometry().GetEnvelope() for f in self.features)
        return pd.Series(data, index=self.index)

    def boundingboxes(self):
        """Return a VectorLayer with the bounding boxes of each
        geometry"""
        geoms = self.envelopes().map(bounding_box)
        [g.AssignSpatialReference(self.proj) for g in geoms]
        return VectorLayer(geoms, self.proj, self.index)

    def upper_left_corners(self):
        """Get a DataFrame with "x" and "y" columns for the
        min_lon, max_lat of each feature"""
        data = [(f.geometry().GetEnvelope()[0], f.geometry().GetEnvelope()[3])
                for f in self.features]
        return pd.DataFrame(data, columns=["x", "y"], index=self.index)

    def size_bytes(self):
        """Get the size of the geometry in bytes"""
        return self.map(lambda x: x.geometry().WkbSize())

    def get_extent(self):
        """The xmin, xmax, ymin, ymax values of the layer"""
        if self._sindex is None:
            self.build_sindex()
        xmin, ymin, xmax, ymax = self._sindex.get_bounds()
        return (xmin, xmax, ymin, ymax)

    def bbox(self):
        """Return a shapely poly representing the bounding box of the layer"""
        (xmin, xmax, ymin, ymax) = self.get_extent()
        return to_geometry(box(xmin, ymin, xmax, ymax), proj=self.proj)

    def keys(self):
        return [i for i in self.index]

    def values(self):
        return [self[i] for i in self.index]

    def iteritems(self):
        return ((k, v) for k, v in self._id_to_features.iteritems())

    def _gen_index(self):
        ix = xrange(len(self.features))
        for i, id, feat in zip(ix, self.index, self.features):
            xmin, xmax, ymin, ymax = feat.geometry().GetEnvelope()
            yield (i, (xmin, ymin, xmax, ymax), id)

    def build_sindex(self):
        if self._sindex is None:
            self._sindex = index.Index(self._gen_index())

    def nearest(self, shp, max_neighbors=5):
        if isinstance(shp, BaseGeometry):
            xmin, ymin, xmax, ymax = shp.bounds
        elif isinstance(shp, ogr.Geometry):
            xmin, xmax, ymin, ymax = shp.GetEnvelope()
        elif isinstance(shp, ogr.Feature):
            xmin, xmax, ymin, ymax = shp.geometry().GetEnvelope()
        else:
            raise ValueError("Unable to compute bounds")

        self.build_sindex()

        neighbors = self._sindex.nearest((xmin, ymin, xmax, ymax),
                                         max_neighbors,
                                         objects="raw")
        ret = []
        i = 0
        while i < max_neighbors:
            try:
                ret.append(neighbors.next())
            except StopIteration:
                i = max_neighbors
            i += 1

        return ret

    def sort(self, kind="upper_left_corners", columns=["y", "x"],
             ascending=True, inplace=False, index_only=False):
        """Sort the vector layer by upper_left_corners or centroids

        Parameters
        ----------
        kind: str
            Either "upper_left_corners" or "centroids"

        columns : list of str (default ["y", "x"]
            Order in which to sort the shapes (ie. sort primarily by y-axis
            or x-axis).  Will be passed to pandas.DataFrame.sort.

        ascending : boolean (default True)
            Sort by columns in ascending or descending order.


        Returns
        -------
        list
            Shape ids sorted by columns.
        """

        kinds = set(["upper_left_corners", "centroids"])
        col_msg = "Sort columns must be in ['x', 'y']"
        assert all([c in ["x", "y"] for c in columns]), col_msg
        assert kind in kinds, "Sort kind not in %s" % ",".join(kinds)

        # Get dataframe with min_lon, max_lat for all shapes.
        df = getattr(self, kind)()
        df.sort(columns=columns, ascending=ascending, inplace=True)

        if index_only:
            return df.index

        return self.filter_by_id(df.index, inplace=inplace)

    def to_dict(self, df=None):
        """Return a dictionary representation of the object.
        Based off the GeoJSON spec.  Will transform the vector
        layer into WGS84 (EPSG:4326).

        Parameters
        ----------
        df: pandas.DataFrame (default=None)
            The dataframe to supply the properties of the features.
            The index of df must match the ids of the VectorLayer.

        Returns
        -------
        dict
        """

        if self.proj.ExportToProj4() != ut.projection_from_string():
            vl = self.transform(ut.projection_from_string())
        else:
            vl = self

        res = {"type": "FeatureCollection"}
        res["features"] = [f.ExportToJson(as_object=True) for f in vl.features]

        if df is not None:
            for i, f in zip(vl.ids, res["features"]):
                props = f["properties"]
                df_props = df.loc[i].to_dict()
                f["properties"] = dict(props.items() + df_props.items())
                f["properties"]["__id__"] = i
        else:
            for i, f in zip(vl.ids, res["features"]):
                f["properties"]["__id__"] = i

        return res

    @property
    def ids(self):
        return self.index

    def to_json(self, path=None, df=None, precision=6):
        """Return the layer as a GeoJSON.  If a path is provided,
        it will save to the path. Otherwise, will return a string.

        Parameters
        ----------
        path: str, (default=None)
            The path to save the geojson data.
        df: pandas.DataFrame (default=None)
            The dataframe to supply the properties of the features.
            The index of df must match the ids of the VectorLayer.

        precision: int
            Number of decimal places to keep for floats

        Returns
        -------
        geojson string
        """
        res = self.to_dict(df=df)
        s = pd.io.json.dumps(res, double_precision=precision)
        if path is None:
            return s
        else:
            with smart_open.smart_open(path, 'wb') as outf:
                outf.write(s)

    def to_svg(self, ids=None):
        """Return svg represention. ids can be one or an
        iterable of the layers 'ids' attribute.  If ids is None,
        returns a list of svg strings"""

        if ids is None:
            ids = self.index

        if hasattr(ids, "__iter__"):
            return map(ut.to_svg, self.to_shapely(ids))
        else:
            return ut.to_svg(to_shapely(self[ids]))


def fetch_geojson(path):
    url = urlparse(path)
    if "http" in url.scheme:
        geojson = requests.get(path).text
    elif "s3" in url.scheme or url.scheme == "" or url.scheme == "file":
        with smart_open.smart_open(path) as inf:
            geojson = inf.read()
    else:
        return path
    return geojson


def read_datasource(ds, layer=0, index=None):
    dslayer = ds.GetLayerByIndex(layer)

    features = [dslayer.GetFeature(i) for i in
                xrange(dslayer.GetFeatureCount())]

    if index is None:
        ids = pd.Index([f.GetFID() for f in features])
    elif isinstance(index, str) or isinstance(index, unicode):
        ids = pd.Index([f[index] for f in features])
    elif hasattr(index, "__iter__"):
        ids = pd.Index(index)
    else:
        raise ValueError("Unable to create index.")

    if len(ids) != len(features):
        msg = "index length doesn't match number of shapes: %d vs %d."
        raise ValueError(msg % (len(ids), len(features)))

    rows = [f.items() for f in features]
    df = pd.DataFrame(rows, index=ids)
    geoms = [to_geometry(f, copy=True) for f in features]
    proj = ut.get_projection(dslayer)
    ds = None
    return VectorLayer(geoms, proj, ids), df


def read_layer(path, layer=0, index=None):
    """Create a vector layer from the specified path.
    Will try to read using ogr.OpenShared.

    Parameters
    ----------
    path_or_str: string
        path or json string

    layer: integer
        The layer number to use.  Use ogrinfo to see
        the available layers.

    index: string or iterable (default=None)
        If string, the column in the "properties" of each feature to use
        as the index. If iterable, use the iterable as the index. If not
        specified, will create an integer based index.

    Returns
    -------
    Tuple of (VectorLayer, pandas.DataFrame of properties)
    """
    ds = get_ogr_datasource(path)
    return read_datasource(ds, layer=layer, index=index)


def read_geojson(path_or_str, index=None):
    """Create a vector layer from a geojson object

    Parameters
    ----------
    path_or_str: string
        path or json string

    index: string or iterable
        If string, the column in the "properties" of each feature to use
        as the index. If iterable, use the iterable as the index.


    Returns
    -------

    Tuple of (VectorLayer, pandas.DataFrame of properties)"""

    if "FeatureCollection" not in path_or_str:
        geojson_str = fetch_geojson(path_or_str)
    else:
        geojson_str = path_or_str

    feats = pd.io.json.loads(geojson_str)["features"]

    if index is None:
        try:
            ids = map(lambda x: x["id"], feats)
        except KeyError:
            ids = range(len(feats))

        name = "index"
    elif isinstance(index, str) or isinstance(index, unicode):
        ids = map(lambda x: x["properties"][index], feats)
        name = index
    else:
        raise ValueError("Unable to create index.")

    proj = ut.projection_from_epsg()
    props = pd.DataFrame(map(lambda x: x["properties"], feats), index=ids)
    geoms = pd.Series(map(lambda x: shape(x["geometry"]), feats), index=ids) \
              .map(lambda x: to_geometry(x, proj=proj))

    props.index.name = name
    geoms.index.name = name

    return VectorLayer(geoms, proj, geoms.index), props


def from_series(geom_series, proj=None):
    """Create a VectorLayer from a pandas.Series object

    Parameters
    ----------
    geom_series: pandas.Series
        The series object with shapely geometries

    proj: osr.SpatialReference
        The projection to use, defaults to EPSG:4326


    Returns
    -------
    VectorLayer

    """

    fids = range(len(geom_series))
    proj = ut.projection_from_string() if proj is None else proj
    features = [to_feature(shp, fid, proj=proj) for shp, fid
                in zip(geom_series, fids)]
    return VectorLayer(features, proj, geom_series.index)
