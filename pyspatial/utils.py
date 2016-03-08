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

import numpy as np
from shapely.geometry import LineString, MultiLineString
from shapely.geometry import Polygon, MultiPolygon
from osgeo import osr, ogr

PROJ_WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
ALBERS_N_AMERICA = ("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 "
                    "+lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 "
                    "+units=m +no_defs")


def projection_from_string(proj_str=PROJ_WGS84):
    """Returns a projection from a Proj4 string.
    If no argument is provided, uses WGS84/EPSG:4326"""
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj_str)
    return srs


def projection_from_epsg(epsg=4326):
    """Returns a projection from an epsg integer.
    If no argument is provided, uses 4326."""
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs


def projection_from_wkt(wkt=osr.SRS_WKT_WGS84):
    """Returns a projection from well known text.
    If no argument is provided, uses 4326 equivalent"""
    srs = osr.SpatialReference()
    srs.ImportfromWkt(wkt)
    return srs


# Create a function to project between spatial coords
def get_projection(obj, layer_index=0):
    if isinstance(obj, ogr.DataSource):
        obj = obj.GetLayerByIndex(layer_index)

    if hasattr(obj, "GetSpatialRef"):
        srs = obj.GetSpatialRef()
    else:
        proj_wkt = obj.GetProjection()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj_wkt)
    return srs


"""
The SVG related functions in this file were adapted from the shapely project
(https://github.com/Toblerity/Shapely) which have been permitted for use
under the BSD license.
"""


def _repr_svg_(shp):
    """SVG representation for iPython notebook"""
    svg_top = ('<svg xmlns="http://www.w3.org/2000/svg" '
               'xmlns:xlink="http://www.w3.org/1999/xlink" ')
    if shp.is_empty:
        return svg_top + '/>'
    else:
        # Establish SVG canvas that will fit all the data + small space
        xmin, ymin, xmax, ymax = shp.bounds
        if xmin == xmax and ymin == ymax:
            # This is a point; buffer using an arbitrary size
            xmin, ymin, xmax, ymax = shp.buffer(1).bounds
        else:
            # Expand bounds by a fraction of the data ranges
            expand = 0.04  # or 4%, same as R plots
            widest_part = max([xmax - xmin, ymax - ymin])
            expand_amount = widest_part * expand
            xmin -= expand_amount
            ymin -= expand_amount
            xmax += expand_amount
            ymax += expand_amount
        dx = xmax - xmin
        dy = ymax - ymin
        width = min([max([100., dx]), 300])
        height = min([max([100., dy]), 300])
        try:
            scale_factor = max([dx, dy]) / max([width, height])
        except ZeroDivisionError:
            scale_factor = 1.
        view_box = "{0} {1} {2} {3}".format(xmin, ymin, dx, dy)
        transform = "matrix(1,0,0,-1,0,{0})".format(ymax + ymin)

        svg_templ = svg_top + (
            'width="{1}" height="{2}" viewBox="{0}" '
            'preserveAspectRatio="# XXX: MinYMin meet">'
            '<g transform="{3}">'
        ).format(view_box, width, height, transform)
        return scale_factor, svg_templ + "{0}</g></svg>"


def svg_polygon(shp, scale_factor, fill):
    """Returns SVG path element for the Polygon geometry.
    Parameters
    ==========
    scale_factor : float
        Multiplication factor for the SVG stroke-width.  Default is 1.
    fill_color : str, optional
        Hex string for fill color. Default is to use "#66cc99" if
        geometry is valid, and "#ff3333" if invalid.
    """
    if shp.is_empty:
        return '<g />'

    fill = fill if shp.is_valid else "#ff3333"
    stroke = fill
    exterior_coords = [
        ["{0:.4f},{1:.4f}".format(*c) for c in shp.exterior.coords]]
    interior_coords = [
        ["{0:.4f},{1:.4f}".format(*c) for c in interior.coords]
        for interior in shp.interiors]

    path = " ".join([
        "M {0} L {1} z".format(coords[0], " L ".join(coords[1:]))
        for coords in exterior_coords + interior_coords])
    return (
        '<path fill-rule="evenodd" fill="{2}" stroke="{3}" '
        'stroke-width="{0}" opacity="0.8" d="{1}" />'
        ).format(2. * scale_factor, path, fill, stroke)


def svg_multipolygon(shp, scale_factor, fill):
    """Returns group of SVG path elements for the MultiPolygon geometry.
    Parameters
    ==========
    scale_factor : float
        Multiplication factor for the SVG stroke-width.  Default is 1.
    fill_color : str, optional
        Hex string for fill color. Default is to use "#66cc99" if
        geometry is valid, and "#ff3333" if invalid.
    """
    if shp.is_empty:
        return '<g />'

    fill = fill if shp.is_valid else "#ff3333"
    return '<g>' + \
        ''.join(svg_polygon(p, scale_factor, fill) for p in shp) + \
        '</g>'


def svg_line(shp, scale_factor, stroke):
    """Returns SVG polyline element for the LineString geometry.
    Parameters
    ==========
    scale_factor : float
        Multiplication factor for the SVG stroke-width.  Default is 1.
    stroke_color : str, optional
        Hex string for stroke color. Default is to use "#66cc99" if
        geometry is valid, and "#ff3333" if invalid.
    """
    if shp.is_empty:
        return '<g />'

    stroke = stroke if shp.is_valid else "#ff3333"
    pnt_format = " ".join(["%0.4f,%0.4f" % c for c in shp.coords])
    return ('<polyline fill="none" stroke="{2}" stroke-width="{1}" '
            'points="{0}" opacity="0.8" />') \
        .format(pnt_format, 2. * scale_factor, stroke)


def svg_multiline(shp, scale_factor, stroke):
    """Returns a group of SVG polyline elements for the LineString geometry.
    Parameters
    ==========
    scale_factor : float
        Multiplication factor for the SVG stroke-width.  Default is 1.
    stroke_color : str, optional
        Hex string for stroke color. Default is to use "#66cc99" if
        geometry is valid, and "#ff3333" if invalid.
    """
    if shp.is_empty:
        return '<g />'

    stroke = stroke if shp.is_valid else "#ff3333"
    return '<g>' + \
        ''.join(svg_line(p, scale_factor, stroke) for p in shp) + \
        '</g>'


def to_svg(shp, color="#1f78b4"):
    """Function that takes a shapely object and returns
    an svg of that object.  Currently only supports (Multi)LineString,
    and (Multi)Polygon."""
    scale_factor, svg_str = _repr_svg_(shp)

    svg_fns = {LineString: svg_line,
               MultiLineString: svg_multiline,
               Polygon: svg_polygon,
               MultiPolygon: svg_multipolygon}

    fn = svg_fns.get(type(shp), None)
    if fn is None:
        return np.NaN

    svg = fn(shp, scale_factor, color)
    return svg_str.format(svg)
