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
from __future__ import print_function
import pandas as pd
import numpy as np
from datetime import datetime, date
import re

# Python3
try:
    basestring
except NameError:
    basestring = str

TYPE_MAP = {float: "number",
            int: "number",
            basestring: "text",
            bool: "bool",
            pd.Timestamp: "datetime",
            date: "datetime",
            datetime: "datetime"}


def get_sample_pt(s):
    pt = s.dropna()
    if pt.shape[0] > 0:
        return pt.iloc[0]
    else:
        return None


def get_type(s, type_map=TYPE_MAP):
    t = type(get_sample_pt(s))

    if t in type_map:
        return type_map[t]
    elif np.dtype(t) == np.object:
        return None
    else:
        np_type = re.findall("[A-z]+", str(np.dtype(t)))[0]
        if np_type in ["int", "float"]:
            return "number"
        else:
            return np_type


def to_dict(df, hidden=None, not_visible=None, labels=None, types=None):
    """Convert a DataFrame into a Dataset dictionary.  This can later
    be serialized to json using the 'dumps' method found in this module.
    Types are either inferred or taken from the 'type' parameter.
    If the types cannot be identified, those columns will not be included.
    Note, if all the values of a column are NaN, the column will also
    be removed.

    Parameters
    -----------
    hidden: list (default=None)
        Columns in df to mark as hidden

    not_visible: list (default=None)
        Columns in df to not mark as visible

    labels: dict
        Keys are the column names and values are the 'label' attribute

    types: dict
        Keys are the column names and values are the 'type' attribute

    Returns
    -------
    Python dictionary with the attributes 'data', 'schema', 'index'
    """
    if df is None:
        df = pd.DataFrame()

    s = []
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("DataFrame columns not unique!")

    hidden = [] if hidden is None else hidden
    not_visible = [] if not_visible is None else not_visible

    cols = []

    if df.index.name is None:
        cols.append("index")
        index = "index"
        hidden.append("index")
    else:
        index = df.index.name

    df = df.reset_index()

    for k, v in df.items():
        row = {"type": get_type(v), "field": k, "label": k,
               "visible": True, "hidden": False}

        if types is not None and k in types:
            row["type"] = types[k]

        if row["type"] is None:
            msg = "Unable to determine type: %s." % k
            msg += " Please specify in types"
            print("WARN: " + msg)
            continue

        # Convert NaT to NaN
        if row["type"] == "datetime":
            df[k] = df[k].map(lambda x: np.NaN if pd.isnull(x) else x)

        if labels is not None and k in labels:
            row["label"] = labels[k]

        if k in hidden:
            row["hidden"] = True

        if k in not_visible:
            row["visible"] = False

        s.append(row)
        cols.append(k)

    return {"schema": s, "index": index,
            "data": df[cols].to_dict(orient="records")}


def dumps(x, double_precision=6):
    return pd.io.json.dumps(x, double_precision=double_precision,
                            iso_dates=True)
