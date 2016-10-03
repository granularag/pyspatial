from smart_open import smart_open, ParseUri

from pyspatial.py3 import urlparse
import boto
from boto import connect_s3
import os
import six


def parse_uri(uri):
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme in ['file', 's3', '']:
        return ParseUri(uri)
    elif parsed_uri.scheme in ['gs']:
        tmp_uri = ParseUri(uri.replace('gs://', 's3://'))
        tmp_uri.scheme = 'gs'
        return tmp_uri
    else:
        raise NotImplementedError("unknown URI scheme %r in %r" % (parsed_uri.scheme, uri))


def get_path(path):

    prefix = ""
    if path.endswith(".gz") and ".tar.gz" not in path:
        prefix = "/vsigzip/"

    uri = parse_uri(path)

    if uri.scheme == "file":
        path = uri.uri_path if os.path.exists(uri.uri_path) else None

    elif uri.scheme == "s3":
        conn = connect_s3()
        bucket = conn.get_bucket(uri.bucket_id)
        key = bucket.lookup(uri.key_id)
        if prefix == "":
            prefix = "/"
        prefix += os.path.join(prefix, "vsicurl")
        path = key.generate_url(60*60) if key is not None else key

    elif uri.scheme == 'gs':
        storage_uri = boto.storage_uri(uri.bucket_id, uri.scheme)
        bucket = storage_uri.get_bucket(uri.bucket_id)
        key = bucket.lookup(uri.key_id)
        if prefix == "":
            prefix = "/"
        prefix += os.path.join(prefix, "vsicurl/")
        path = key.generate_url(60*60) if key is not None else key

    return prefix+path


def open(path, mode="rb", **kw):
    uri = urlparse(path)

    if uri.scheme in ['file', 's3', '']:
        return smart_open(path, mode=mode, **kw)
    elif uri.scheme in ['gs']:
        if mode in ('r', 'rb'):
            storage_uri = boto.storage_uri(uri.netloc, uri.scheme)
            bucket = storage_uri.get_bucket(uri.netloc)
            key = bucket.get_key(uri.path)
            if key is None:
                raise KeyError(uri.path)
            return GSOpenRead(key, **kw)
        elif mode in ('w', 'wb'):
            storage_uri = boto.storage_uri(uri.netloc + '/' + uri.path, uri.scheme)
            key = storage_uri.new_key()
            if key is None:
                raise KeyError(uri.path)
            return GSOpenWrite(key, **kw)
        else:
            raise NotImplementedError("file mode %s not supported for %r scheme", mode, uri.scheme)
    else:
        raise NotImplementedError("scheme %r is not supported", uri.scheme)


class GSOpenRead(object):
    """
    Implement streamed reader from GS.
    """
    def __init__(self, read_key):
        if not hasattr(read_key, "bucket") and not hasattr(read_key, "name") and not hasattr(read_key, "read") \
                and not hasattr(read_key, "close"):
            raise TypeError("can only process GS keys")
        self.read_key = read_key

    def read(self, size=None):
        """
        Read a specified number of bytes from the key.
        """
        if not size or size < 0:
            # For compatibility with standard Python, `read(negative)` = read the rest of the file.
            # Otherwise, boto would read *from the start* if given size=-1.
            size = 0
        return self.read_key.read(size)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.read_key.close()

    def __str__(self):
        return "%s<key: %s>" % (
            self.__class__.__name__, self.read_key
        )

class GSOpenWrite(object):
    """
    Context manager for writing into GS files.
    """
    def __init__(self, outkey):
        if not hasattr(outkey, "bucket") and not hasattr(outkey, "name"):
            raise TypeError("can only process GS keys")

        self.outkey = outkey

    def __str__(self):
        return "%s<key: %s>" % (
            self.__class__.__name__, self.outkey
            )

    def write(self, b):
        """
        Write the given bytes (binary string) into the GS file from constructor.
        """
        if isinstance(b, six.text_type):
            # not part of API: also accept unicode => encode it as utf8
            b = b.encode('utf8')

        if not isinstance(b, six.binary_type):
            raise TypeError("input must be a binary string")

        self.outkey.set_contents_from_string(b)

    def close(self):
        self.outkey.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
