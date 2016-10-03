try:
    from urlparse import urlparse
except ImportError:
    # Python3
    from urllib.parse import urlparse
