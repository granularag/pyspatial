For sample scripts, see scripts/requirements-*.sh

# OSX

*  Scripts provided are based off of homebrew (copy and paste into terminal).  Should not install if you already have MacPorts.
*  No support for MacPorts is provided.
*  To install Homebrew:
  * $ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

#GDAL

1. If you don't have root access, you should download the source and build packages like
  * $ ./configure --prefix ~/local
* To make the binaries available add the following to your bashrc
  * export HOME_DIR=/my/home/dir
  * export PATH=$PATH:$HOME_DIR/local/bin
* To build gdal (assume geos installed in /usr/local), in non-standard localtion:
  * $ export HOME_DIR=/my/home/dir
  * $ ./configure --enable-64bit --prefix ~/local --with-includes=$HOME_DIR/local/include/ --with-libs=$HOME_DIR/local/lib/ --with-sqlite=no --with-geos=/usr/local/bin/geos-config --with-opengl=no --with-cairo=no --with-freetype=no --with-lapack --with-blas --with-readline
* In your scripts/bashrc:
  * export HOME_DIR=/my/home/dir
  * export GDALHOME=$HOME_DIR/local/
  * export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_HOME/local/lib/

# Python
* go into the root dir containing the setup.py file:
  * $ pip install .
