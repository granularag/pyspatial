For sample scripts, see scripts/requirements-*.sh

# OSX

*  Scripts provided are based off of homebrew (copy and paste into terminal).  Should not install if you already have MacPorts.
*  No support for MacPorts is provided.

```bash
# Install Homebrew
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

brew install geos
brew install spatialindex

# Install latest GDAL (1.11.2)
brew install gdal

## Alternatively, install GDAL 2.0 (Still in testing)
# brew install
# https://gist.githubusercontent.com/brianreavis/261cf46b44669366df9c/raw/aa7f2f2a8a511975d7d1dae9e5acf5ac203ba969/gdal.rb
```

# Python

```bash
git clone https://github.com/granularag/pyspatial.git
cd pyspatial
pip install -r requirements.txt
pip install .
```

# Ubuntu 14.04
```bash
# install the system dependencies
sudo apt-get update
sudo apt-get install libgdal-dev
sudo apt-get install libspatialindex-dev
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

# create python virtual environment
virtualenv venv
source venv/bin/activate

# update GDAL version in requirements.txt to match system version 
# (GDAL 2.0.2 doesn’t seem to be available in any PPA)
# GDAL==1.11.2

# install python dependencies
pip install -r requirements-dev.txt
pip install -r requirements.txt
pip install -e .

# run the tests
nosetests -v
```

# Appendix

## Compiling GDAL from source

* A good overview is provided here: https://docs.djangoproject.com/en/1.9/ref/contrib/gis/install/geolibs/
* More detailed information can be found here: https://trac.osgeo.org/gdal/wiki/BuildHints


1. If you don't have root access, you should download the source and build packages like
  * $ ./configure --prefix ~/local
* To make the binaries available add the following to your bashrc

```bash
export HOME_DIR=/my/home/dir
export PATH=$PATH:$HOME_DIR/local/bin
```

* To build gdal (assume geos installed in /usr/local), in non-standard localtion:
  * $ export HOME_DIR=/my/home/dir
  * $ ./configure --enable-64bit --prefix $HOME_DIR/local --with-includes=$HOME_DIR/local/include/ --with-libs=$HOME_DIR/local/lib/ --with-sqlite=no --with-geos=/usr/local/bin/geos-config --with-opengl=no --with-cairo=no --with-freetype=no --with-lapack --with-blas --with-readline
* In your scripts/bashrc:


```bash
export HOME_DIR=/my/home/dir
export GDALHOME=$HOME_DIR/local/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$USER_HOME/local/lib/
```
