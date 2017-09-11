#!/bin/bash
set -e
set -x

export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal

pip install --upgrade pip
pip install numpy==1.13.1
pip install scipy==0.19.1
GDAL_VERSION=`gdal-config --version`
pip install "GDAL==$GDAL_VERSION"

pushd /tmp/
wget https://files.pythonhosted.org/packages/27/c8/279807519a8c76115c5bb38a454784b69513c889e50afee874470f884223/pyspatial-0.2.4.tar.gz
tar -xvzf pyspatial-0.2.4.tar.gz
cat pyspatial-0.2.4/requirements.txt | grep -v numpy | grep -v scipy | grep -v GDAL > requirements.txt
pip install -r requirements.txt
pip install pyspatial-0.2.4/
popd

pip install jupyter[notebook] ipykernel
mkdir -p /etc/jupyter/certs
mkdir -p /usr/local/share/jupyter
mkdir /usr/lib/systemd/system
mkdir /notebooks
