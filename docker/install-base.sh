#!/bin/bash
set -e

add-apt-repository -y ppa:ubuntugis/ppa
apt-get update
apt-get install -y wget htop curl vim jq \
                   python-dev python-pip \
                   gcc gfortran \
                   liblapack-dev libatlas-dev \
                   libatlas-base-dev \
                   systemd

apt-get install -y gdal-bin libgdal-dev \
                   libspatialindex-dev
