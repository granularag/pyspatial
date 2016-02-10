#!/bin/bash
apt-get update
apt-get -y install \
    curl \
    libcurl4-openssl-dev \
    libcurl3-nss \
    libsasl2-dev \
    python-dev \
    python-pip \
    libblas-dev \
    liblapack-dev \
    libfreetype6-dev \
    libreadline-dev \
    libgeos++-dev \
    libspatialindex-dev \
    gfortran
