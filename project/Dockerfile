# We will use Ubuntu for our image
FROM ubuntu:18.04
# Updating Ubuntu packages
RUN apt-get update && yes|apt-get upgrade &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget curl \
    python3 python3-pip python3-dev python3-setuptools  \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip install -U pip
RUN pip install torchnet==0.0.4
RUN pip install catalyst==20.3
RUN pip install deep-pipe==0.0.7.post1
RUN pip install trafaret
RUN pip install trafaret_config
RUN pip install toolz==0.10.0
RUN pip install alchemy-catalyst
RUN pip install pytorch-argus

ENV PYTHONPATH $PYTHONPATH:/workdir/src:/workdir

COPY ./cnd/ocr /workdir
COPY ./CropNumbers /workdir/data/CropNumbers
#COPY ../data/Numbase


WORKDIR /workdir