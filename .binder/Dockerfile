FROM ghcr.io/osgeo/gdal:ubuntu-full-latest


# This is meant to use repo2docker to build and publish a
# Docker image that can be used for our PMTiles generation
# It is published with the help of the repo2docker github action

USER root

RUN apt-get update \
&& apt-get -y install \
    make \
    gcc \
    g++ \
    libsqlite3-dev \
    zlib1g-dev \
    curl \
    git \
    wget \
    ca-certificates \
    software-properties-common \
    libproj-dev \
    proj-bin \
    libgeos-dev \
    cmake \
    golang \
&& rm -rf /var/lib/apt/lists/*

# install tippecanoe
WORKDIR /tmp
RUN git clone https://github.com/felt/tippecanoe.git /tmp/tippecanoe-src
WORKDIR /tmp/tippecanoe-src
RUN make

# move tippecanoe
WORKDIR /tmp/tippecanoe-src
RUN cp tippecanoe /usr/local/bin/
RUN cp tile-join /usr/local/bin/

# Download and install the pre-built s5cmd binary
WORKDIR /tmp
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz -o /tmp/s5cmd.tar.gz \
    && tar -xvzf /tmp/s5cmd.tar.gz -C /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz

# install gpq
RUN curl -L https://github.com/planetlabs/gpq/releases/download/v0.24.0/gpq-linux-amd64.tar.gz -o /tmp/gpq.tar.gz \
    && tar -xvzf /tmp/gpq.tar.gz -C /usr/local/bin/ \
    && rm /tmp/gpq.tar.gz


WORKDIR /app
