FROM pangeo/base-notebook:latest


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
        libgdal-dev \
        gdal-bin \
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

# Download and install the pre-built s5cmd binary for ARM
WORKDIR /tmp
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-arm64.tar.gz -o /tmp/s5cmd.tar.gz \
    && tar -xvzf /tmp/s5cmd.tar.gz -C /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz

# try to override entrypoint?
ENTRYPOINT ["/srv/start"]
CMD ["python"]

USER $NB_UID
WORKDIR $HOME
