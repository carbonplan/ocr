FROM daskdev/dask:latest

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

# s5cmd
WORKDIR /tmp
RUN curl -L https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz -o /tmp/s5cmd.tar.gz \
    && tar -xvzf /tmp/s5cmd.tar.gz -C /usr/local/bin/ \
    && rm /tmp/s5cmd.tar.gz

    
USER 1000
WORKDIR /home/dask