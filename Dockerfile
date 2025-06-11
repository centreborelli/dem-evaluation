FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
       nvidia-cuda-toolkit \
       nvidia-cuda-toolkit-gcc \
       libgdal-dev gdal-bin \
       build-essential libfftw3-dev libgeotiff-dev libtiff5-dev \
       libgl1 git cmake make ccache imagemagick \
       libimage-exiftool-perl exiv2 proj-bin libx11-dev \
       wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python and pip
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install S2P
RUN git clone https://github.com/centreborelli/s2p-hd.git --recursive \
    && pip install -e "./s2p-hd[test]" \
    && pyproj sync -v --file us_nga_egm96_15 \
    && make -C s2p-hd test

# Install cars
RUN pip install cars

# Install ASP
ENV ASP_VERSION=3.4.0
ENV ASP_FILE=StereoPipeline-3.4.0-2024-06-19-x86_64-Linux.tar.bz2
ENV ASP_URL=https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/${ASP_VERSION}/${ASP_FILE}

RUN wget -O asp.tar.bz2 "$ASP_URL" \
    && mkdir asp && tar -xvf asp.tar.bz2 -C asp --strip-components=1 \
    && rm asp.tar.bz2 \
    && echo "export PATH=\$PATH:/workdir/asp/bin" >> /etc/profile

# Install MicMac
RUN git clone https://github.com/micmacIGN/micmac.git \
    && cd micmac \
    && mkdir build && cd build \
    && cmake .. \
    && make install -j$(nproc --all) \
    && cd ../..

# Install demcompare
RUN pip install -U setuptools setuptools_scm wheel
RUN pip install -e ./demcompare

# Set PATH for ASP and MicMac
ENV PATH="/workspace/asp/bin:/workspace/micmac/bin:$PATH"

CMD ["/bin/bash"]
