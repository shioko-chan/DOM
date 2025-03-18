FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

# Install basic packages
RUN apt-get update && apt-get install -y \
    ca-certificates apt-transport-https software-properties-common \
    build-essential gnupg pkg-config ninja-build \
    wget curl zip unzip tar git \
    libpng-dev libjpeg-dev libtiff-dev libxxf86vm1 libxxf86vm-dev \
    libxi-dev libxrandr-dev \
    libopencv-dev libceres-dev libboost-all-dev \
    libatlas-base-dev libglfw3-dev libmpfr-dev libgmp-dev  \
    && rm -rf /var/lib/apt/lists/*

# Install CMake
WORKDIR /opt
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && tar -xvf onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && cp -r onnxruntime-linux-x64-gpu-1.21.0/* /usr/local/ \
    && wget https://github.com/Exiv2/exiv2/releases/download/v0.28.5/exiv2-0.28.5-Linux-x86_64.tar.gz \
    && tar -xvf exiv2-0.28.5-Linux-x86_64.tar.gz \
    && cp -r exiv2-0.28.5-Linux-x86_64/* /usr/local/ \
    && rm -r onnxruntime onnxruntime-linux-x64-gpu-1.21.0.tgz exiv2-0.28.5-Linux-x86_64 exiv2-0.28.5-Linux-x86_64.tar.gz

# Install Dependi CGAL
WORKDIR /opt
RUN wget https://github.com/CGAL/cgal/releases/download/v6.0.1/CGAL-6.0.1.tar.xz \
    && tar -xvf CGAL-6.0.1.tar.xz \
    && cd CGAL-6.0.1 \
    && cmake . && make install \
    && rm -r /opt/CGAL-6.0.1/ /opt/CGAL-6.0.1.tar.xz

WORKDIR /opt
ENTRYPOINT [ "/bin/bash" ]