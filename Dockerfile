FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

RUN apt-get update && apt-get install -y \
    ca-certificates apt-transport-https software-properties-common \
    build-essential gnupg pkg-config ninja-build \
    wget curl zip unzip tar git \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-get update && apt-get install -y cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && tar -xvf onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && mkdir -p /usr/local/onnxruntime \
    && cp -r onnxruntime-linux-x64-gpu-1.21.0/include /usr/local/onnxruntime/include \
    && cp -r onnxruntime-linux-x64-gpu-1.21.0/lib /usr/local/onnxruntime/lib \
    && ln -s /usr/local/onnxruntime/lib /usr/local/onnxruntime/lib64 \
    && wget https://github.com/Exiv2/exiv2/releases/download/v0.28.5/exiv2-0.28.5-Linux-x86_64.tar.gz \
    && tar -xvf exiv2-0.28.5-Linux-x86_64.tar.gz \
    && mkdir /usr/local/exiv2 -p \
    && cp -r exiv2-0.28.5-Linux-x86_64/include /usr/local/exiv2/include \
    && cp -r exiv2-0.28.5-Linux-x86_64/lib /usr/local/exiv2/lib \
    && rm -r onnxruntime-linux-x64-gpu-1.21.0 onnxruntime-linux-x64-gpu-1.21.0.tgz exiv2-0.28.5-Linux-x86_64 exiv2-0.28.5-Linux-x86_64.tar.gz

COPY . /opt/DOM
RUN cd /opt/DOM \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j$(nproc) \
    && make install

WORKDIR /opt
ENTRYPOINT [ "/bin/bash" ]