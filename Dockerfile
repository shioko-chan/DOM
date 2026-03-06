FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive

ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}

RUN echo 'Acquire::Languages "none";' > /etc/apt/apt.conf.d/99nolanguage

WORKDIR /opt

RUN apt-get update \
    && apt-get install -y --no-install-recommends gnupg wget ca-certificates

RUN wget -O - "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBC5934FD3DEBD4DAEA544F791E2824A7F22B44BD" | gpg --dearmor -o /etc/apt/keyrings/apt-fast.gpg \
    && echo 'deb [signed-by=/etc/apt/keyrings/apt-fast.gpg] http://ppa.launchpad.net/apt-fast/stable/ubuntu noble main' | tee /etc/apt/sources.list.d/apt-fast.list >/dev/null \
    && apt-get update && apt-get install -y --no-install-recommends apt-fast

RUN apt-fast install -y --no-install-recommends \
    tar build-essential pkg-config ninja-build \
    libtbb-dev \
    libgtest-dev \
    libopencv-dev \
    libgoogle-glog-dev libgflags-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libmetis-dev \
    libgeographiclib-dev \
    geographiclib-tools \
    libpcl-dev

RUN wget -O - "https://apt.kitware.com/keys/kitware-archive-latest.asc" | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
    && apt-fast update && apt-fast install -y cmake

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget http://ceres-solver.org/ceres-solver-2.2.0.tar.gz \
    && tar -xf ceres-solver-2.2.0.tar.gz \
    && mkdir build \
    && cmake -G Ninja -B build ./ceres-solver-2.2.0 \
    && cmake --build build \
    && cmake --install build \
    && rm -rf ceres-solver-2.2.0.tar.gz ceres-solver-2.2.0 build

RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && wget https://github.com/Exiv2/exiv2/releases/download/v0.28.5/exiv2-0.28.5-Linux-x86_64.tar.gz \
    && tar -xf onnxruntime-linux-x64-gpu-1.21.0.tgz \
    && tar -xf exiv2-0.28.5-Linux-x86_64.tar.gz \
    && mkdir -p /usr/local/onnxruntime /usr/local/exiv2 \
    && cp -r onnxruntime-linux-x64-gpu-1.21.0/include /usr/local/onnxruntime/include \
    && cp -r onnxruntime-linux-x64-gpu-1.21.0/lib /usr/local/onnxruntime/lib \
    && cp -r exiv2-0.28.5-Linux-x86_64/include /usr/local/exiv2/include \
    && cp -r exiv2-0.28.5-Linux-x86_64/lib /usr/local/exiv2/lib \
    && ln -s /usr/local/onnxruntime/lib /usr/local/onnxruntime/lib64 \
    && rm -r onnxruntime-linux-x64-gpu-1.21.0 onnxruntime-linux-x64-gpu-1.21.0.tgz exiv2-0.28.5-Linux-x86_64 exiv2-0.28.5-Linux-x86_64.tar.gz

COPY . /opt/DOM

RUN mkdir build \
    && cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release ./DOM \
    && cmake --build build \
    && cmake --install build \
    && rm -rf build

RUN echo "/usr/local/exiv2/lib" > /etc/ld.so.conf.d/exiv2.conf && \
    echo "/usr/local/onnxruntime/lib" > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig

ENTRYPOINT [ "/bin/bash" ]
