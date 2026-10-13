# Known build baseline: CUDA 12.6.3, Ubuntu 24.04, GCC 13.2.
FROM nvidia/cuda@sha256:392c0df7b577ecae17a17f6ba7f2009c217bb4422f8431c053ae9af61a8c148a
RUN test "$(g++-13 -dumpfullversion)" = "13.2.0" \
 && apt-get update -qq --snapshot 20261003T000000Z \
 && apt-get install -y --no-install-recommends --snapshot 20261003T000000Z \
    cmake=3.28.3-1build7 cmake-data=3.28.3-1build7 ninja-build=1.11.1-2 \
    libexpat1=2.6.1-2ubuntu0.6 libkrb5support0=1.20.1-6ubuntu2.10 \
    libk5crypto3=1.20.1-6ubuntu2.10 libkeyutils1=1.6.3-3build1 \
    libkrb5-3=1.20.1-6ubuntu2.10 libgssapi-krb5-2=1.20.1-6ubuntu2.10 \
    libicu74=74.2-1ubuntu3.1 libxml2=2.9.14+dfsg-1.3ubuntu3.9 \
    libnghttp2-14=1.59.0-1ubuntu0.4 libpsl5t64=0.21.2-1.1build1 \
    libuv1t64=1.48.0-1.1build1 libarchive13t64=3.7.2-2ubuntu0.8 \
    libbrotli1=1.1.0-2build2 librtmp1=2.4+20151223.gitfa8646d.1-2build7 \
    libssh-4=0.10.6-2ubuntu0.5 libcurl4t64=8.5.0-2ubuntu10.15 \
    libjsoncpp25=1.9.5-6build1 librhash0=1.4.3-3build1 \
 && rm -rf /var/lib/apt/lists/*
ENV CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
WORKDIR /workspace
ENTRYPOINT []
CMD ["bash"]
