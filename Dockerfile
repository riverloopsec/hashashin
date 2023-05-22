FROM amd64/ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    unzip \
    libdbus-1-3; # for binary ninja install_api.py
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install requests

RUN mkdir /root/.binaryninja
# TODO: securely add license to /root/.binaryninja/license.dat
COPY pilot_2023_headless_license.dat /root/.binaryninja/license.dat

WORKDIR /
COPY ./binaryninja-api/BinaryNinja-headless.zip .
RUN unzip BinaryNinja-headless.zip -x 'binaryninja/docs/*' 'binaryninja/api-docs/*' 'binaryninja/scc-docs/*' && rm BinaryNinja-headless.zip
WORKDIR /binaryninja
RUN chmod +x scripts/install_api.py && python3 scripts/install_api.py

COPY . /processor
WORKDIR /processor
RUN python3 -m pip install .

# From Dockerfile.base; unsure if this is needed or what it does
#WORKDIR /root/.binaryninja/
#RUN mkdir license_mnt && ln -s license_mnt/license.dat license.dat

# From Dockerfile.base; unsure if this is needed
# NOTE: We do this after calling Binja setup, as otherwise it tries to double-install to Python3 and fails the 2nd time:
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get clean; \
 rm -rf /var/lib/apt/lists/*

WORKDIR /data
CMD ["hashashin-lib-match", "-h"]
