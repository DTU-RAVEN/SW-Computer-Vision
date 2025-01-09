FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    locales \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
RUN sed -i -e 's/# en_DK.UTF-8 UTF-8/en_DK.UTF-8 UTF-8/' /etc/locale.gen && locale-gen
ENV LC_ALL en_DK.UTF-8
ENV LANG en_DK.UTF-8  
ENV LANGUAGE en_DK:en   
ENV TZ=Europe/Copenhagen
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    apt-utils \
    astyle \
    build-essential \
    clang-format \
    clang-tidy \
    clangd \
    cmake \
    cppcheck \
    cpputest \
    doxygen \
    gcc-arm-none-eabi \
    gcovr \
    gdb \
    gdb-multiarch \
    git \
    graphviz \
    lcov \
    mercurial \
    plantuml \
    python-is-python3 \
    python3 \
    python3-pip \
    python-is-python3 \
    sudo \
    nano \
    npm \
    libgbm-dev \
    wget \
    git-lfs \
    cpputest \
    && apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN git lfs install \
    && git config --global --add safe.directory "*"

RUN ln -s /usr/bin/gdb-multiarch /usr/bin/arm-none-eabi-gdb \
    && echo "set auto-load safe-path /" >> /root/.gdbinit
 
RUN python3 -m pip install mbed-cli \
    && python3 -m pip install -U mbed-cli \
    && python3 -m pip install compiledb \
    && python3 -m pip install pytest-runner

RUN cd /tmp \
  && wget --tries=5 --timeout=3 https://github.com/cli/cli/releases/download/v2.11.3/gh_2.11.3_linux_amd64.deb \
  && dpkg -i gh_2.11.3_linux_amd64.deb \
  && rm -rf /tmp/*

RUN groupadd -r -g 1000 raven \
    && useradd -rm -d /home/raven -s /bin/bash --uid 1000 --gid 1000 --groups sudo raven \
    && echo raven:raven | chpasswd

USER raven
WORKDIR /home/raven