FROM ubuntu:20.04

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
    curl  \
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

# Install ROS2
RUN apt-get update -y
RUN apt-get install software-properties-common -y
RUN apt-get update -y
RUN add-apt-repository universe
RUN apt-get update -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install ros-foxy-ros-base python3-argcomplete -y
RUN apt-get install ros-dev-tools -y

RUN groupadd -r -g 1001 raven \
    && useradd -rm -d /home/raven -s /bin/bash --uid 1001 --gid 1001 --groups sudo raven \
    && echo raven:raven | chpasswd

USER raven
WORKDIR /home/raven


