# ------------------------------------------------------------
# Base image
# ------------------------------------------------------------
FROM ubuntu:20.04

ENV TZ=Europe/Copenhagen \
    LC_ALL=en_DK.UTF-8 \
    LANG=en_DK.UTF-8 \
    LANGUAGE=en_DK:en \
    DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------------
# Locale
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y locales && \
    sed -i 's/# en_DK.UTF-8 UTF-8/en_DK.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

# ------------------------------------------------------------
# System & development tooling (incl. Python 3.8)
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # general utilities
        apt-utils software-properties-common sudo nano curl wget git git-lfs \
        # compilers / build‑chain
        build-essential gcc-arm-none-eabi clang-format clang-tidy clangd cmake \
        gdb gdb-multiarch cppcheck cpputest lcov gcovr doxygen plantuml graphviz \
        astyle \
        # explicit Python 3.8 + tooling
        python3.8 python3.8-venv python3.8-dev python3-pip python-is-python3 \
        # node (for PlantUML / docs)
        npm \
        # graphics / ML deps
        libgbm-dev \
        # -------- PyGObject / pycairo native build deps ----------
        pkg-config libcairo2-dev libgirepository1.0-dev libglib2.0-dev \
        # ---------------------------------------------------------
        gnupg2 lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Git‑LFS needs an init step
RUN git lfs install && git config --global --add safe.directory '*'

# Convenient alias so arm‑none‑eabi‑gdb exists
RUN ln -s /usr/bin/gdb-multiarch /usr/bin/arm-none-eabi-gdb && \
    echo 'set auto-load safe-path /' >> /root/.gdbinit

# ------------------------------------------------------------
# ROS 2 Foxy base install
# ------------------------------------------------------------
RUN add-apt-repository universe && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        ros-foxy-ros-base \
        ros-foxy-vision-msgs \
        ros-dev-tools \
        python3-argcomplete && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir -p /workspaces/SW-Computer-Vision && chown -R 1000:1000 /workspaces

# --- GStreamer runtime + Python bindings -------------------
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3-gi python3-gst-1.0 \
        gir1.2-gstreamer-1.0 gir1.2-gst-plugins-base-1.0 \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav && \
    rm -rf /var/lib/apt/lists/*


# ------------------------------------------------------------
# Non‑root user
# ------------------------------------------------------------
RUN groupadd -r -g 1000 raven && \
    useradd  -m -r -u 1000 -g raven -s /bin/bash -G sudo raven && \
    echo 'raven:raven' | chpasswd

USER raven
WORKDIR /home/raven

# ------------------------------------------------------------
# Python 3.8 virtual‑env + requirements
# ------------------------------------------------------------
COPY requirements.txt /home/raven/

COPY --chown=raven:raven requirements.txt /home/raven/

USER raven
WORKDIR /home/raven

RUN python3.8 -m venv /workspaces/SW-Computer-Vision/venv \
 && /workspaces/SW-Computer-Vision/venv/bin/pip install --upgrade pip \
 && /workspaces/SW-Computer-Vision/venv/bin/pip install -r /home/raven/requirements.txt


ENV PATH="/home/raven/venv/bin:${PATH}"

# ------------------------------------------------------------
# Default shell
# ------------------------------------------------------------
CMD ["/bin/bash"]
