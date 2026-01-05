# ============================================================================
# Docker Image: Ubuntu with LXDE Desktop, and VNC Access
# ============================================================================
FROM ghcr.io/astral-sh/uv:0.9.20 AS uv-builder

# ============================================================================
# Stage 2: Conda Builder for Python 3.8
# ============================================================================
FROM continuumio/miniconda3:main AS conda-builder

# Create a Python 3.8 environment
RUN conda create -n py38 python=3.8 -y && \
    conda clean -afy

# ============================================================================
# Stage 3: Main Runtime Image - CUDA 11.8 Devel on Ubuntu 22.04
# ============================================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

LABEL maintainer="Saswata Sarkar <sarkarsaswata01@gmail.com>"
LABEL description="CUDA 11.8 Development Environment with COLMAP, ImageMagick, Python, and Desktop"

# ============================================================================
# CUDA and Build Configuration
# ============================================================================
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/root/.local/bin:${CUDA_HOME}/bin:/usr/local/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MAX_JOBS=4 \
    TORCH_CUDA_ARCH_LIST="8.6" \
    TCNN_CUDA_ARCHITECTURES="86" \
    TCNN_HALF_PRECISION=1

# ============================================================================
# Desktop Environment & Python Configuration
# ============================================================================
ENV DISPLAY=:0 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_VENV_CLEAR=1 \
    UV_HTTP_TIMEOUT=300 \
    PYTHONWARNINGS=ignore

# ============================================================================
# APT Optimization: Use Ubuntu mirror network
# ============================================================================
RUN sed -i 's#http://archive.ubuntu.com/ubuntu/#mirror://mirrors.ubuntu.com/mirrors.txt#' /etc/apt/sources.list

# ============================================================================
# System Dependencies Installation
# ============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools and development
    build-essential cmake pkg-config git curl wget ninja-build \
    gcc g++ make ca-certificates \
    # Python and pip
    python3-pip \
    # COLMAP and its dependencies
    colmap \
    libboost-program-options1.74.0 libboost-filesystem1.74.0 libboost-graph1.74.0 \
    libboost-system1.74.0 libeigen3-dev libsuitesparse-dev libmetis5 \
    libfreeimage3 libflann1.9 libsqlite3-0 \
    libgoogle-glog0v5 libceres2 \
    libglew2.2 libqt5core5a libqt5gui5 libqt5widgets5 libqt5opengl5 \
    libcgal-dev libopenblas0 liblapack3 libblas3 \
    # ImageMagick and dependencies
    imagemagick imagemagick-6.q16 \
    libltdl7 libbz2-1.0 libwebp7 libxml2 libpng16-16 libjpeg-turbo8 libtiff5 \
    libopenjp2-7 libomp-dev \
    # CUDA-specific tools and libraries
    libcublas-11-8 libcublas-dev-11-8 \
    libcusolver-11-8 libcusolver-dev-11-8 \
    libcusparse-11-8 libcusparse-dev-11-8 \
    # Graphics and display
    libegl1-mesa libgl1 libgles2-mesa \
    # Desktop environment (GUI/VNC)
    dbus-x11 x11-utils xvfb x11vnc lxde supervisor nginx tini \
    # Graphics libraries
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    # Desktop themes and icons
    arc-theme adwaita-icon-theme papirus-icon-theme ubuntu-wallpapers \
    # Utilities
    tmux net-tools procps sudo htop vim nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/*

# ============================================================================
# Copy Conda with Python 3.8 environment from builder stage
# ============================================================================
COPY --from=conda-builder /opt/conda /opt/conda

# ============================================================================
# Initialize conda but don't activate by default (keep system Python as default)
# ============================================================================
RUN /opt/conda/bin/conda init bash && \
    /opt/conda/bin/python -m pip install --upgrade pip setuptools wheel

# ============================================================================
# Copy UV from builder stage
# ============================================================================
COPY --from=uv-builder /uv /uvx /bin/

# ============================================================================
# Verify COLMAP and ImageMagick installation
# ============================================================================
RUN which colmap && colmap help | head -n 1 && \
    which convert && convert -version | head -n 1

# ============================================================================
# Configuration Setup
# ============================================================================
RUN mkdir -p /etc/xdg/autostart /root/.config/lxsession/LXDE /workspace /var/log/supervisor && \
    echo "[Desktop Entry]\nHidden=true" > /etc/xdg/autostart/light-locker.desktop && \
    echo "[Session]\nlock_manager/command=" > /root/.config/lxsession/LXDE/desktop.conf

# ============================================================================
# noVNC, Websockify & Brave Browser
# ============================================================================
RUN mkdir -p /opt/novnc && \
    curl -fsSL https://github.com/novnc/noVNC/archive/refs/tags/v1.6.0.tar.gz | \
    tar xz --strip 1 -C /opt/novnc && \
    ln -s /opt/novnc/vnc.html /opt/novnc/index.html && \
    pip3 install --no-cache-dir websockify setuptools wheel && \
    curl -fsSL https://dl.brave.com/install.sh | sh && \
    rm -rf /tmp/* /var/tmp/*

# ============================================================================
# Nginx Configuration (Reverse Proxy for noVNC)
# ============================================================================
RUN printf "server {\n  listen 80;\n  location / {\n    proxy_pass http://127.0.0.1:6080/;\n    proxy_http_version 1.1;\n    proxy_set_header Upgrade \$http_upgrade;\n    proxy_set_header Connection \"Upgrade\";\n  }\n}\n" \
    > /etc/nginx/sites-available/default

WORKDIR /workspace

# ============================================================================
# Copy Configuration & Startup Scripts
# ============================================================================
COPY supervisord.conf /etc/supervisor/supervisord.conf
COPY startup.sh /startup.sh
COPY entrypoint-user.sh /usr/local/bin/entrypoint-user.sh
RUN chmod +x /startup.sh /usr/local/bin/entrypoint-user.sh

# ============================================================================
# Expose Service Ports
# ============================================================================
EXPOSE 80 5900 6080

# ============================================================================
# Entrypoint
# ============================================================================
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/entrypoint-user.sh"]
CMD ["/startup.sh"]
