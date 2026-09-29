FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    UV_PROJECT=/opt/uv/project \
    UV_PROJECT_ENVIRONMENT=/opt/uv/venv \
    VIRTUAL_ENV=/opt/uv/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON=/usr/bin/python3 \
    PATH=/opt/uv/venv/bin:/usr/local/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /usr/local/bin/uv
COPY python /opt/uv/project

WORKDIR /opt/uv/project
RUN uv venv --python /usr/bin/python3 --system-site-packages /opt/uv/venv \
    && uv sync --locked \
    && /opt/uv/venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 12)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libboost-python-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Jazzy's binary cv_bridge was built against NumPy 1; LeRobot 0.6.1 needs NumPy 2.
# Rebuild the same cv_bridge release against the NumPy headers in this venv.
RUN git clone --depth 1 --branch 4.1.0 \
    https://github.com/ros-perception/vision_opencv.git /opt/vision_opencv
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /opt/cv_bridge_ws \
    && source /opt/ros/jazzy/setup.bash \
    && cd /opt/cv_bridge_ws \
    && colcon build --merge-install --packages-select cv_bridge \
       --base-paths /opt/vision_opencv \
       --cmake-args -DCMAKE_BUILD_TYPE=Release \
       -DPython3_EXECUTABLE=/opt/uv/venv/bin/python \
       -DPython3_NumPy_INCLUDE_DIRS=/opt/uv/venv/lib/python3.12/site-packages/numpy/_core/include

COPY entrypoint.sh /usr/local/bin/franka-jazzy-entrypoint
RUN chmod 0755 /usr/local/bin/franka-jazzy-entrypoint

WORKDIR /workspace/data_collect
ENTRYPOINT ["/usr/local/bin/franka-jazzy-entrypoint"]
