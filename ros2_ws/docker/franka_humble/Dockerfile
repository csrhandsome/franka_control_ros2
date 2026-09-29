ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO=humble
ARG USERNAME=franka
ARG USER_UID=1000
ARG USER_GID=1000
# v2.5.1's dependency.repos pins libfranka 0.20.4 and franka_description 2.8.1.
ARG FRANKA_ROS2_REF=v2.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=${ROS_DISTRO} \
    RCUTILS_COLORIZED_OUTPUT=1 \
    COLCON_HOME=/home/${USERNAME}/.colcon

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    cmake \
    git \
    iproute2 \
    iputils-ping \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-${ROS_DISTRO}-controller-manager \
    ros-${ROS_DISTRO}-joint-state-broadcaster \
    ros-${ROS_DISTRO}-joint-trajectory-controller \
    ros-${ROS_DISTRO}-moveit \
    ros-${ROS_DISTRO}-ros2controlcli \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-xacro \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && rosdep init 2>/dev/null || true

COPY entrypoint.sh /usr/local/bin/franka-entrypoint
COPY scripts/fake_hardware_smoke_test.sh /usr/local/bin/fake-hardware-smoke-test
COPY scripts/verify_fci_network.sh /usr/local/bin/verify-fci-network
RUN chmod 0755 /usr/local/bin/franka-entrypoint \
    /usr/local/bin/fake-hardware-smoke-test \
    /usr/local/bin/verify-fci-network

USER ${USERNAME}
WORKDIR /opt/franka_ros2_ws

RUN git clone --branch ${FRANKA_ROS2_REF} --depth 1 \
    https://github.com/frankarobotics/franka_ros2.git src/franka_ros2 \
    && vcs import src < src/franka_ros2/dependency.repos --recursive --skip-existing \
    && rosdep update \
    && sudo apt-get update \
    && rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y \
       --skip-keys="franka_selfcollision gz_sim_vendor parallel_gripper_controller gz_plugin_vendor" \
    && sudo rm -rf /var/lib/apt/lists/* \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && colcon build --symlink-install --cmake-args \
       -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF

ENTRYPOINT ["/usr/local/bin/franka-entrypoint"]
CMD ["bash"]
