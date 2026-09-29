#!/usr/bin/env bash
# Configure Docker Engine (not just the shell) to use the local HTTPS proxy.
# Run this in a terminal owned by the logged-in desktop user; sudo will prompt there.
set -euo pipefail

http_proxy="${HTTP_PROXY:-http://127.0.0.1:7897}"
https_proxy="${HTTPS_PROXY:-${http_proxy}}"
no_proxy="${NO_PROXY:-localhost,127.0.0.1,::1}"
drop_in_dir=/etc/systemd/system/docker.service.d
drop_in_file="${drop_in_dir}/http-proxy.conf"

sudo install -d -m 0755 "${drop_in_dir}"
sudo tee "${drop_in_file}" >/dev/null <<EOF
[Service]
Environment="HTTP_PROXY=${http_proxy}"
Environment="HTTPS_PROXY=${https_proxy}"
Environment="NO_PROXY=${no_proxy}"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
sudo systemctl show docker --property=Environment --no-pager

echo 'Docker daemon proxy configured. Verify with: docker pull ros:humble-ros-base'
