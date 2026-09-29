#!/usr/bin/env bash
# Read-only FCI network check. It sends ICMP only; it never enables FCI or commands motion.
set -euo pipefail

robot_ip="${FRANKA_ROBOT_IP:?FRANKA_ROBOT_IP is required}"
interface="${FRANKA_INTERFACE:?FRANKA_INTERFACE is required}"
count="${PING_COUNT:-20}"

if ! ip link show dev "${interface}" >/dev/null 2>&1; then
  echo "Interface does not exist: ${interface}" >&2
  exit 2
fi

carrier_file="/sys/class/net/${interface}/carrier"
if [[ -r "${carrier_file}" ]] && [[ "$(<"${carrier_file}")" != '1' ]]; then
  echo "No Ethernet carrier on ${interface}; connect and power the robot/control box first." >&2
  exit 3
fi

echo "--- ${interface} IPv4 address ---"
ip -4 address show dev "${interface}"
echo "--- route to robot ---"
route="$(ip route get "${robot_ip}")"
printf '%s\n' "${route}"

if ! grep -Eq "(^| )dev ${interface}( |$)" <<<"${route}"; then
  echo "Route to ${robot_ip} does not use ${interface}. Refusing to test another network." >&2
  exit 4
fi

echo "--- ICMP (${count} packets, 1200 bytes) ---"
ping -n -I "${interface}" -c "${count}" -s 1200 "${robot_ip}"
